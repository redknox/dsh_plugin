/**
 * Discovery tests (issue #10): probing /v1/models + /props, bounded TTL cache
 * with single-flight and cancellation, graceful degradation, configured
 * overrides taking precedence over discovered facts, and discovered
 * capabilities feeding listModels/resolveModel and the #9 routing policy.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { LlmError, type GenerateOptions } from '@deepseek-ai/dsh-llm';
import { LlamacppAdapter, type LlamaCppChatHandle } from '../src/adapter.ts';
import { resolveAdapterOptions } from '../src/config.ts';
import { EndpointDiscovery, mergeCapabilities } from '../src/discovery.ts';
import { collect, contentDelta } from './helpers.ts';

const fetchMock = vi.fn();

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

function stubFetch(handler: (url: string, init?: RequestInit) => Response | Promise<Response>): void {
  fetchMock.mockImplementation((url: string, init?: RequestInit) => handler(String(url), init));
  vi.stubGlobal('fetch', fetchMock);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

function okClient(): LlamaCppChatHandle {
  return { chat: vi.fn().mockImplementation(async function* () { yield contentDelta('ok'); yield contentDelta('', 'stop'); }) };
}

const baseOptions: GenerateOptions = { provider: 'llamacpp-local', model: 'qwen3', messages: [] };

describe('EndpointDiscovery probes', () => {
  it('reports healthy when /health answers 2xx and models are discovered', async () => {
    stubFetch((url) => (
      url.endsWith('/health')
        ? jsonResponse({ status: 'ok' })
        : url.endsWith('/v1/models')
          ? jsonResponse({ data: [{ id: 'm1' }] })
          : jsonResponse({})
    ));
    const result = await new EndpointDiscovery('http://a').discover();
    expect(result.healthy).toBe(true);
    expect(result.models.map((m) => m.id)).toEqual(['m1']);
    expect(result.error).toBeUndefined();
  });

  it('degrades gracefully when /health fails but /v1/models works (review regression)', async () => {
    stubFetch((url) => (
      url.endsWith('/health')
        ? new Response('not found', { status: 404 })
        : url.endsWith('/v1/models')
          ? jsonResponse({ data: [{ id: 'm1' }] })
          : jsonResponse({})
    ));
    const result = await new EndpointDiscovery('http://a').discover();
    expect(result.healthy).toBe(false);
    // Models discovery is unaffected by a missing /health.
    expect(result.models.map((m) => m.id)).toEqual(['m1']);
    expect(result.error).toBeUndefined();
  });

  it('treats a /health timeout as unhealthy without blocking models discovery', async () => {
    stubFetch((url, init) => (
      url.endsWith('/health')
        ? new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener('abort', () => reject(init.signal?.reason));
          })
        : url.endsWith('/v1/models')
          ? jsonResponse({ data: [{ id: 'm1' }] })
          : jsonResponse({})
    ));
    const result = await new EndpointDiscovery('http://a', { timeoutMs: 50 }).discover();
    expect(result.healthy).toBe(false);
    expect(result.models.map((m) => m.id)).toEqual(['m1']);
  });

  it('discovers model ids and context windows from /v1/models meta', async () => {
    stubFetch((url) => (
      url.endsWith('/v1/models')
        ? jsonResponse({ data: [{ id: '/models/qwen3.gguf', meta: { n_ctx: 204800, n_ctx_train: 262144 } }] })
        : jsonResponse({ model_alias: '/models/qwen3.gguf', total_slots: 1 })
    ));
    const result = await new EndpointDiscovery('http://a').discover();
    expect(result.error).toBeUndefined();
    expect(result.models).toEqual([{ id: '/models/qwen3.gguf', contextWindow: 204800 }]);
  });

  it('handles the llama.cpp/ollama-style models[] list and capabilities markers', async () => {
    stubFetch((url) => (
      url.endsWith('/v1/models')
        ? jsonResponse({ object: 'list', models: [{ model: 'm1', capabilities: ['completion', 'tools'] }], data: [{ id: 'm1' }] })
        : jsonResponse({})
    ));
    const result = await new EndpointDiscovery('http://a').discover();
    expect(result.models).toEqual([{ id: 'm1', supportsTools: true }]);
  });

  it('fills the context window from /props n_ctx for a single model without meta', async () => {
    stubFetch((url) => (
      url.endsWith('/v1/models')
        ? jsonResponse({ data: [{ id: 'm1' }] })
        : jsonResponse({ n_ctx: 8192 })
    ));
    const result = await new EndpointDiscovery('http://a').discover();
    expect(result.models).toEqual([{ id: 'm1', contextWindow: 8192 }]);
  });

  it('falls back to the /props loaded model when the model list is unavailable', async () => {
    stubFetch((url) => (
      url.endsWith('/v1/models') ? new Response('boom', { status: 500 }) : jsonResponse({ model_alias: 'fallback-model', n_ctx: 4096 })
    ));
    const result = await new EndpointDiscovery('http://a').discover();
    expect(result.models).toEqual([{ id: 'fallback-model', contextWindow: 4096 }]);
    expect(result.error).toContain('HTTP 500');
  });

  it('degrades gracefully when both probes fail', async () => {
    stubFetch(() => new Response('not found', { status: 404 }));
    const result = await new EndpointDiscovery('http://a').discover();
    expect(result.models).toEqual([]);
    expect(result.error).toContain('HTTP 404');
  });
});

describe('EndpointDiscovery caching and cancellation', () => {
  it('serves fresh results from the TTL cache and re-probes after invalidation', async () => {
    let calls = 0;
    stubFetch(() => {
      calls += 1;
      return jsonResponse({ data: [{ id: 'm1' }] });
    });
    const discovery = new EndpointDiscovery('http://a', { ttlMs: 60_000 });
    await discovery.discover();
    expect(calls).toBe(3); // /health + /v1/models + /props
    await discovery.discover(); // cached
    expect(calls).toBe(3);
    discovery.clear();
    await discovery.discover();
    expect(calls).toBe(6);
  });

  it('honors cancellation promptly', async () => {
    stubFetch((_url, init) => new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(init.signal?.reason));
    }));
    const discovery = new EndpointDiscovery('http://a');
    const controller = new AbortController();
    const pending = discovery.discover(controller.signal);
    setTimeout(() => controller.abort(), 10);
    const error = await pending.then(() => undefined, (e: unknown) => e);
    expect((error as LlmError).code).toBe('ABORTED');
  });
});

describe('mergeCapabilities precedence', () => {
  it('lets configured capabilities win per field and discovered fill gaps', () => {
    const merged = mergeCapabilities(
      { contextWindow: 1000, tools: false },
      { id: 'qwen3', contextWindow: 2000, supportsTools: true, supportsReasoning: true },
    );
    expect(merged).toEqual({ contextWindow: 1000, tools: false, reasoning: true });
  });
});

describe('discovery feeding the adapter', () => {
  it('listModels advertises discovered models when enabled, configured model otherwise', async () => {
    stubFetch((url) => (
      url.endsWith('/v1/models')
        ? jsonResponse({ data: [{ id: 'm-a' }, { id: 'm-b' }] })
        : jsonResponse({})
    ));
    const options = resolveAdapterOptions({ baseURL: 'http://a', model: 'qwen3', discovery: { enabled: true, ttlMs: 60_000 } });
    const adapter = new LlamacppAdapter({ options: () => options, resolveApiKey: async () => undefined, createClient: vi.fn() });
    expect((await adapter.listModels('llamacpp-local')).map((m) => m.id)).toEqual(['m-a', 'm-b']);

    const plain = resolveAdapterOptions({ baseURL: 'http://a', model: 'qwen3' });
    const plainAdapter = new LlamacppAdapter({ options: () => plain, resolveApiKey: async () => undefined, createClient: vi.fn() });
    expect((await plainAdapter.listModels('llamacpp-local')).map((m) => m.id)).toEqual(['qwen3']);
  });

  it('resolveModel surfaces discovered context and lets configured capabilities win', async () => {
    stubFetch((url) => (
      url.endsWith('/v1/models')
        ? jsonResponse({ data: [{ id: 'qwen3', meta: { n_ctx: 8192 } }] })
        : jsonResponse({})
    ));
    const discovered = resolveAdapterOptions({ baseURL: 'http://a', model: 'qwen3', discovery: { enabled: true, ttlMs: 60_000 } });
    const dAdapter = new LlamacppAdapter({ options: () => discovered, resolveApiKey: async () => undefined, createClient: vi.fn() });
    expect((await dAdapter.resolveModel('llamacpp-local', 'qwen3')).context).toEqual({ contextWindow: 8192 });

    const configured = resolveAdapterOptions({
      baseURL: 'http://a',
      endpoints: [{ url: 'http://a', capabilities: { contextWindow: 4096 } }],
      discovery: { enabled: true, ttlMs: 60_000 },
    });
    const cAdapter = new LlamacppAdapter({ options: () => configured, resolveApiKey: async () => undefined, createClient: vi.fn() });
    expect((await cAdapter.resolveModel('llamacpp-local', 'qwen3')).context).toEqual({ contextWindow: 4096 });
  });

  it('listModels passes auth to the discovery probes (authenticated server, review regression)', async () => {
    const seen: string[] = [];
    stubFetch((url, init) => {
      const headers = (init?.headers ?? {}) as Record<string, string>;
      seen.push(`${url} => ${headers.authorization ?? 'no-auth'}`);
      if (url.endsWith('/v1/models')) return jsonResponse({ data: [{ id: 'm1', meta: { n_ctx: 8192 } }] });
      return jsonResponse({});
    });
    const options = resolveAdapterOptions({ baseURL: 'http://a', model: 'qwen3', apiKeyEnv: 'LLAMA_API_TOKEN', discovery: { enabled: true, ttlMs: 60_000 } });
    const adapter = new LlamacppAdapter({
      options: () => options,
      resolveApiKey: async () => 'sekrit',
      createClient: vi.fn(),
    });
    const models = await adapter.listModels('llamacpp-local');
    expect(models.map((m) => m.id)).toEqual(['m1']);
    // Every probe (health + models + props) carries the resolved auth.
    expect(seen).toHaveLength(3);
    expect(seen.every((line) => line.includes('Bearer sekrit'))).toBe(true);
    // The discovered context window is also surfaced via resolveModel.
    const info = await adapter.resolveModel('llamacpp-local', 'm1');
    expect(info.context).toEqual({ contextWindow: 8192 });
  });

  it('routes using freshly cached discovered capabilities (never blocks on probes)', async () => {
    stubFetch((url) => {
      if (url.includes('http://a') && url.endsWith('/v1/models')) return jsonResponse({ data: [{ id: 'qwen3', meta: { n_ctx: 512 } }] });
      if (url.includes('http://b') && url.endsWith('/v1/models')) return jsonResponse({ data: [{ id: 'qwen3', meta: { n_ctx: 65_536 } }] });
      return jsonResponse({});
    });
    const options = resolveAdapterOptions({
      baseURL: 'http://a',
      endpoints: ['http://a', 'http://b'],
      discovery: { enabled: true, ttlMs: 60_000 },
    });
    const createClient = vi.fn((_baseURL: string) => okClient());
    const adapter = new LlamacppAdapter({ options: () => options, resolveApiKey: async () => undefined, createClient });

    // Prime the discovery caches (listModels triggers probes).
    await adapter.listModels('llamacpp-local');

    // A ~2100-char prompt estimates >512 tokens: endpoint a is routed out by
    // its discovered context window; b (65k) receives the request.
    await collect(adapter.stream({
      ...baseOptions,
      messages: [{
        id: 'm' as never,
        role: 'user' as const,
        content: [{ type: 'text' as const, text: 'x'.repeat(2_100) }],
        source: { kind: 'user' as const },
      }],
    }));
    expect(createClient.mock.calls.map((c) => c[0])).toEqual(['http://b']);
  });
});

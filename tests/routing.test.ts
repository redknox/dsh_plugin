/**
 * Capability-aware routing tests (issue #9): eligible-endpoint selection from
 * request/model capabilities, deterministic ordering, explicit failure when
 * nothing is eligible, plain-list behavior identical to #7, the routing
 * telemetry event, and interaction with reliability fallback.
 */
import { describe, expect, it, vi } from 'vitest';
import { LlmError, type GenerateOptions } from '@deepseek-ai/dsh-llm';
import { LlamacppAdapter, type LlamaCppChatHandle } from '../src/adapter.ts';
import { resolveAdapterOptions } from '../src/config.ts';
import {
  CapabilityRoutingPolicy,
  deriveWorkload,
  routeEndpoints,
  type EndpointRoutingProfile,
  type RoutingPolicy,
  type RoutingRequest,
} from '../src/routing.ts';
import type { TelemetryEvent, TelemetrySink } from '../src/telemetry.ts';
import { NoopTelemetry } from '../src/telemetry.ts';
import { collect, contentDelta } from './helpers.ts';

const baseReq: RoutingRequest = { model: 'qwen3', toolsAvailable: false, reasoningEnabled: false };

function okClient(): LlamaCppChatHandle {
  return { chat: vi.fn().mockImplementation(async function* () { yield contentDelta('ok'); yield contentDelta('', 'stop'); }) };
}
function failClient(code: string): LlamaCppChatHandle {
  return { chat: vi.fn().mockImplementation(async function* () { throw new LlmError('boom', code); }) };
}

function recording(): { sink: () => TelemetrySink; events: TelemetryEvent[] } {
  const events: TelemetryEvent[] = [];
  return { sink: () => ({ emit: (event) => void events.push(event) }), events };
}

function adapterWith(
  config: Record<string, unknown>,
  clients: Record<string, LlamaCppChatHandle>,
  sink: () => TelemetrySink = () => NoopTelemetry,
) {
  const options = () => resolveAdapterOptions(config);
  const createClient = vi.fn((baseURL: string) => clients[baseURL] ?? clients['*'] ?? okClient());
  const adapter = new LlamacppAdapter({ options, resolveApiKey: async () => undefined, createClient, telemetry: sink });
  return { adapter, createClient };
}

const baseOptions: GenerateOptions = { provider: 'llamacpp-local', model: 'qwen3', messages: [] };

describe('routeEndpoints: plain list behaves as #7', () => {
  it('keeps every endpoint eligible in configuration order without metadata', () => {
    const decision = routeEndpoints(baseReq, [
      { baseURL: 'http://a' },
      { baseURL: 'http://b' },
      { baseURL: 'http://c' },
    ]);
    expect(decision.candidates).toEqual(['http://a', 'http://b', 'http://c']);
    expect(decision.rationale.join(' ')).toContain('candidates: http://a -> http://b -> http://c');
  });
});

describe('routeEndpoints: capability constraints', () => {
  it('routes by exact model compatibility first', () => {
    const decision = routeEndpoints(baseReq, [
      { baseURL: 'http://a', capabilities: { models: ['qwen3'] } },
      { baseURL: 'http://b', capabilities: { models: ['qwen2.5'] } },
      { baseURL: 'http://c' }, // no metadata: any model
    ]);
    expect(decision.candidates).toEqual(['http://a', 'http://c']);
    expect(decision.rationale.some((r) => r.includes('http://b: excluded (model "qwen3" not served)'))).toBe(true);
  });

  it('excludes endpoints whose context window cannot fit the prompt', () => {
    const req: RoutingRequest = { ...baseReq, estimatedPromptTokens: 40_000 };
    const decision = routeEndpoints(req, [
      { baseURL: 'http://small', capabilities: { contextWindow: 32_768 } },
      { baseURL: 'http://big', capabilities: { contextWindow: 65_536 } },
    ]);
    expect(decision.candidates).toEqual(['http://big']);
    expect(decision.rationale.some((r) => r.includes('http://small: excluded (prompt'))).toBe(true);
  });

  it('excludes endpoints without tool support when tools are requested', () => {
    const req: RoutingRequest = { ...baseReq, toolsAvailable: true };
    const decision = routeEndpoints(req, [
      { baseURL: 'http://no-tools', capabilities: { tools: false } },
      { baseURL: 'http://unknown', capabilities: { tools: true } },
    ]);
    expect(decision.candidates).toEqual(['http://unknown']);
  });

  it('excludes endpoints without reasoning support when reasoning is enabled', () => {
    const req: RoutingRequest = { ...baseReq, reasoningEnabled: true };
    const decision = routeEndpoints(req, [
      { baseURL: 'http://plain', capabilities: { reasoning: false } },
      { baseURL: 'http://thinker' },
    ]);
    expect(decision.candidates).toEqual(['http://thinker']);
  });

  it('fails explicitly when no configured endpoint satisfies mandatory capabilities', () => {
    const throwsCode = (fn: () => unknown, code: string): void => {
      try {
        fn();
      } catch (error) {
        expect((error as LlmError).code).toBe(code);
        return;
      }
      throw new Error(`expected an LlmError with code ${code}`);
    };
    throwsCode(() => routeEndpoints(baseReq, [
      { baseURL: 'http://a', capabilities: { models: ['other-model'] } },
      { baseURL: 'http://b', capabilities: { models: ['another'] } },
    ]), 'NO_ELIGIBLE_ENDPOINT');
    throwsCode(() => routeEndpoints({ ...baseReq, toolsAvailable: true }, [
      { baseURL: 'http://a', capabilities: { tools: false } },
    ]), 'NO_ELIGIBLE_ENDPOINT');
  });
});

describe('routeEndpoints: deterministic ordering', () => {
  it('prefers matching workload classes with stable configuration order', () => {
    const profiles: EndpointRoutingProfile[] = [
      { baseURL: 'http://a', capabilities: { workload: ['chat'] } },
      { baseURL: 'http://b' },
      { baseURL: 'http://c', capabilities: { workload: ['chat'] } },
    ];
    const req: RoutingRequest = { ...baseReq, workload: 'chat' };
    const first = routeEndpoints(req, profiles);
    const second = routeEndpoints(req, profiles);
    expect(first.candidates).toEqual(['http://a', 'http://c', 'http://b']);
    expect(second.candidates).toEqual(first.candidates);
  });

  it('keeps configuration order when no workload preference applies', () => {
    const decision = routeEndpoints({ ...baseReq, workload: 'code' }, [
      { baseURL: 'http://a', capabilities: { workload: ['chat'] } },
      { baseURL: 'http://b' },
    ]);
    expect(decision.candidates).toEqual(['http://a', 'http://b']);
  });

  it('derives the workload class from purpose and reasoning state', () => {
    expect(deriveWorkload(undefined, false)).toBe('chat');
    expect(deriveWorkload(undefined, true)).toBe('reasoning');
    expect(deriveWorkload('session-title', true)).toBe('title');
    expect(deriveWorkload('compaction', true)).toBe('compaction');
  });
});

describe('routing through the adapter', () => {
  it('never sends the request to an incompatible endpoint', async () => {
    const { adapter, createClient } = adapterWith({
      baseURL: 'http://a',
      endpoints: [
        { url: 'http://a', capabilities: { models: ['qwen3'], tools: false } },
        { url: 'http://b', capabilities: { models: ['qwen3'] } },
      ],
    }, { 'http://b': okClient() });
    await collect(adapter.stream({ ...baseOptions, tools: [{ name: 'get_time', description: 'd', parameters: { type: 'object' } }] }));
    // a is excluded (no tools); only b receives the request.
    expect(createClient.mock.calls.map((c) => c[0])).toEqual(['http://b']);
  });

  it('reliability fallback still works within the routed eligible set', async () => {
    const { adapter, createClient } = adapterWith({
      baseURL: 'http://a',
      endpoints: [
        { url: 'http://a', capabilities: { models: ['qwen3'] } },
        { url: 'http://b', capabilities: { models: ['qwen3'] } },
        { url: 'http://c', capabilities: { models: ['other'] } },
      ],
      retryPolicy: { mode: 'normal', maxRetries: 2 },
    }, { 'http://a': failClient('TRANSPORT'), 'http://b': okClient() });
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    expect(chunks.filter((c) => c.type === 'text-delta').map((c) => c.text)).toEqual(['ok']);
    // c is routed out; fallback goes a -> b.
    expect(createClient.mock.calls.map((c) => c[0])).toEqual(['http://a', 'http://b']);
  });

  it('emits the routing decision through the telemetry seam', async () => {
    const { sink, events } = recording();
    const { adapter } = adapterWith({
      baseURL: 'http://a',
      endpoints: [
        { url: 'http://a', capabilities: { models: ['other'] } },
        { url: 'http://b', capabilities: { models: ['qwen3'] } },
      ],
    }, { 'http://b': okClient() }, sink);
    await collect(adapter.stream(baseOptions));
    const routing = events.find((e) => e.type === 'routing');
    expect(routing).toBeDefined();
    if (routing?.type !== 'routing') return;
    expect(routing.decision.candidates).toEqual(['http://b']);
    expect(routing.decision.rationale.some((r) => r.includes('http://a: excluded'))).toBe(true);
  });

  it('honors an injected custom RoutingPolicy without changing adapter code (review regression)', async () => {
    // A policy that reverses the configured order; the adapter must follow it.
    const customPolicy: RoutingPolicy = {
      route(request, profiles) {
        return {
          candidates: [...profiles].reverse().map((p) => p.baseURL),
          rationale: [`custom reversed order for ${request.model}`],
        };
      },
    };
    const options = resolveAdapterOptions({
      endpoints: ['http://a', 'http://b'],
      retryPolicy: { mode: 'normal', maxRetries: 2 },
    });
    const createClient = vi.fn((baseURL: string) => (
      baseURL === 'http://b' ? failClient('TRANSPORT') : okClient()
    ));
    const adapter = new LlamacppAdapter({
      options: () => options,
      resolveApiKey: async () => undefined,
      createClient,
      routing: customPolicy,
    });
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    // Custom order [b, a]: b fails, reliability falls back to a.
    expect(createClient.mock.calls.map((c) => c[0])).toEqual(['http://b', 'http://a']);
    expect(chunks.some((c) => c.type === 'text-delta')).toBe(true);
  });
});

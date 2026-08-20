/**
 * Observability tests for issue #8: one request is traced from adapter entry
 * through endpoint selection to the terminal result; metrics (TTFT, latency,
 * chunks, finish, usage, tools) are recorded; retry/fallback and reasoning
 * decisions are observable in structured form; telemetry can be disabled;
 * and no prompt content, tool arguments, or secrets ever appear in events.
 */
import { describe, expect, it, vi } from 'vitest';
import { CallId, LlmError, type GenerateOptions, type StreamChunk } from '@deepseek-ai/dsh-llm';
import { LlamacppAdapter, type LlamaCppChatHandle } from '../src/adapter.ts';
import { resolveAdapterOptions } from '../src/config.ts';
import {
  NoopTelemetry,
  logTelemetry,
  type TelemetryEvent,
  type TelemetrySink,
} from '../src/telemetry.ts';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from '../src/protocol.ts';
import { chunk as wireChunk, collect, contentDelta, usageChunk } from './helpers.ts';

function wire(partial: Parameters<typeof wireChunk>[0]): LlamaCppChatCompletionChunk {
  return wireChunk(partial);
}

function okClient(texts: string[] = ['ok']): LlamaCppChatHandle {
  return { chat: vi.fn().mockImplementation(async function* () { for (const t of texts) yield contentDelta(t); yield contentDelta('', 'stop'); }) };
}

function failClient(code: string): LlamaCppChatHandle {
  return { chat: vi.fn().mockImplementation(async function* () { throw new LlmError('boom', code); }) };
}

/** Build an adapter whose clients are scripted per endpoint URL. */
function harness(
  config: Record<string, unknown>,
  clients: Record<string, LlamaCppChatHandle>,
  telemetry: () => TelemetrySink = () => NoopTelemetry,
) {
  const options = () => resolveAdapterOptions(config);
  const createClient = vi.fn((baseURL: string) => clients[baseURL] ?? clients['*'] ?? okClient());
  const adapter = new LlamacppAdapter({
    options,
    resolveApiKey: async () => undefined,
    createClient,
    telemetry,
  });
  return { adapter, createClient };
}

function recording(): { sink: () => TelemetrySink; events: TelemetryEvent[] } {
  const events: TelemetryEvent[] = [];
  return { sink: () => ({ emit: (event) => void events.push(event) }), events };
}

const baseOptions: GenerateOptions = {
  provider: 'llamacpp-local',
  model: 'qwen3',
  messages: [],
};

function started(events: TelemetryEvent[]) {
  return events.find((e) => e.type === 'started');
}
function finished(events: TelemetryEvent[]) {
  return events.find((e) => e.type === 'finished');
}
function attempts(events: TelemetryEvent[]) {
  return events.filter((e) => e.type === 'attempt');
}

describe('telemetry on the success path', () => {
  it('traces started -> selected attempt -> finished with metrics', async () => {
    const { sink, events } = recording();
    const { adapter } = harness(
      { baseURL: 'http://127.0.0.1:8080' },
      { '*': okClient(['Hel', 'lo']) },
      sink,
    );
    // Inject usage + a tool schema so usage/tools are recorded.
    const clients = {
      '*': { chat: vi.fn().mockImplementation(async function* () {
        yield contentDelta('Hel');
        yield contentDelta('lo');
        yield contentDelta('', 'stop');
        yield usageChunk();
      }) },
    };
    const options = resolveAdapterOptions({ baseURL: 'http://127.0.0.1:8080' });
    const createClient = vi.fn((baseURL: string) => clients['*']);
    const adapter2 = new LlamacppAdapter({
      options: () => options,
      resolveApiKey: async () => undefined,
      createClient,
      telemetry: sink,
    });
    await collect(adapter2.stream({ ...baseOptions, tools: [{ name: 'get_time', description: 'd', parameters: { type: 'object' } }] }));

    expect(started(events)).toMatchObject({
      type: 'started',
      context: { model: 'qwen3', toolsAvailable: true },
    });
    expect(attempts(events)).toHaveLength(1);
    expect(attempts(events)[0]?.attempt).toMatchObject({ attempt: 1, baseURL: 'http://127.0.0.1:8080', outcome: 'selected' });

    const f = finished(events);
    expect(f).toBeDefined();
    if (f?.type !== 'finished') return;
    expect(f.outcome.endpoint).toBe('http://127.0.0.1:8080');
    expect(f.outcome.retryCount).toBe(0);
    expect(f.outcome.fallbackCount).toBe(0);
    expect(f.outcome.ttftMs).toBeGreaterThanOrEqual(0);
    expect(f.outcome.totalMs).toBeGreaterThanOrEqual(f.outcome.ttftMs ?? 0);
    expect(f.outcome.streamChunkCount).toBeGreaterThan(0);
    expect(f.outcome.finishReason).toEqual({ kind: 'stop' });
    expect(f.outcome.usage).toBeDefined();
    expect(f.outcome.usage?.outputTokens).toBe(5);
    void adapter;
  });

  it('records reasoning effort/budget as a structured decision and purpose in context', async () => {
    const { sink, events } = recording();
    const { adapter } = harness({ baseURL: 'http://127.0.0.1:8080', reasoning: { preset: 'xhigh' } }, { '*': okClient() }, sink);
    await collect(adapter.stream({ ...baseOptions, purpose: 'session-title' }));
    // session-title disables thinking: purpose recorded, no budget decision.
    expect(started(events)?.context).toMatchObject({ model: 'qwen3', purpose: 'session-title' });
    const titledReasoning = events.find((e) => e.type === 'reasoning');
    if (titledReasoning?.type !== 'reasoning') return;
    expect(titledReasoning.decision.enabled).toBe(false);
    expect(titledReasoning.decision.budgetTokens).toBeUndefined();

    const { sink: sink2, events: events2 } = recording();
    const { adapter: adapter2 } = harness({ baseURL: 'http://127.0.0.1:8080', reasoning: { preset: 'xhigh' } }, { '*': okClient() }, sink2);
    await collect(adapter2.stream(baseOptions));
    expect(started(events2)?.context).toMatchObject({ model: 'qwen3' });
    expect(started(events2)?.context.purpose).toBeUndefined();
    const decision = events2.find((e) => e.type === 'reasoning');
    if (decision?.type !== 'reasoning') return;
    expect(decision.decision).toMatchObject({ enabled: true, effort: 'xhigh', budgetTokens: 16384 });
  });
});

describe('telemetry on retry/fallback', () => {
  it('records a fallback attempt and fallbackCount', async () => {
    const { sink, events } = recording();
    const { adapter } = harness(
      { endpoints: ['http://a', 'http://b'], retryPolicy: { mode: 'normal', maxRetries: 2 } },
      { 'http://a': failClient('TRANSPORT'), 'http://b': okClient(['from b']) },
      sink,
    );
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    expect(chunks.filter((c) => c.type === 'text-delta').map((c) => c.text)).toEqual(['from b']);
    expect(attempts(events).map((e) => e.attempt)).toEqual([
      { attempt: 1, baseURL: 'http://a', outcome: 'selected' },
      { attempt: 2, baseURL: 'http://b', outcome: 'fallback', failureCode: 'TRANSPORT' },
    ]);
    const f = finished(events);
    if (f?.type !== 'finished') return;
    expect(f.outcome.endpoint).toBe('http://b');
    expect(f.outcome.retryCount).toBe(0);
    expect(f.outcome.fallbackCount).toBe(1);
  });

  it('records same-endpoint retries', async () => {
    const { sink, events } = recording();
    let calls = 0;
    const flaky: LlamaCppChatHandle = {
      chat: vi.fn().mockImplementation(async function* () {
        calls += 1;
        if (calls < 3) throw new LlmError('flaky', 'SERVER');
        yield contentDelta('recovered');
        yield contentDelta('', 'stop');
      }),
    };
    const { adapter } = harness(
      { baseURL: 'http://127.0.0.1:8080', retryPolicy: { mode: 'normal', maxRetries: 3, backoff: { initialDelayMs: 5, maxDelayMs: 20, jitterRatio: 0 } } },
      { '*': flaky },
      sink,
    );
    await collect(adapter.stream(baseOptions));
    expect(attempts(events).map((e) => e.attempt.outcome)).toEqual(['selected', 'retry', 'retry']);
    const f = finished(events);
    if (f?.type !== 'finished') return;
    expect(f.outcome.retryCount).toBe(2);
    expect(f.outcome.fallbackCount).toBe(0);
  });
});

describe('telemetry on failure paths', () => {
  it('traces a pre-transport credential failure with the same requestId (review regression)', async () => {
    const { sink, events } = recording();
    const options = resolveAdapterOptions({ baseURL: 'http://127.0.0.1:8080', apiKeyEnv: 'LLAMA_API_TOKEN' });
    const adapter = new LlamacppAdapter({
      options: () => options,
      resolveApiKey: async () => {
        throw new LlmError('llm-llamacpp: no API key for "LLAMA_API_TOKEN"', 'MISSING_CREDENTIAL');
      },
      createClient: vi.fn(),
      telemetry: sink,
    });
    const { error } = await collect(adapter.stream(baseOptions));
    expect((error as LlmError).code).toBe('MISSING_CREDENTIAL');

    const startedEvent = started(events);
    const finishedEvent = finished(events);
    expect(startedEvent).toBeDefined();
    expect(finishedEvent).toBeDefined();
    expect(startedEvent?.requestId).toBe(finishedEvent?.requestId);
    if (finishedEvent?.type !== 'finished') return;
    expect(finishedEvent.outcome.failureCode).toBe('MISSING_CREDENTIAL');
    expect(finishedEvent.outcome.endpoint).toBe('unresolved');
    expect(finishedEvent.outcome.totalMs).toBeGreaterThanOrEqual(0);
    expect(finishedEvent.outcome.retryCount).toBe(0);
    // No transport attempt happened for a pre-transport failure.
    expect(attempts(events)).toHaveLength(0);
  });

  it('records ABORTED as the terminal failure code on cancellation', async () => {
    const { sink, events } = recording();
    const hanging: LlamaCppChatHandle = {
      chat: vi.fn().mockImplementation(async function* (_request, opts?: { signal?: AbortSignal }) {
        await new Promise<never>((_resolve, reject) => {
          opts?.signal?.addEventListener('abort', () => reject(new LlmError('llama.cpp request aborted by caller', 'ABORTED')));
        });
        throw new LlmError('never', 'TRANSPORT');
      }),
    };
    const { adapter } = harness({ baseURL: 'http://127.0.0.1:8080', retryPolicy: { mode: 'normal', maxRetries: 5 } }, { '*': hanging }, sink);
    const controller = new AbortController();
    const iterator = adapter.stream({ ...baseOptions, signal: controller.signal })[Symbol.asyncIterator]();
    const pending = iterator.next();
    setTimeout(() => controller.abort(), 10);
    const result = await pending.then(() => undefined, (e: unknown) => e);
    expect((result as LlmError).code).toBe('ABORTED');
    const f = finished(events);
    if (f?.type !== 'finished') return;
    expect(f.outcome.failureCode).toBe('ABORTED');
  });

  it('records TIMEOUT as the terminal failure code', async () => {
    const { sink, events } = recording();
    const { adapter } = harness(
      { baseURL: 'http://127.0.0.1:8080', retryPolicy: { mode: 'normal', maxRetries: 0 } },
      { '*': failClient('TIMEOUT') },
      sink,
    );
    const { error } = await collect(adapter.stream(baseOptions));
    expect((error as LlmError).code).toBe('TIMEOUT');
    const f = finished(events);
    if (f?.type !== 'finished') return;
    expect(f.outcome.failureCode).toBe('TIMEOUT');
    expect(f.outcome.streamChunkCount).toBe(0);
  });
});

describe('telemetry on tool-call paths', () => {
  it('records the streamed tool-call count without executing tools', async () => {
    const { sink, events } = recording();
    const toolCalls: LlamaCppChatHandle = {
      chat: vi.fn().mockImplementation(async function* () {
        yield wire({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_a', function: { name: 'get_time', arguments: '{}' } }] }, finish_reason: null }] });
        yield wire({ choices: [{ index: 0, delta: { tool_calls: [{ index: 1, id: 'call_b', function: { name: 'echo', arguments: '{"text":"hi"}' } }] }, finish_reason: null }] });
        yield wire({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] });
      }),
    };
    const { adapter } = harness({ baseURL: 'http://127.0.0.1:8080' }, { '*': toolCalls }, sink);
    const chunks: StreamChunk[] = [];
    for await (const c of adapter.stream(baseOptions)) chunks.push(c);
    void CallId;
    const f = finished(events);
    if (f?.type !== 'finished') return;
    expect(f.outcome.toolCallCount).toBe(2);
    expect(f.outcome.finishReason).toEqual({ kind: 'tool-calls' });
    expect(chunks.some((c) => c.type === 'block-end' && c.block.type === 'tool-call')).toBe(true);
  });
});

describe('telemetry disable and privacy', () => {
  it('emits nothing when the sink is Noop, and behavior is unchanged', async () => {
    const { sink, events } = recording();
    const { adapter } = harness({ baseURL: 'http://127.0.0.1:8080' }, { '*': okClient(['fine']) }, () => NoopTelemetry);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    expect(chunks.filter((c) => c.type === 'text-delta').map((c) => c.text)).toEqual(['fine']);
    expect(events).toHaveLength(0);
    void sink;
  });

  it('never emits prompt content, tool arguments, or secrets', async () => {
    const { sink, events } = recording();
    const prompt = 'S3CR3T-PROMPT-CONTENT';
    const args = '{"password":"S3CR3T-ARGS"}';
    const key = 'sk-S3CR3T-KEY-VALUE';
    const client: LlamaCppChatHandle = {
      chat: vi.fn().mockImplementation(async function* () {
        yield wire({ choices: [{ index: 0, delta: { reasoning_content: 'think' }, finish_reason: null }] });
        yield wire({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_1', function: { name: 'echo', arguments: args } }] }, finish_reason: null }] });
        yield wire({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] });
      }),
    };
    const options = resolveAdapterOptions({ baseURL: 'http://127.0.0.1:8080', apiKeyEnv: 'LLAMA_API_TOKEN' });
    const createClient = vi.fn(() => client);
    const adapter = new LlamacppAdapter({
      options: () => options,
      resolveApiKey: async () => key,
      createClient,
      telemetry: sink,
    });
    await collect(adapter.stream({
      ...baseOptions,
      messages: [{
        id: 'm1' as never,
        role: 'user' as const,
        content: [{ type: 'text' as const, text: prompt }],
        source: { kind: 'user' as const },
      }],
    }));
    const serialized = JSON.stringify(events);
    expect(serialized).not.toContain(prompt);
    expect(serialized).not.toContain(args);
    expect(serialized).not.toContain(key);
    // Event shape stays structured and content-free.
    for (const event of events) {
      const keys = JSON.stringify(event);
      expect(keys).not.toMatch(/"content"|"arguments"/);
    }
  });

  it('log sink emits one JSON line per event', () => {
    const lines: unknown[][] = [];
    const sink = logTelemetry({ debug: (message, ...rest) => void lines.push([message, ...rest]) });
    sink.emit({ type: 'started', requestId: 'llm-x', at: 1, context: { model: 'qwen3', toolsAvailable: false } });
    expect(lines).toHaveLength(1);
    expect(String(lines[0]?.[1])).toContain('"type":"started"');
  });
});

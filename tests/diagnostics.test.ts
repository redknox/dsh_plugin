/**
 * Diagnostics tests (issue #12): the bounded store consumes #8 telemetry
 * events into counters and rolling windows, reflects #7 endpoint health, and
 * emits machine-readable snapshots plus a human-readable render — all
 * content-free, bounded, and deterministic.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DiagnosticsStore, MAX_LATENCY_WINDOW, MAX_RECENT_FAILURES, MAX_RECENT_ROUTING, renderDiagnostics, requestsPerMinute } from '../src/diagnostics.ts';
import { EndpointPool } from '../src/reliability.ts';
import type { RequestOutcome, TelemetryEvent } from '../src/telemetry.ts';
import type { ResolvedRetryPolicy } from '@deepseek-ai/dsh-llm';
import { LlamacppAdapter } from '../src/adapter.ts';
import { resolveAdapterOptions } from '../src/config.ts';

const fetchMock = vi.fn();

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

function stubFetch(handler: (url: string) => Response): void {
  fetchMock.mockImplementation((url: string) => handler(String(url)));
  vi.stubGlobal('fetch', fetchMock);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

function started(requestId: string, tools = false): TelemetryEvent {
  return { type: 'started', requestId, at: 1, context: { model: 'qwen3', toolsAvailable: tools } };
}
function reasoning(requestId: string, enabled = true): TelemetryEvent {
  return { type: 'reasoning', requestId, at: 2, decision: { enabled, effort: 'medium', budgetTokens: 4096 } };
}
function attempt(requestId: string, outcome: 'selected' | 'retry' | 'fallback', baseURL: string, failureCode?: string): TelemetryEvent {
  return { type: 'attempt', requestId, at: 3, attempt: { attempt: 1, baseURL, outcome, ...(failureCode !== undefined ? { failureCode } : {}) } };
}
function routing(requestId: string, candidates: string[], rationale: string[]): TelemetryEvent {
  return { type: 'routing', requestId, at: 4, decision: { candidates, rationale } };
}
function finished(requestId: string, outcome: Partial<RequestOutcome> & { endpoint: string; totalMs: number }): TelemetryEvent {
  return { type: 'finished', requestId, at: 5, outcome: { retryCount: 0, fallbackCount: 0, streamChunkCount: 1, ...outcome } };
}

describe('DiagnosticsStore', () => {
  it('reports an empty state', () => {
    const store = new DiagnosticsStore();
    const snapshot = store.snapshot();
    expect(snapshot.requests.total).toBe(0);
    expect(snapshot.endpoints).toEqual([]);
    expect(snapshot.latency.ttftMs.count).toBe(0);
    expect(snapshot.latency.totalMs.count).toBe(0);
    expect(snapshot.recentFailures).toEqual([]);
    expect(snapshot.recentRouting).toEqual([]);
  });

  it('aggregates request counters from a realistic event sequence', () => {
    const store = new DiagnosticsStore();
    store.emit(started('r1', true));
    store.emit(reasoning('r1'));
    store.emit(attempt('r1', 'selected', 'http://a'));
    store.emit(finished('r1', { endpoint: 'http://a', totalMs: 100, ttftMs: 30, streamChunkCount: 5, toolCallCount: 2, usage: { inputTokens: 10, outputTokens: 5, reasoningTokens: 40 }, finishReason: { kind: 'tool-calls' } }));

    store.emit(started('r2'));
    store.emit(attempt('r2', 'selected', 'http://a'));
    store.emit(attempt('r2', 'retry', 'http://a', 'SERVER'));
    store.emit(finished('r2', { endpoint: 'http://a', totalMs: 200, failureCode: 'SERVER' }));

    store.emit(started('r3'));
    store.emit(attempt('r3', 'selected', 'http://b'));
    store.emit(attempt('r3', 'fallback', 'http://b', 'TIMEOUT'));
    store.emit(finished('r3', { endpoint: 'http://b', totalMs: 300, failureCode: 'TIMEOUT' }));

    const snapshot = store.snapshot();
    expect(snapshot.requests.total).toBe(3);
    expect(snapshot.requests.success).toBe(1);
    expect(snapshot.requests.failure).toBe(1);
    expect(snapshot.requests.timeout).toBe(1);
    expect(snapshot.requests.retries).toBe(1);
    expect(snapshot.requests.fallbacks).toBe(1);
    expect(snapshot.requests.toolCalls).toBe(2);
    expect(snapshot.requests.requestsWithTools).toBe(1);
    expect(snapshot.requests.requestsWithReasoning).toBe(1);
    expect(snapshot.requests.reasoningTokensTotal).toBe(40);
    expect(snapshot.requests.byFailureCode).toEqual({ SERVER: 1, TIMEOUT: 1 });
    expect(snapshot.requests.byFinishReason).toEqual({ 'tool-calls': 1 });
    expect(snapshot.requests.byEndpoint).toEqual({ 'http://a': 2, 'http://b': 1 });
    expect(snapshot.latency.ttftMs).toEqual({ count: 1, avgMs: 30, minMs: 30, maxMs: 30 });
    expect(snapshot.latency.totalMs).toEqual({ count: 3, avgMs: 200, minMs: 100, maxMs: 300 });
  });

  it('bounded rolling windows drop the oldest samples', () => {
    const store = new DiagnosticsStore();
    for (let i = 0; i < MAX_LATENCY_WINDOW + 10; i += 1) {
      store.emit(finished(`r${i}`, { endpoint: 'http://a', totalMs: i, ttftMs: i }));
    }
    const snapshot = store.snapshot();
    expect(snapshot.latency.ttftMs.count).toBe(MAX_LATENCY_WINDOW);
    expect(snapshot.latency.totalMs.count).toBe(MAX_LATENCY_WINDOW);
    // The oldest 10 samples (0..9) dropped; min is now 10.
    expect(snapshot.latency.totalMs.minMs).toBe(10);
  });

  it('bounded retention for recent routing decisions and failures', () => {
    const store = new DiagnosticsStore();
    for (let i = 0; i < MAX_RECENT_FAILURES + 5; i += 1) {
      store.emit(routing(`r${i}`, ['http://a'], ['candidates']));
      store.emit(finished(`f${i}`, { endpoint: 'http://a', totalMs: 1, failureCode: 'SERVER' }));
    }
    const snapshot = store.snapshot();
    expect(snapshot.recentRouting.length).toBe(MAX_RECENT_ROUTING);
    expect(snapshot.recentFailures.length).toBe(MAX_RECENT_FAILURES);
  });

  it('reflects endpoint health/backoff state from the reliability pool', () => {
    const store = new DiagnosticsStore();
    store.emit(finished('r1', { endpoint: 'http://a', totalMs: 10, failureCode: 'TRANSPORT' }));
    const pool = new EndpointPool();
    const policy: ResolvedRetryPolicy = { mode: 'normal', maxRetries: 2, retryableCodes: ['TRANSPORT'], initialDelayMs: 10_000, maxDelayMs: 10_000, jitterRatio: 0 };
    pool.recordFailure('http://a', policy);
    const snapshot = store.snapshot(pool, ['http://a', 'http://b']);
    const [a, b] = snapshot.endpoints;
    expect(a).toMatchObject({ baseURL: 'http://a', inBackoff: true, healthy: false, consecutiveFailures: 1, requests: 1 });
    expect(b).toMatchObject({ baseURL: 'http://b', inBackoff: false, healthy: true, requests: 0 });
  });

  it('never retains prompt content, tool arguments, or secrets', () => {
    const store = new DiagnosticsStore();
    const secretPrompt = 'S3CR3T-PROMPT';
    const secretKey = 'sk-S3CR3T-KEY';
    store.emit(started('r1', true));
    store.emit(finished('r1', { endpoint: 'http://a', totalMs: 10, toolCallCount: 1, usage: { inputTokens: 1, outputTokens: 1 } }));
    const serialized = JSON.stringify(store.snapshot());
    expect(serialized).not.toContain(secretPrompt);
    expect(serialized).not.toContain(secretKey);
    expect(serialized).not.toContain('"arguments"');
    expect(serialized).not.toContain('"content"');
  });

  it('renders a human-readable block for local operations', () => {
    const store = new DiagnosticsStore();
    store.emit(started('r1'));
    store.emit(attempt('r1', 'selected', 'http://a'));
    store.emit(finished('r1', { endpoint: 'http://a', totalMs: 150, ttftMs: 40 }));
    const text = store.render(undefined, ['http://a'], [{ id: 'qwen3', source: 'configured' }]);
    expect(text).toContain('llm-llamacpp diagnostics');
    expect(text).toContain('http://a');
    expect(text).toContain('requests: 1 total (1 ok');
    expect(text).toContain('total avg 150ms');
    expect(text).toContain('qwen3 [configured]');
  });

  it('computes a deterministic request rate over the bounded window (review regression)', () => {
    const store = new DiagnosticsStore();
    // Started events carry fixed `at: 1`; three starts within the 60s window
    // ending at a fixed `now` give a deterministic 3/min.
    store.emit(started('r1'));
    store.emit(started('r2'));
    store.emit(started('r3'));
    const snapshot = store.snapshot(undefined, [], [], 60_001);
    expect(snapshot.requests.requestsPerMinute).toBe(3);
    // Extrapolation over a shorter span: 3 starts in 30s -> 6/min.
    expect(requestsPerMinute([0, 10_000, 20_000], 30_000)).toBe(6);
    // Empty window and stale-outside-window starts contribute nothing.
    expect(requestsPerMinute([], 60_000)).toBe(0);
    expect(requestsPerMinute([0], 120_000)).toBe(0);
    // A second snapshot at a different fixed `now` drops the window.
    expect(store.snapshot(undefined, [], [], 120_000).requests.requestsPerMinute).toBe(0);
  });

  it('aggregates reasoning effort and budget from reasoning events (review regression)', () => {
    const store = new DiagnosticsStore();
    store.emit(reasoning('r1', true)); // medium, budget 4096
    store.emit(finished('r1', { endpoint: 'http://a', totalMs: 10, usage: { inputTokens: 1, outputTokens: 1, reasoningTokens: 500 } }));
    store.emit(started('r2'));
    store.emit(reasoning('r2', true)); // medium again
    store.emit(finished('r2', { endpoint: 'http://a', totalMs: 20, usage: { inputTokens: 1, outputTokens: 1, reasoningTokens: 700 } }));
    store.emit(started('r3'));
    store.emit(reasoning('r3', true)); // medium again
    store.emit(finished('r3', { endpoint: 'http://a', totalMs: 30, usage: { inputTokens: 1, outputTokens: 1, reasoningTokens: 900 } }));
    const snapshot = store.snapshot();
    expect(snapshot.requests.requestsWithReasoning).toBe(3);
    expect(snapshot.requests.reasoningByEffort).toEqual({ medium: 3 });
    expect(snapshot.requests.reasoningTokensTotal).toBe(2100);
    expect(snapshot.requests.reasoningBudgetTokens).toEqual({ count: 3, avgMs: 4096, minMs: 4096, maxMs: 4096 });
  });
});

describe('renderDiagnostics', () => {
  it('renders an empty snapshot without throwing', () => {
    const text = renderDiagnostics({
      at: 0,
      uptimeMs: 0,
      endpoints: [],
      models: [],
      requests: { total: 0, success: 0, failure: 0, timeout: 0, aborted: 0, retries: 0, fallbacks: 0, toolCalls: 0, requestsWithTools: 0, requestsWithReasoning: 0, reasoningTokensTotal: 0, requestsPerMinute: 0, reasoningByEffort: {}, reasoningBudgetTokens: { count: 0, avgMs: 0 }, byFailureCode: {}, byFinishReason: {}, byEndpoint: {} },
      latency: { ttftMs: { count: 0, avgMs: 0 }, totalMs: { count: 0, avgMs: 0 } },
      recentRouting: [],
      recentFailures: [],
    });
    expect(text).toContain('llm-llamacpp diagnostics');
  });
});

describe('diagnostic model facts', () => {
  it('exposes structured capabilities with configured overrides authoritative (review regression)', async () => {
    stubFetch((url) => (
      url.endsWith('/v1/models')
        ? jsonResponse({ data: [
            { id: 'qwen3', meta: { n_ctx: 8192 } },
            { id: 'other-model', meta: { n_ctx: 16_384 } },
          ] })
        : jsonResponse({})
    ));
    const options = resolveAdapterOptions({
      baseURL: 'http://a',
      endpoints: [{ url: 'http://a', capabilities: { models: ['qwen3'], contextWindow: 4096, tools: false } }],
      discovery: { enabled: true, ttlMs: 60_000 },
    });
    const adapter = new LlamacppAdapter({ options: () => options, resolveApiKey: async () => undefined, createClient: vi.fn() });
    await adapter.listModels('llamacpp-local'); // primes the discovery cache

    const models = adapter.diagnosticModels();
    const qwen3 = models.find((m) => m.id === 'qwen3');
    // Configured facts win: discovered n_ctx 8192 does NOT override 4096.
    expect(qwen3).toMatchObject({ id: 'qwen3', source: 'configured', contextWindow: 4096, supportsTools: false });
    // A purely discovered model keeps discovered facts with its source.
    const other = models.find((m) => m.id === 'other-model');
    expect(other).toMatchObject({ id: 'other-model', source: 'discovered', contextWindow: 16_384 });
  });
});

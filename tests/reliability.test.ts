/**
 * Reliability tests for issue #7: ordered fallback, bounded retry with
 * backoff, hard timeouts, health state, cancellation safety, and structured
 * logs. All endpoints are scripted fakes; no HTTP happens here.
 */
import { describe, expect, it, vi } from 'vitest';
import { LlmError, type ResolvedRetryPolicy } from '@deepseek-ai/dsh-llm';
import {
  EndpointPool,
  streamReliably,
  type ReliableChatHandle,
  type ReliabilityEndpoint,
} from '../src/reliability.ts';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from '../src/protocol.ts';

const request: LlamaCppChatCompletionRequest = {
  model: 'qwen3',
  messages: [{ role: 'user', content: 'hi' }],
  stream: true,
};

function chunk(content: string): LlamaCppChatCompletionChunk {
  return { id: 'c', model: 'qwen3', choices: [{ index: 0, delta: { content }, finish_reason: null }] };
}

/** A bounded normal policy with tiny deterministic backoff. */
function policy(overrides: Partial<ResolvedRetryPolicy> = {}): ResolvedRetryPolicy {
  return {
    mode: 'normal',
    maxRetries: 2,
    retryableCodes: ['TRANSPORT', 'SERVER', 'TIMEOUT', 'RATE_LIMIT'],
    initialDelayMs: 10,
    maxDelayMs: 50,
    jitterRatio: 0,
    ...overrides,
  } as ResolvedRetryPolicy;
}

function failClient(code: string, message = 'boom'): ReliableChatHandle {
  return { chat: vi.fn().mockImplementation(async function* () { throw new LlmError(message, code); }) };
}

function okClient(texts: string[] = ['ok']): ReliableChatHandle {
  return { chat: vi.fn().mockImplementation(async function* () { for (const t of texts) yield chunk(t); }) };
}

function midStreamFail(): ReliableChatHandle {
  return {
    chat: vi.fn().mockImplementation(async function* () {
      yield chunk('partial');
      throw new LlmError('connection reset mid-stream', 'TRANSPORT');
    }),
  };
}

async function collect(stream: AsyncIterable<LlamaCppChatCompletionChunk>): Promise<{
  texts: string[];
  error?: unknown;
}> {
  const texts: string[] = [];
  try {
    for await (const c of stream) {
      for (const choice of c.choices) if (choice.delta.content) texts.push(choice.delta.content);
    }
    return { texts };
  } catch (error) {
    return { texts, error };
  }
}

function endpoints(...urls: string[]): ReliabilityEndpoint[] {
  return urls.map((baseURL) => ({ baseURL }));
}

describe('streamReliably: ordered fallback', () => {
  it('falls back from a failed primary to a healthy secondary before output', async () => {
    const primary = failClient('TRANSPORT');
    const secondary = okClient(['from secondary']);
    const createClient = vi.fn((baseURL: string) => (baseURL === 'http://a' ? primary : secondary));
    const { texts, error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a', 'http://b'),
      retryPolicy: policy({ maxRetries: 2 }),
      streamIdleTimeoutMs: 1000,
      createClient,
    }));
    expect(error).toBeUndefined();
    expect(texts).toEqual(['from secondary']);
    expect(createClient.mock.calls.map((c) => c[0])).toEqual(['http://a', 'http://b']);
  });

  it('ordered fallback with 3+ endpoints never skips a healthy secondary', async () => {
    // A fails and enters backoff; the next attempt must select B (the first
    // remaining healthy endpoint in configuration order), not C.
    const a = failClient('TRANSPORT');
    const b = okClient(['from B']);
    const c = okClient(['never C']);
    const createClient = vi.fn((baseURL: string) => (
      baseURL === 'http://a' ? a : baseURL === 'http://b' ? b : c
    ));
    const { texts, error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a', 'http://b', 'http://c'),
      retryPolicy: policy({ maxRetries: 2 }),
      streamIdleTimeoutMs: 1000,
      createClient,
    }));
    expect(error).toBeUndefined();
    expect(texts).toEqual(['from B']);
    expect(createClient.mock.calls.map((call) => call[0])).toEqual(['http://a', 'http://b']);
  });

  it('selects the first remaining healthy candidate when A and B are backing off', async () => {
    const pool = new EndpointPool();
    const longBackoff = policy({ initialDelayMs: 10_000, maxDelayMs: 10_000 });
    pool.recordFailure('http://a', longBackoff);
    pool.recordFailure('http://b', longBackoff);
    const a = okClient(['never A']);
    const b = okClient(['never B']);
    const c = okClient(['from C']);
    const createClient = vi.fn((baseURL: string) => (
      baseURL === 'http://a' ? a : baseURL === 'http://b' ? b : c
    ));
    const { texts, error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a', 'http://b', 'http://c'),
      retryPolicy: policy({ maxRetries: 2 }),
      streamIdleTimeoutMs: 1000,
      createClient,
      pool,
    }));
    expect(error).toBeUndefined();
    expect(texts).toEqual(['from C']);
    expect(createClient.mock.calls.map((call) => call[0])).toEqual(['http://c']);
  });

  it('a single-endpoint success path behaves exactly as before (one attempt)', async () => {
    const client = okClient(['hello']);
    const createClient = vi.fn(() => client);
    const { texts, error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a'),
      retryPolicy: policy(),
      streamIdleTimeoutMs: 1000,
      createClient,
    }));
    expect(error).toBeUndefined();
    expect(texts).toEqual(['hello']);
    expect(createClient).toHaveBeenCalledTimes(1);
  });

  it('never falls back after user-visible streamed output has begun', async () => {
    const primary = midStreamFail();
    const secondary = okClient(['never reached']);
    const createClient = vi.fn((baseURL: string) => (baseURL === 'http://a' ? primary : secondary));
    const { texts, error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a', 'http://b'),
      retryPolicy: policy({ maxRetries: 2 }),
      streamIdleTimeoutMs: 1000,
      createClient,
    }));
    expect(texts).toEqual(['partial']);
    expect((error as LlmError).code).toBe('TRANSPORT');
    expect(createClient).toHaveBeenCalledTimes(1);
  });
});

describe('streamReliably: retry policy', () => {
  it('retries a retryable failure on the same endpoint up to maxRetries, then fails', async () => {
    const client = failClient('SERVER');
    const createClient = vi.fn(() => client);
    const { error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a'),
      retryPolicy: policy({ maxRetries: 2 }),
      streamIdleTimeoutMs: 1000,
      createClient,
    }));
    expect(createClient).toHaveBeenCalledTimes(3); // 1 initial + 2 retries
    expect((error as LlmError).code).toBe('SERVER');
  });

  it('does not retry non-retryable codes (e.g. AUTH)', async () => {
    const client = failClient('AUTH');
    const createClient = vi.fn(() => client);
    const { error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a'),
      retryPolicy: policy({ maxRetries: 5 }),
      streamIdleTimeoutMs: 1000,
      createClient,
    }));
    expect(createClient).toHaveBeenCalledTimes(1);
    expect((error as LlmError).code).toBe('AUTH');
  });

  it('never retries an explicitly aborted request', async () => {
    const client: ReliableChatHandle = {
      chat: vi.fn().mockImplementation(async function* (_request, opts?: { signal?: AbortSignal }) {
        await new Promise<never>((_resolve, reject) => {
          opts?.signal?.addEventListener('abort', () => {
            reject(new LlmError('llama.cpp request aborted by caller', 'ABORTED'));
          });
        });
        throw new LlmError('never', 'TRANSPORT');
      }),
    };
    const createClient = vi.fn(() => client);
    const controller = new AbortController();
    const iterator = streamReliably(request, {
      endpoints: endpoints('http://a', 'http://b'),
      retryPolicy: policy({ maxRetries: 5 }),
      streamIdleTimeoutMs: 1000,
      createClient,
      signal: controller.signal,
    })[Symbol.asyncIterator]();
    const pending = iterator.next();
    setTimeout(() => controller.abort(), 10);
    const result = await pending.then(() => undefined, (e: unknown) => e);
    // The abort surfaces as a terminal ABORTED rejection; never a retry or
    // fallback to the secondary endpoint.
    expect((result as LlmError).code).toBe('ABORTED');
    expect(createClient).toHaveBeenCalledTimes(1);
  });

  it('always mode retries every failure until success', async () => {
    let calls = 0;
    const client: ReliableChatHandle = {
      chat: vi.fn().mockImplementation(async function* () {
        calls += 1;
        if (calls < 3) throw new LlmError('flaky', 'TRANSPORT');
        yield chunk('recovered');
      }),
    };
    const createClient = vi.fn(() => client);
    const { texts, error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a'),
      retryPolicy: { mode: 'always', initialDelayMs: 5, maxDelayMs: 20, jitterRatio: 0 },
      streamIdleTimeoutMs: 1000,
      createClient,
    }));
    expect(error).toBeUndefined();
    expect(texts).toEqual(['recovered']);
    expect(calls).toBe(3);
  });
});

describe('streamReliably: health state', () => {
  it('skips an endpoint in backoff on the next request (persistent pool)', async () => {
    const pool = new EndpointPool();
    const primary = failClient('TRANSPORT');
    const secondary = okClient(['healthy']);
    const createClient = vi.fn((baseURL: string) => (baseURL === 'http://a' ? primary : secondary));
    const common = {
      endpoints: endpoints('http://a', 'http://b'),
      retryPolicy: policy({ maxRetries: 1, initialDelayMs: 10_000, maxDelayMs: 10_000 }),
      streamIdleTimeoutMs: 1000,
      createClient,
      pool,
    };

    // First request: primary fails; with maxRetries 1 the secondary is tried
    // immediately (no backoff wait yet) and succeeds.
    const first = await collect(streamReliably(request, common));
    expect(first.texts).toEqual(['healthy']);
    expect(createClient.mock.calls.map((c) => c[0])).toEqual(['http://a', 'http://b']);

    // Second request: primary is still in its 10s backoff, so the healthy
    // secondary is selected first.
    createClient.mockClear();
    const second = await collect(streamReliably(request, common));
    expect(second.texts).toEqual(['healthy']);
    expect(createClient.mock.calls.map((c) => c[0])).toEqual(['http://b']);
  });

  it('resets health after a success', () => {
    const pool = new EndpointPool();
    const backoff = policy({ initialDelayMs: 100, maxDelayMs: 1000 });
    expect(pool.recordFailure('http://a', backoff)).toBe(100);
    // A success clears the failure streak: the next failure is back to the
    // initial delay, not exponential.
    pool.recordSuccess('http://a');
    expect(pool.recordFailure('http://a', backoff)).toBe(100);
  });
});

describe('streamReliably: structured logs and timeouts', () => {
  it('logs endpoint/model/retry context on failures', async () => {
    const warn = vi.fn();
    const primary = failClient('TRANSPORT');
    const secondary = okClient();
    const createClient = vi.fn((baseURL: string) => (baseURL === 'http://a' ? primary : secondary));
    await collect(streamReliably(request, {
      endpoints: endpoints('http://a', 'http://b'),
      retryPolicy: policy({ maxRetries: 1 }),
      streamIdleTimeoutMs: 1000,
      createClient,
      logger: { debug: vi.fn(), warn },
    }));
    expect(warn).toHaveBeenCalledTimes(1);
    const log = warn.mock.calls[0]?.[0] as string;
    expect(log).toContain('http://a');
    expect(log).toContain('qwen3');
    expect(log).toContain('TRANSPORT');
    expect(log).toContain('attempt 1/2');
  });

  it('treats a request timeout as retryable and falls back', async () => {
    const timingOut = failClient('TIMEOUT');
    const recovered = okClient(['recovered']);
    const createClient = vi.fn((baseURL: string) => (baseURL === 'http://a' ? timingOut : recovered));
    const { texts, error } = await collect(streamReliably(request, {
      endpoints: endpoints('http://a', 'http://b'),
      retryPolicy: policy({ maxRetries: 1 }),
      streamIdleTimeoutMs: 1000,
      createClient,
    }));
    expect(error).toBeUndefined();
    expect(texts).toEqual(['recovered']);
  });
});

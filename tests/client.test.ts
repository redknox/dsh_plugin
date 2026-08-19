/**
 * Transport tests for issue #2: the llama.cpp client must stream SSE deltas
 * incrementally, honor cancellation, surface typed HTTP/protocol failures, and
 * probe health — all against mocked HTTP/SSE responses, with no Harness
 * runtime involved.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { LlmError } from '@deepseek-ai/dsh-llm';
import { LlamaCppClient, checkHealth, httpErrorCode, providerRetryAfterMs } from '../src/client.ts';
import { SSE_DONE } from '../src/protocol.ts';
import type { LlamaCppChatCompletionChunk } from '../src/protocol.ts';

/** Build a streamed `Response` from already-framed SSE text. */
function sseResponse(frames: string[], status = 200, headers: Record<string, string> = {}): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(new TextEncoder().encode(frame));
      controller.close();
    },
  });
  return new Response(stream, {
    status,
    headers: { 'content-type': 'text/event-stream', ...headers },
  });
}

/** Frame one JSON payload as an SSE data event. */
function data(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/** A standard content delta chunk. */
function contentChunk(content: string, finishReason: string | null = null): unknown {
  return {
    id: 'chatcmpl-1',
    model: 'qwen3',
    choices: [{ index: 0, delta: { content }, finish_reason: finishReason }],
  };
}

const fetchMock = vi.fn();

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

function stubFetch(response: Response | ((signal: AbortSignal) => Response | Promise<Response>)) {
  fetchMock.mockImplementation(async (_url: string | URL | Request, init?: RequestInit) => {
    if (typeof response === 'function') return response(init?.signal as AbortSignal);
    return response;
  });
  vi.stubGlobal('fetch', fetchMock);
}

describe('LlamaCppClient.chat', () => {
  it('streams textual deltas incrementally', async () => {
    stubFetch(
      sseResponse([
        data(contentChunk('Hel')),
        data(contentChunk('lo')),
        data(contentChunk('', 'stop')),
        `data: ${SSE_DONE}\n\n`,
      ]),
    );
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const texts: string[] = [];
    const reasons: (string | null)[] = [];
    for await (const chunk of client.chat({ model: 'qwen3', messages: [{ role: 'user', content: 'hi' }], stream: true })) {
      for (const choice of chunk.choices) {
        if (choice.delta.content) texts.push(choice.delta.content);
        reasons.push(choice.finish_reason);
      }
    }
    expect(texts).toEqual(['Hel', 'lo']);
    expect(reasons).toEqual([null, null, 'stop']);
  });

  it('POSTs to /v1/chat/completions with attribution and JSON body, forcing stream', async () => {
    stubFetch(sseResponse([`data: ${SSE_DONE}\n\n`]));
    const client = new LlamaCppClient('http://127.0.0.1:8080', {
      auth: { name: 'authorization', value: 'Bearer secret' },
      headers: { 'x-extra': '1' },
    });
    const request = {
      model: 'qwen3',
      messages: [{ role: 'user' as const, content: 'hi' as const }],
      stream: false as unknown as true, // deliberately wrong: client must force it
    };
    for await (const _chunk of client.chat(request)) void _chunk;

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8080/v1/chat/completions');
    expect(init.method).toBe('POST');
    const headers = init.headers as Record<string, string>;
    expect(headers['content-type']).toBe('application/json');
    expect(headers.accept).toBe('text/event-stream');
    expect(headers['user-agent']).toMatch(/^deepseek-harness\//);
    expect(headers.authorization).toBe('Bearer secret');
    expect(headers['x-extra']).toBe('1');
    const body = JSON.parse(init.body as string) as { stream: boolean };
    expect(body.stream).toBe(true);
  });

  it('yields the first chunk before the stream completes', async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(data(contentChunk('first'))));
        void gate.then(() => controller.close());
      },
    });
    stubFetch(new Response(stream, { status: 200 }));
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    const first = await Promise.race([
      iterator.next().then((r) => r.value as LlamaCppChatCompletionChunk | undefined),
      new Promise<'timeout'>((resolve) => setTimeout(() => resolve('timeout'), 200)),
    ]);
    expect(first).not.toBe('timeout');
    expect((first as LlamaCppChatCompletionChunk).choices[0]?.delta.content).toBe('first');
    release();
    await iterator.return?.();
  });

  it('maps non-2xx responses to typed errors with diagnostics', async () => {
    stubFetch(
      new Response(JSON.stringify({ error: { message: 'invalid api key' } }), {
        status: 401,
        headers: { 'content-type': 'application/json' },
      }),
    );
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    const error = await iterator.next().then(() => null, (e: unknown) => e) as LlmError;
    expect(error.code).toBe('AUTH');
    expect(error.message).toBe('invalid api key');
    expect(error.failure.status).toBe(401);
  });

  it('carries Retry-After and request id facts on rate limits', async () => {
    stubFetch(
      new Response(JSON.stringify({ error: { message: 'slow down' } }), {
        status: 429,
        headers: { 'content-type': 'application/json', 'retry-after': '2', 'x-request-id': 'req-123' },
      }),
    );
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    const error = await iterator.next().then(() => null, (e: unknown) => e) as LlmError;
    expect(error.code).toBe('RATE_LIMIT');
    expect(error.failure.status).toBe(429);
    expect(error.failure.providerRetryAfterMs).toBe(2000);
    expect(error.failure.requestId).toBe('req-123');
  });

  it('maps server errors and context-window wording', async () => {
    stubFetch(new Response('boom', { status: 500 }));
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    const serverError = await iterator.next().then(() => null, (e: unknown) => e) as LlmError;
    expect(serverError.code).toBe('SERVER');
    expect(serverError.failure.status).toBe(500);

    stubFetch(
      new Response(JSON.stringify({ error: { message: 'maximum context length exceeded' } }), {
        status: 400,
        headers: { 'content-type': 'application/json' },
      }),
    );
    const iterator2 = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    const contextError = await iterator2.next().then(() => null, (e: unknown) => e) as LlmError;
    expect(contextError.code).toBe('CONTEXT_WINDOW_EXCEEDED');
    expect(contextError.failure.status).toBe(400);
  });

  it('rejects malformed SSE payloads with MALFORMED_RESPONSE', async () => {
    stubFetch(sseResponse(['data: {not json}\n\n', `data: ${SSE_DONE}\n\n`]));
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    await expect(iterator.next()).rejects.toMatchObject({ code: 'MALFORMED_RESPONSE' });
  });

  it('rejects a stream that ends without [DONE] as STREAM_CLOSED', async () => {
    stubFetch(sseResponse([data(contentChunk('partial'))]));
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const texts: string[] = [];
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    await expect((async () => {
      for await (const chunk of { [Symbol.asyncIterator]: () => iterator } as unknown as AsyncIterable<LlamaCppChatCompletionChunk>) {
        for (const choice of chunk.choices) if (choice.delta.content) texts.push(choice.delta.content);
      }
    })()).rejects.toMatchObject({ code: 'STREAM_CLOSED' });
    expect(texts).toEqual(['partial']);
  });

  it('maps caller cancellation to ABORTED and aborts the fetch signal', async () => {
    const controller = new AbortController();
    let fetchSignal: AbortSignal | undefined;
    stubFetch((signal) => {
      fetchSignal = signal;
      return new Promise<Response>((_resolve, reject) => {
        signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      });
    });
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true }, { signal: controller.signal })[Symbol.asyncIterator]();
    const pending = iterator.next();
    setTimeout(() => controller.abort(), 20);
    await expect(pending).rejects.toMatchObject({ code: 'ABORTED' });
    expect(fetchSignal?.aborted).toBe(true);
  });

  it('fails with TIMEOUT when the stream idles too long', async () => {
    stubFetch((signal) => new Promise<Response>((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason));
    }));
    const client = new LlamaCppClient('http://127.0.0.1:8080', { streamIdleTimeoutMs: 50 });
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    const error = await iterator.next().then(() => null, (e: unknown) => e) as LlmError;
    expect(error.code).toBe('TIMEOUT');
    expect(error.message).toContain('stream idle timeout');
  });

  it('wraps network failures as TRANSPORT with cause', async () => {
    stubFetch(() => {
      throw new TypeError('fetch failed');
    });
    const client = new LlamaCppClient('http://127.0.0.1:8080');
    const iterator = client.chat({ model: 'qwen3', messages: [], stream: true })[Symbol.asyncIterator]();
    const error = await iterator.next().then(() => null, (e: unknown) => e);
    expect(error).toBeInstanceOf(LlmError);
    expect((error as LlmError).code).toBe('TRANSPORT');
    expect((error as LlmError).cause).toBeInstanceOf(TypeError);
  });
});

describe('checkHealth', () => {
  it('returns true on 2xx /health', async () => {
    stubFetch(new Response(JSON.stringify({ status: 'ok' }), { status: 200 }));
    await expect(checkHealth('http://127.0.0.1:8080')).resolves.toBe(true);
    expect(fetchMock.mock.calls[0]?.[0]).toBe('http://127.0.0.1:8080/health');
  });

  it('returns false on non-2xx and network failures', async () => {
    stubFetch(new Response('down', { status: 503 }));
    await expect(checkHealth('http://127.0.0.1:8080')).resolves.toBe(false);
    fetchMock.mockImplementation(() => {
      throw new TypeError('ECONNREFUSED');
    });
    await expect(checkHealth('http://127.0.0.1:8080')).resolves.toBe(false);
  });
});

describe('error mapping helpers', () => {
  it('maps httpErrorCode across statuses', () => {
    expect(httpErrorCode(401)).toBe('AUTH');
    expect(httpErrorCode(403)).toBe('AUTH');
    expect(httpErrorCode(429)).toBe('RATE_LIMIT');
    expect(httpErrorCode(400)).toBe('INVALID_REQUEST');
    expect(httpErrorCode(400, { message: 'context length exceeded' })).toBe('CONTEXT_WINDOW_EXCEEDED');
    expect(httpErrorCode(500)).toBe('SERVER');
    expect(httpErrorCode(404)).toBe('HTTP_404');
  });

  it('parses Retry-After seconds and dates', () => {
    expect(providerRetryAfterMs('2')).toBe(2000);
    expect(providerRetryAfterMs('not-a-number')).toBeUndefined();
    expect(providerRetryAfterMs('0')).toBeUndefined();
    expect(providerRetryAfterMs(null)).toBeUndefined();
  });
});

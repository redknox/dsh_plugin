/**
 * llama.cpp OpenAI-compatible streaming client.
 *
 * Transport only: HTTP request construction, SSE streaming, cancellation, idle
 * timeout, and protocol parsing against `/v1/chat/completions`. Harness
 * message/tool conversion belongs to the adapter (`adapter.ts`), which depends
 * on this client — never the other way around. The client imports no Harness
 * runtime or Cordis code (only leaf public types/errors), so it can be
 * unit-tested with mocked HTTP/SSE responses alone.
 *
 * @module llm-llamacpp/client
 */
import { EventSourceParserStream } from 'eventsource-parser/stream';
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  LlmError,
  QUOTA_EXCEEDED_CODE,
  attributionHeaders,
  isContextWindowExceededError,
  isQuotaExceededError,
  type ProviderRequestId,
} from '@deepseek-ai/dsh-llm';
import type {
  LlamaCppChatCompletionChunk,
  LlamaCppChatCompletionRequest,
} from './protocol.ts';
import { SSE_DONE } from './protocol.ts';

/** Optional bearer/custom-header auth attached to every request. */
export interface LlamaCppAuth {
  /** Lowercase header name; `authorization` sends `Bearer <key>`, anything else sends the raw key. */
  readonly name: string;
  /** Full header value, e.g. `Bearer sk-...` or a raw API key. */
  readonly value: string;
}

/** Per-request client options. */
export interface LlamaCppRequestOptions {
  /** Caller cancellation for the whole request, including body reads. */
  signal?: AbortSignal;
}

/** Client-level options bound at construction. */
export interface LlamaCppClientOptions {
  /** Maximum idle interval (ms) for one outstanding provider read; default 300s. */
  readonly streamIdleTimeoutMs?: number;
  /**
   * Hard per-request-attempt timeout (ms), regardless of activity. Optional;
   * absent means no total deadline (the idle watchdog still applies).
   */
  readonly requestTimeoutMs?: number;
  /** Optional auth header attached to every request. */
  readonly auth?: LlamaCppAuth;
  /** Additional lowercase header name → value pairs merged into every request. */
  readonly headers?: Readonly<Record<string, string>>;
}

/** Default maximum idle interval while a provider stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000;

/**
 * Per-read idle watchdog: aborts the combined signal with a `TIMEOUT` LlmError
 * when no transport activity arrives for `timeoutMs`. It pulses on raw body
 * bytes (before text/SSE parsing) and on SSE comments, so a long fragmented
 * SSE event that keeps receiving bytes does not trip it.
 */
class IdleWatchdog {
  readonly signal: AbortSignal;
  readonly reason: LlmError;
  private readonly controller = new AbortController();
  private readonly timeoutMs: number;
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(timeoutMs: number, message: string) {
    this.timeoutMs = timeoutMs;
    this.reason = new LlmError(message, 'TIMEOUT');
    this.signal = this.controller.signal;
    this.arm();
  }

  private arm(): void {
    if (this.timer !== undefined) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.controller.abort(this.reason), this.timeoutMs);
  }

  /** Re-arm the watchdog; call on any transport activity (data or comment). */
  pulse(): void {
    this.arm();
  }

  /** Stop the watchdog and cancel any in-flight read. */
  dispose(): void {
    if (this.timer !== undefined) clearTimeout(this.timer);
    this.timer = undefined;
    this.controller.abort();
  }
}

/**
 * One-shot hard deadline for an entire request attempt: aborts the combined
 * signal with a `TIMEOUT` LlmError after `timeoutMs`, regardless of activity
 * (unlike the idle watchdog, which re-arms on transport activity).
 */
class Deadline {
  readonly signal: AbortSignal;
  readonly reason: LlmError;
  private readonly controller = new AbortController();
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(timeoutMs: number, message: string) {
    this.reason = new LlmError(message, 'TIMEOUT');
    this.signal = this.controller.signal;
    this.timer = setTimeout(() => this.controller.abort(this.reason), timeoutMs);
  }

  dispose(): void {
    if (this.timer !== undefined) clearTimeout(this.timer);
    this.timer = undefined;
    this.controller.abort();
  }
}

/** Strip a trailing slash so `baseURL + path` never doubles one. */
function joinURL(baseURL: string, path: string): string {
  return `${baseURL.replace(/\/+$/, '')}${path}`;
}

/**
 * Map an HTTP status and parsed provider error to a stable Harness error code.
 */
export function httpErrorCode(
  status: number,
  error?: { code?: string; type?: string; message?: string },
): string {
  if (status === 401 || status === 403) return 'AUTH';
  const detail = [error?.code, error?.type, error?.message].filter(Boolean).join(' ');
  if (isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE;
  if (status === 429) return 'RATE_LIMIT';
  if (status === 400) {
    if (isContextWindowExceededError(detail)) return CONTEXT_WINDOW_EXCEEDED_CODE;
    return 'INVALID_REQUEST';
  }
  if (status >= 500) return 'SERVER';
  return `HTTP_${status}`;
}

/** Parse a `Retry-After` header (seconds or HTTP date) into milliseconds. */
export function providerRetryAfterMs(value: string | null): number | undefined {
  if (value === null) return undefined;
  if (/^\d+$/.test(value)) {
    const delay = Number(value) * 1000;
    return Number.isFinite(delay) && delay > 0 ? delay : undefined;
  }
  const delay = Date.parse(value) - Date.now();
  return Number.isFinite(delay) && delay > 0 ? delay : undefined;
}

/** Read a provider-issued request id from common response headers. */
function requestIdOf(headers: Headers): ProviderRequestId | undefined {
  const value = headers.get('x-request-id') ?? headers.get('x-llamacpp-request-id');
  return value === null || value.length === 0 ? undefined : value as ProviderRequestId;
}

/**
 * Minimal connectivity probe against the llama.cpp server. Returns a boolean
 * and never throws, so it is safe to call from health surfaces.
 * @param baseURL - the server base URL; `/health` is probed.
 * @param signal - optional caller cancellation.
 * @param timeoutMs - probe timeout in milliseconds (default 5s).
 */
export async function checkHealth(
  baseURL: string,
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 5_000);
  const signal = options.signal !== undefined ? AbortSignal.any([options.signal, controller.signal]) : controller.signal;
  try {
    const response = await fetch(joinURL(baseURL, '/health'), { signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Streaming client for llama.cpp's OpenAI-compatible chat completions.
 * One instance is bound to one endpoint; issue #7 adds endpoint selection.
 */
export class LlamaCppClient {
  readonly streamIdleTimeoutMs: number;
  readonly requestTimeoutMs?: number;
  readonly auth?: LlamaCppAuth;
  readonly headers: Readonly<Record<string, string>>;

  constructor(
    readonly baseURL: string,
    options: LlamaCppClientOptions = {},
  ) {
    this.streamIdleTimeoutMs = options.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS;
    this.requestTimeoutMs = options.requestTimeoutMs;
    this.auth = options.auth;
    this.headers = options.headers ?? {};
  }

  private requestHeaders(): Record<string, string> {
    // Reserved headers (attribution, transport, and the configured auth
    // header) are owned by the client: user-supplied `headers` entries for
    // them are stripped so they can never suppress the mandatory Harness
    // attribution, break the content-type/accept contract, or spoof auth.
    const reserved = new Set([
      'content-type',
      'accept',
      'user-agent',
      'authorization',
      ...(this.auth !== undefined ? [this.auth.name] : []),
    ]);
    const user: Record<string, string> = {};
    for (const [name, value] of Object.entries(this.headers)) {
      if (!reserved.has(name.toLowerCase())) user[name] = value;
    }
    return {
      ...user,
      'content-type': 'application/json',
      accept: 'text/event-stream',
      ...attributionHeaders(),
      ...this.auth !== undefined ? { [this.auth.name]: this.auth.value } : {},
    };
  }

  /**
   * POST one chat completion and yield its parsed SSE data payloads
   * incrementally. `stream` is forced true. The response is consumed as a
   * stream, never buffered; each payload is parsed as it arrives.
   *
   * Failure taxonomy: caller aborts become `ABORTED`, idle timeouts become
   * `TIMEOUT`, non-2xx responses become typed `LlmError`s with `status` and
   * provider diagnostics, malformed JSON payloads become `MALFORMED_RESPONSE`,
   * and a stream that ends without `[DONE]` becomes `STREAM_CLOSED`.
   *
   * @param request - the wire request; `stream` is forced true.
   * @param options - cancellation for the whole request, including body reads.
   */
  async *chat(
    request: LlamaCppChatCompletionRequest,
    options: LlamaCppRequestOptions = {},
  ): AsyncIterable<LlamaCppChatCompletionChunk> {
    const watchdog = new IdleWatchdog(
      this.streamIdleTimeoutMs,
      `llama.cpp stream idle timeout after ${this.streamIdleTimeoutMs}ms from ${this.baseURL}`,
    );
    const deadline = this.requestTimeoutMs !== undefined
      ? new Deadline(
          this.requestTimeoutMs,
          `llama.cpp request timeout after ${this.requestTimeoutMs}ms from ${this.baseURL}`,
        )
      : undefined;
    const signals: AbortSignal[] = [watchdog.signal];
    if (options.signal !== undefined) signals.push(options.signal);
    if (deadline !== undefined) signals.push(deadline.signal);
    const signal = signals.length === 1 ? signals[0]! : AbortSignal.any(signals);
    try {
      const response = await this.post(request, signal);
      if (response.body === null) {
        throw new LlmError('llama.cpp returned no response body', 'EMPTY_RESPONSE');
      }
      for await (const payload of parseSse(response.body, {
        onBytes: () => watchdog.pulse(),
        onComment: () => watchdog.pulse(),
      })) {
        watchdog.pulse();
        if (payload === SSE_DONE) break;
        yield this.parsePayload(payload);
      }
    } catch (error) {
      if (options.signal?.aborted) {
        throw new LlmError('llama.cpp request aborted by caller', 'ABORTED', { cause: error });
      }
      if (watchdog.signal.aborted) throw watchdog.reason;
      if (deadline?.signal.aborted) throw deadline.reason;
      throw error;
    } finally {
      watchdog.dispose();
      deadline?.dispose();
    }
  }

  private async post(request: LlamaCppChatCompletionRequest, signal: AbortSignal): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(joinURL(this.baseURL, '/v1/chat/completions'), {
        method: 'POST',
        headers: this.requestHeaders(),
        body: JSON.stringify({ ...request, stream: true }),
        signal,
      });
    } catch (error) {
      throw new LlmError(
        `llama.cpp request to ${this.baseURL} failed`,
        'TRANSPORT',
        { cause: error },
      );
    }
    if (!response.ok) throw await this.httpError(response);
    return response;
  }

  private async httpError(response: Response): Promise<LlmError> {
    let detail: { code?: string; type?: string; message?: string } | undefined;
    let raw: string | undefined;
    try {
      raw = await response.text();
      const parsed: unknown = JSON.parse(raw);
      if (parsed !== null && typeof parsed === 'object' && 'error' in parsed) {
        detail = (parsed as { error: unknown }).error as { code?: string; type?: string; message?: string };
      } else {
        detail = parsed as { code?: string; type?: string; message?: string };
      }
    } catch {
      // raw text is retained for the bounded fallback below
    }
    const status = response.status;
    let message: string;
    if (detail?.message !== undefined) {
      message = detail.message;
    } else {
      // Non-JSON bodies (nginx 502, HTML error pages, …) are surfaced, bounded.
      const body = raw?.trim().slice(0, 200);
      message = body !== undefined && body.length > 0
        ? `llama.cpp API error (HTTP ${status}): ${body}`
        : `llama.cpp API error (HTTP ${status})`;
    }
    const retryAfter = providerRetryAfterMs(response.headers.get('retry-after'));
    const requestId = requestIdOf(response.headers);
    return new LlmError(message, httpErrorCode(status, detail), {
      status,
      ...retryAfter !== undefined ? { providerRetryAfterMs: retryAfter } : {},
      ...requestId !== undefined ? { requestId } : {},
    });
  }

  private parsePayload(payload: string): LlamaCppChatCompletionChunk {
    try {
      return JSON.parse(payload) as LlamaCppChatCompletionChunk;
    } catch {
      throw new LlmError(
        `llama.cpp sent malformed SSE payload: ${payload.slice(0, 120)}`,
        'MALFORMED_RESPONSE',
      );
    }
  }
}

/** Options for {@link parseSse}. */
export interface ParseSseOptions {
  /** Called for every raw byte chunk received, before text/SSE parsing. */
  onBytes?: (chunk: Uint8Array) => void;
  /** Called for every SSE comment encountered. */
  onComment?: (comment: string) => void;
}

/**
 * Parse an SSE byte stream into data payloads incrementally. Yields each
 * event's `data` payload in arrival order, with `[DONE]` as the final value;
 * throws `STREAM_CLOSED` when the stream ends without it (truncated response).
 * Malformed SSE framing is handled defensively by the parser: a malformed
 * frame is skipped as `onError` (default) rather than aborting the stream.
 * @param stream - raw SSE bytes; reads may split anywhere, including mid-UTF-8.
 * @param options - byte/comment activity callbacks (e.g. watchdog pulses).
 */
export async function* parseSse(
  stream: ReadableStream<Uint8Array>,
  options: ParseSseOptions = {},
): AsyncIterable<string> {
  // Pulse on raw bytes BEFORE text decoding/SSE framing so a long fragmented
  // event that keeps receiving bytes stays alive.
  const bytePulse = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      options.onBytes?.(chunk);
      controller.enqueue(chunk);
    },
  });
  const events = stream
    .pipeThrough(bytePulse)
    .pipeThrough(new TextDecoderStream() as unknown as TransformStream<Uint8Array, string>)
    .pipeThrough(new EventSourceParserStream({ onComment: options.onComment }));
  for await (const { data } of events) {
    yield data;
    if (data === SSE_DONE) return;
  }
  throw new LlmError('llama.cpp SSE stream ended without [DONE]', 'STREAM_CLOSED');
}

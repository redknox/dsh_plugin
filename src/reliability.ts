/**
 * Provider reliability layer (issue #7): ordered multi-endpoint fallback,
 * bounded retry with backoff for retryable HTTP/network failures only, hard
 * timeouts, endpoint health state, and cancellation-safe behavior.
 *
 * This layer is separate from the core `LlmAdapter` translation logic and
 * from the `ReasoningPolicy` inference policy, so a single-server local
 * deployment stays simple while production/self-hosted deployments get
 * fallback and health awareness. It does not touch agent-loop internals.
 *
 * Rules (all tested):
 * - Retry only configured retryable failure codes (normal mode) or every
 *   failure (always mode).
 * - Never retry an explicitly aborted request.
 * - Never retry/fall back after user-visible streamed output has begun.
 * - Ordered fallback: candidates are tried in configuration order, then
 *   cycled; endpoints in backoff are skipped.
 * - Repeatedly failing endpoints accrue exponential backoff with jitter.
 *
 * @module llm-llamacpp/reliability
 */
import { LlmError, type ResolvedRetryPolicy } from '@deepseek-ai/dsh-llm';
import type { LlamaCppAuth, LlamaCppClientOptions } from './client.ts';
import type { LlamacppLogger } from './adapter.ts';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from './protocol.ts';

/** The chat surface the reliability layer drives (structural: real client or fake). */
export interface ReliableChatHandle {
  chat(
    request: LlamaCppChatCompletionRequest,
    options?: { signal?: AbortSignal },
  ): AsyncIterable<LlamaCppChatCompletionChunk>;
}

/** One candidate endpoint. */
export interface ReliabilityEndpoint {
  readonly baseURL: string;
  readonly auth?: LlamaCppAuth;
}

/**
 * Per-attempt reliability report for observability (issue #8): the first
 * attempt is `selected`; later attempts are `retry` (same endpoint) or
 * `fallback` (different endpoint), carrying the failure code that triggered
 * them. Emitted before each attempt begins.
 */
export interface AttemptReport {
  readonly attempt: number;
  readonly baseURL: string;
  readonly outcome: 'selected' | 'retry' | 'fallback';
  readonly failureCode?: string;
}

/** Per-attempt client factory (injected so tests can script fake endpoints). */
export type ReliabilityClientFactory = (
  baseURL: string,
  options: LlamaCppClientOptions,
) => ReliableChatHandle;

/** Options for {@link streamReliably}. */
export interface ReliableStreamOptions {
  readonly endpoints: readonly ReliabilityEndpoint[];
  readonly retryPolicy: ResolvedRetryPolicy;
  readonly streamIdleTimeoutMs: number;
  readonly requestTimeoutMs?: number;
  readonly createClient: ReliabilityClientFactory;
  /** Optional debug/warn logger (structured endpoint/model/retry context). */
  readonly logger?: LlamacppLogger;
  /** Optional persistent health pool; a per-call pool is used when absent. */
  readonly pool?: EndpointPool;
  /** Optional per-attempt observability report (issue #8). */
  readonly onAttempt?: (report: AttemptReport) => void;
  readonly signal?: AbortSignal;
}

/** Stable machine code of a thrown failure, when it has one. */
function failureCode(error: unknown): string | undefined {
  if (error === null || typeof error !== 'object') return undefined;
  const code = (error as { code?: unknown }).code;
  return typeof code === 'string' ? code : undefined;
}

/**
 * Whether a failure is eligible for retry under the given policy.
 * Normal mode: the failure code must be in the configured retryable set.
 * Always mode: every failure is retried until success, cancellation, or
 * disposal.
 */
export function isRetryableFailure(error: unknown, policy: ResolvedRetryPolicy): boolean {
  if (policy.mode === 'always') return true;
  const code = failureCode(error);
  return code !== undefined && policy.retryableCodes.includes(code);
}

/**
 * Per-endpoint health state with exponential backoff. Consecutive failures
 * push the endpoint into backoff (skipped as a candidate); any success resets
 * it. Deterministic backoff delay follows the policy's backoff config, with
 * symmetric jitter applied only when `jitterRatio > 0`. One pool per adapter
 * instance keeps health state across requests, so a repeatedly failing
 * endpoint stays out of rotation.
 */
export class EndpointPool {
  private readonly states = new Map<string, { consecutiveFailures: number; backoffUntil: number }>();

  private backoffFor(failures: number, policy: ResolvedRetryPolicy): number {
    const { initialDelayMs, maxDelayMs, jitterRatio } = policy;
    const exponential = Math.min(initialDelayMs * 2 ** (failures - 1), maxDelayMs);
    if (jitterRatio > 0) {
      const jitter = exponential * jitterRatio * (2 * Math.random() - 1);
      return Math.max(0, Math.round(exponential + jitter));
    }
    return exponential;
  }

  /** Whether an endpoint is currently not in backoff. */
  isHealthy(baseURL: string): boolean {
    const state = this.states.get(baseURL);
    if (state === undefined) return true;
    return Date.now() >= state.backoffUntil;
  }

  /** Expose one endpoint's health facts for diagnostics (issue #12). */
  healthOf(baseURL: string): {
    healthy: boolean;
    inBackoff: boolean;
    backoffUntilMs?: number;
    consecutiveFailures?: number;
  } {
    const state = this.states.get(baseURL);
    if (state === undefined) return { healthy: true, inBackoff: false };
    const now = Date.now();
    return {
      healthy: now >= state.backoffUntil,
      inBackoff: now < state.backoffUntil,
      backoffUntilMs: state.backoffUntil,
      consecutiveFailures: state.consecutiveFailures,
    };
  }

  /**
   * Ordered, health-aware candidate selection. Healthy endpoints in
   * configuration order; when every endpoint is in backoff, fall back to the
   * full ordered list so a request still makes progress (failures keep
   * accruing backoff).
   *
   * With `previous` given (the endpoint just attempted), the next candidate is
   * the next healthy endpoint AFTER it in configuration order, wrapping — so
   * for `[A, B, C]` with `A` failed and in backoff, the next attempt selects
   * `B` (never skipping it because the global attempt counter no longer lines
   * up with the shortened healthy list). Without `previous`, the first healthy
   * endpoint in configuration order is selected.
   */
  nextCandidate(
    endpoints: readonly ReliabilityEndpoint[],
    previous?: ReliabilityEndpoint,
  ): ReliabilityEndpoint {
    const healthy = endpoints.filter((endpoint) => this.isHealthy(endpoint.baseURL));
    const candidates = healthy.length > 0 ? healthy : endpoints;
    if (previous === undefined) return candidates[0]!;
    const start = candidates.findIndex((endpoint) => endpoint.baseURL === previous.baseURL);
    const index = start < 0 ? 0 : (start + 1) % candidates.length;
    return candidates[index]!;
  }

  /**
   * Record a failure, advance backoff under `policy`, and return the delay
   * (ms) to wait before the next attempt (0 means immediately try the next
   * candidate).
   */
  recordFailure(baseURL: string, policy: ResolvedRetryPolicy): number {
    const current = this.states.get(baseURL)?.consecutiveFailures ?? 0;
    const consecutiveFailures = current + 1;
    const delay = this.backoffFor(consecutiveFailures, policy);
    this.states.set(baseURL, { consecutiveFailures, backoffUntil: Date.now() + delay });
    return delay;
  }

  /** Record a success, resetting health for the endpoint. */
  recordSuccess(baseURL: string): void {
    this.states.delete(baseURL);
  }
}

/** Cancellable sleep; aborts immediately (with the abort reason) when signalled. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason instanceof Error ? signal.reason : new LlmError('llama.cpp request aborted by caller', 'ABORTED'));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(signal?.reason instanceof Error ? signal.reason : new LlmError('llama.cpp request aborted by caller', 'ABORTED'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * Stream one model call through the reliability layer: ordered fallback plus
 * bounded retry with backoff. Yields chunks as they arrive; once the first
 * chunk has been yielded, any later failure is fatal (never retry after
 * user-visible streamed output). Cancellation terminates retry/fallback
 * immediately. A single-endpoint configuration behaves exactly as before on
 * the success path.
 */
export async function* streamReliably(
  request: LlamaCppChatCompletionRequest,
  options: ReliableStreamOptions,
): AsyncIterable<LlamaCppChatCompletionChunk> {
  const {
    endpoints,
    retryPolicy,
    streamIdleTimeoutMs,
    requestTimeoutMs,
    createClient,
    logger,
    signal,
  } = options;
  const pool = options.pool ?? new EndpointPool();
  const maxAttempts = retryPolicy.mode === 'always'
    ? Number.POSITIVE_INFINITY
    : retryPolicy.maxRetries + 1;
  let attempts = 0;
  let yielded = false;
  let lastError: unknown;
  let previous: ReliabilityEndpoint | undefined;

  while (attempts < maxAttempts) {
    if (signal?.aborted) {
      if (lastError !== undefined) throw lastError;
      throw new LlmError('llama.cpp request aborted by caller', 'ABORTED');
    }
    const endpoint = pool.nextCandidate(endpoints, previous);
    const outcome: AttemptReport['outcome'] = previous === undefined
      ? 'selected'
      : previous.baseURL === endpoint.baseURL
        ? 'retry'
        : 'fallback';
    options.onAttempt?.({
      attempt: attempts + 1,
      baseURL: endpoint.baseURL,
      outcome,
      ...(outcome !== 'selected' && lastError !== undefined ? { failureCode: failureCode(lastError) } : {}),
    });
    previous = endpoint;
    attempts++;
    const client = createClient(endpoint.baseURL, {
      streamIdleTimeoutMs,
      ...(requestTimeoutMs !== undefined ? { requestTimeoutMs } : {}),
      ...(endpoint.auth !== undefined ? { auth: endpoint.auth } : {}),
    });
    try {
      for await (const chunk of client.chat(request, { signal })) {
        yielded = true;
        pool.recordSuccess(endpoint.baseURL);
        yield chunk;
      }
      pool.recordSuccess(endpoint.baseURL);
      return;
    } catch (error) {
      lastError = error;
      if (signal?.aborted) throw error;
      if (yielded) throw error; // never retry after user-visible streamed output
      if (!isRetryableFailure(error, retryPolicy)) throw error;
      const code = failureCode(error) ?? 'UNKNOWN';
      const label = maxAttempts === Number.POSITIVE_INFINITY ? '∞' : String(maxAttempts);
      const delayMs = pool.recordFailure(endpoint.baseURL, retryPolicy);
      logger?.warn?.(
        `llm-llamacpp: endpoint ${endpoint.baseURL} failed for model ${request.model} ` +
          `(attempt ${attempts}/${label}, code=${code}); ` +
          (delayMs > 0 ? `retrying in ${delayMs}ms` : 'trying next candidate'),
      );
      // Backoff applies when the next attempt retries the SAME endpoint;
      // falling back to a different candidate proceeds immediately.
      const nextEndpoint = pool.nextCandidate(endpoints, endpoint);
      if (nextEndpoint.baseURL === endpoint.baseURL && delayMs > 0) {
        await sleep(delayMs, signal);
      }
    }
  }
  throw lastError ?? new LlmError('llm-llamacpp: all endpoints failed', 'TRANSPORT');
}

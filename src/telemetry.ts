/**
 * Structured observability for provider requests (issue #8).
 *
 * A narrow, provider-owned telemetry model: every model request gets a stable
 * trace `requestId`, and structured events are emitted through a minimal
 * {@link TelemetrySink} instead of ad-hoc log strings. The adapter owns the
 * request context and outcome; the reliability layer reports endpoint
 * selection/retry/fallback; the reasoning layer's decisions ride the request
 * context. Nothing here imports Harness agent-loop internals.
 *
 * Privacy rules (documented and tested): events carry only field names and
 * counts — never prompt content, tool arguments, completions, or secrets.
 * Telemetry can be disabled (Noop sink) without changing provider behavior.
 *
 * @module llm-llamacpp/telemetry
 */
import type { FinishReason, TokenUsage } from '@deepseek-ai/dsh-llm';

/** Stable trace id format prefix for readability in logs. */
export function newRequestId(): string {
  return `llm-${crypto.randomUUID()}`;
}

/** Request context captured at adapter entry, before any wire traffic. */
export interface RequestContext {
  readonly model: string;
  /** Request purpose when the harness labelled the call. */
  readonly purpose?: 'compaction' | 'session-title';
  readonly reasoningEffort?: string;
  readonly reasoningBudgetTokens?: number;
  readonly toolsAvailable: boolean;
}

/**
 * Endpoint selection/retry/fallback report from the reliability layer,
 * classified relative to the previous attempt (first attempt = `selected`).
 */
export interface AttemptEvent {
  readonly attempt: number;
  readonly baseURL: string;
  readonly outcome: 'selected' | 'retry' | 'fallback';
  /** Failure code that triggered the retry/fallback, when any. */
  readonly failureCode?: string;
}

/** Final outcome of one request. */
export interface RequestOutcome {
  /** Endpoint that produced the terminal result. */
  readonly endpoint: string;
  readonly retryCount: number;
  readonly fallbackCount: number;
  /** Time to first user-visible token, ms from adapter entry. */
  readonly ttftMs?: number;
  /** Total request latency, ms from adapter entry to terminal result. */
  readonly totalMs: number;
  /** Completion duration, ms from first token to terminal result. */
  readonly completionMs?: number;
  readonly streamChunkCount: number;
  readonly finishReason?: FinishReason;
  readonly usage?: TokenUsage;
  readonly toolCallCount?: number;
  /** Terminal failure code when the request did not complete normally. */
  readonly failureCode?: string;
}

/** Resolved reasoning decision for the request (emitted after policy resolution). */
export interface ReasoningDecisionEvent {
  readonly enabled: boolean;
  readonly effort?: string;
  readonly budgetTokens?: number;
  /** Adaptive/feedback rationale, when the policy produced one. */
  readonly reason?: string;
}

/** Capability-aware routing decision (issue #9). */
export interface RoutingDecisionEvent {
  /** Eligible endpoint base URLs in preference order. */
  readonly candidates: readonly string[];
  /** Exclusions and final ordering rationale. */
  readonly rationale: readonly string[];
}

/** One structured telemetry event. */
export type TelemetryEvent =
  | {
      readonly type: 'started';
      readonly requestId: string;
      readonly at: number;
      readonly context: RequestContext;
    }
  | {
      readonly type: 'reasoning';
      readonly requestId: string;
      readonly at: number;
      readonly decision: ReasoningDecisionEvent;
    }
  | {
      readonly type: 'routing';
      readonly requestId: string;
      readonly at: number;
      readonly decision: RoutingDecisionEvent;
    }
  | {
      readonly type: 'attempt';
      readonly requestId: string;
      readonly at: number;
      readonly attempt: AttemptEvent;
    }
  | {
      readonly type: 'finished';
      readonly requestId: string;
      readonly at: number;
      readonly outcome: RequestOutcome;
    };

/**
 * The narrow observability interface. A sink receives structured events;
 * `NoopTelemetry` disables emission entirely without changing provider
 * behavior. Implementations must never log prompt content or secrets.
 */
export interface TelemetrySink {
  emit(event: TelemetryEvent): void;
}

/** Disabled sink: drops every event. */
export const NoopTelemetry: TelemetrySink = { emit: () => {} };

/** Minimal structured logger surface the log sink needs. */
export interface TelemetryLogger {
  debug(message: string, ...args: unknown[]): void;
}

/**
 * Structured log sink: emits each event as one JSON line at debug level.
 * Field names and counts only — the caller is responsible for never putting
 * content or secrets into the event (the model enforces it structurally).
 */
export function logTelemetry(logger: TelemetryLogger): TelemetrySink {
  return {
    emit(event) {
      logger.debug('llm-llamacpp telemetry %s', JSON.stringify(event));
    },
  };
}

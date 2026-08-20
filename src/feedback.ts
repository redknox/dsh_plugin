/**
 * Adaptive reasoning feedback loop (issue #11): a bounded, deterministic
 * policy layer that learns from recent provider outcomes.
 *
 * A small {@link ReasoningFeedback} record captures provider-observable
 * outcome signals (outcome class, failure code, retry/fallback use, reasoning
 * token consumption, latency, finish reason) independent of HTTP transport
 * and of Harness agent-loop internals. {@link FeedbackHistory} keeps a bounded
 * window (decay = stale entries drop out; `reset()` clears it). The
 * {@link FeedbackReasoningPolicy} layers over the existing {@link ReasoningPolicy}
 * seam: deterministic adjustment rules, hard min/default/max bounds preserved
 * from #6, explicit per-request effort / expert budgets always win, and a
 * clean/no-history deployment behaves exactly like the base policy.
 *
 * @module llm-llamacpp/feedback
 */
import {
  buildReasoningPolicy,
  DEFAULT_ADAPTIVE_MIN_BUDGET,
  DEFAULT_ADAPTIVE_MAX_BUDGET,
  type ReasoningDecision,
  type ReasoningPolicy,
  type ReasoningPolicyConfig,
  type ReasoningPolicyInput,
} from './reasoning.ts';

/** Outcome class of one completed reasoning-bearing request. */
export type FeedbackOutcome = 'success' | 'failure' | 'timeout' | 'aborted';

/** Bounded outcome signal for one completed request (provider-observable only). */
export interface ReasoningFeedback {
  readonly outcome: FeedbackOutcome;
  /** Terminal failure code when the request did not complete normally. */
  readonly failureCode?: string;
  /** Whether the request used provider-side retry/fallback. */
  readonly retried: boolean;
  /**
   * Whether a tool call was retried. Tri-state: ABSENT means the provider
   * cannot observe it (tool execution/retry lives in Harness `ctx.tools`
   * outside this provider), so it is never recorded as a known `false` —
   * unknown is not negative evidence.
   */
  readonly toolCallRetried?: boolean;
  /** Reasoning tokens actually consumed, when the provider reported them. */
  readonly reasoningTokens?: number;
  /** Total request latency in ms. */
  readonly latencyMs: number;
  /** Finish reason kind when one was produced. */
  readonly finishReason?: string;
}

/** Deterministic summary over the recent feedback window. */
export interface FeedbackSummary {
  readonly count: number;
  /** (failure + timeout + aborted) / count. */
  readonly failureRatio: number;
  readonly timeoutRatio: number;
  readonly abortedRatio: number;
  /** (retried + toolCallRetried) / count. */
  readonly retriedRatio: number;
  /** Average reasoning tokens over entries that reported them. */
  readonly avgReasoningTokens?: number;
  readonly avgLatencyMs: number;
}

/** Default bounded feedback window size (decay: oldest entries drop out). */
export const DEFAULT_FEEDBACK_WINDOW = 20;

/**
 * Bounded outcome history. Appending beyond the window drops the oldest entry
 * (natural decay, so stale failures cannot permanently bias the policy);
 * `reset()` clears it entirely.
 */
export class FeedbackHistory {
  private readonly entries: ReasoningFeedback[] = [];

  constructor(readonly windowSize: number = DEFAULT_FEEDBACK_WINDOW) {}

  get size(): number {
    return this.entries.length;
  }

  record(feedback: ReasoningFeedback): void {
    this.entries.push(feedback);
    if (this.entries.length > this.windowSize) this.entries.shift();
  }

  reset(): void {
    this.entries.length = 0;
  }

  /** Detached snapshot of the current window (for tests and diagnostics). */
  snapshot(): readonly ReasoningFeedback[] {
    return [...this.entries];
  }

  /** Deterministic summary over the current window. */
  summarize(): FeedbackSummary {
    const count = this.entries.length;
    if (count === 0) {
      return { count: 0, failureRatio: 0, timeoutRatio: 0, abortedRatio: 0, retriedRatio: 0, avgLatencyMs: 0 };
    }
    let failures = 0;
    let timeouts = 0;
    let aborted = 0;
    let retried = 0;
    let reasoningTokensTotal = 0;
    let reasoningTokensSeen = 0;
    let latencyTotal = 0;
    for (const entry of this.entries) {
      if (entry.outcome === 'timeout') timeouts += 1;
      else if (entry.outcome === 'aborted') aborted += 1;
      else if (entry.outcome === 'failure') failures += 1;
      // Only KNOWN retry signals count: provider retry/fallback is always
      // known; tool-call retry counts only when explicitly observed (true).
      // An absent (unknown) toolCallRetried is not negative evidence.
      if (entry.retried || entry.toolCallRetried === true) retried += 1;
      if (entry.reasoningTokens !== undefined) {
        reasoningTokensTotal += entry.reasoningTokens;
        reasoningTokensSeen += 1;
      }
      latencyTotal += entry.latencyMs;
    }
    return {
      count,
      failureRatio: (failures + timeouts + aborted) / count,
      timeoutRatio: timeouts / count,
      abortedRatio: aborted / count,
      retriedRatio: retried / count,
      ...(reasoningTokensSeen > 0 ? { avgReasoningTokens: reasoningTokensTotal / reasoningTokensSeen } : {}),
      avgLatencyMs: latencyTotal / count,
    };
  }
}

/**
 * Bounded, deterministic budget adjustment from a summary. Pure: identical
 * inputs always produce identical output and rationale.
 */
export function feedbackBudgetAdjustment(
  base: number,
  summary: FeedbackSummary,
): { budgetTokens: number; reason: string } {
  if (summary.count === 0) return { budgetTokens: base, reason: 'no feedback history' };
  let budget = base;
  const factors: string[] = [];
  // Repeated timeouts/aborts/failures suggest the budget overruns: reduce it.
  if (summary.timeoutRatio >= 0.3) {
    budget = Math.round(budget * 0.8);
    factors.push(`timeouts ${Math.round(summary.timeoutRatio * 100)}%`);
  } else if (summary.abortedRatio >= 0.3) {
    budget = Math.round(budget * 0.85);
    factors.push(`aborted ${Math.round(summary.abortedRatio * 100)}%`);
  } else if (summary.failureRatio >= 0.5) {
    budget = Math.round(budget * 0.9);
    factors.push(`failures ${Math.round(summary.failureRatio * 100)}%`);
  }
  // Reasoning consumption near/at the budget means the cap truncated thinking:
  // increase it.
  if (summary.avgReasoningTokens !== undefined && summary.avgReasoningTokens >= base * 0.9) {
    budget = Math.round(budget * 1.25);
    factors.push('reasoning near budget');
  }
  return { budgetTokens: budget, reason: factors.length > 0 ? factors.join(', ') : 'no adjustment' };
}

/**
 * Feedback-aware reasoning policy: a decorator layered over the base policy
 * (static or adaptive from #6). Explicit per-request effort and expert
 * `budgetTokens` always win (no feedback adjustment); the adjusted budget is
 * clamped to the hard adaptive min/max safety bounds; an empty history
 * behaves exactly like the base policy. The rationale rides the `reason`
 * field, surfaced through #8 observability.
 */
export class FeedbackReasoningPolicy implements ReasoningPolicy {
  readonly base: ReasoningPolicy;
  readonly config: ReasoningPolicyConfig;
  readonly history: FeedbackHistory;

  constructor(base: ReasoningPolicy, config: ReasoningPolicyConfig, history: FeedbackHistory) {
    this.base = base;
    this.config = config;
    this.history = history;
  }

  resolve(input: ReasoningPolicyInput): ReasoningDecision {
    const decision = this.base.resolve(input);
    if (this.config.feedback?.enabled !== true) return decision;
    if (!decision.enabled || decision.budgetTokens === undefined) return decision;
    // Explicit per-request effort and expert budgets win predictably.
    if (input.effort !== undefined) return decision;
    if (this.config.expert?.budgetTokens !== undefined) return decision;
    const summary = this.history.summarize();
    if (summary.count === 0) return decision;
    const adjusted = feedbackBudgetAdjustment(decision.budgetTokens, summary);
    const min = this.config.adaptive?.minBudgetTokens ?? DEFAULT_ADAPTIVE_MIN_BUDGET;
    const max = this.config.adaptive?.maxBudgetTokens ?? DEFAULT_ADAPTIVE_MAX_BUDGET;
    const clamped = Math.min(Math.max(adjusted.budgetTokens, min), max);
    return {
      ...decision,
      budgetTokens: clamped,
      reason: `feedback: ${adjusted.reason}${clamped !== adjusted.budgetTokens ? `, clamped[${min},${max}]` : ''}`,
    };
  }
}

/** Build the full policy: base (static/adaptive) + optional feedback layer. */
export function buildFeedbackPolicy(config: ReasoningPolicyConfig, history: FeedbackHistory): ReasoningPolicy {
  const base = buildReasoningPolicy(config);
  return config.feedback?.enabled === true
    ? new FeedbackReasoningPolicy(base, config, history)
    : base;
}

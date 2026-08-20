/**
 * Qwen reasoning/thinking controls: semantic presets, expert overrides, and
 * policy resolution.
 *
 * The provider exposes stable *semantic* reasoning levels (`off`, `low`,
 * `medium`, `xhigh`) to Harness users; `resolveReasoningPolicy` turns the
 * selected level (plus optional expert overrides) into a resolved policy whose
 * fields are still semantic — `effort` and `budgetTokens` stay separate
 * concepts. Only the adapter/request-builder layer (`serialize.ts`) translates
 * a resolved policy into concrete llama.cpp request fields, because those wire
 * fields depend on the installed llama.cpp/Qwen version (documented there).
 *
 * Issue #6 adds an optional adaptive layer: when `adaptive.enabled`, the
 * resolved preset budget is adjusted deterministically from request context
 * within hard min/max bounds, with the choice explained in debug metadata.
 * Static preset resolution remains the default and unchanged when adaptive is
 * off.
 *
 * Precedence (documented and tested): explicit per-request effort > selected
 * preset (with expert adjustments) > adaptive adjustment > provider
 * defaults / safety bounds. An explicit expert `budgetTokens` always beats the
 * adaptive layer.
 *
 * @module llm-llamacpp/reasoning
 */
import {
  LlmError,
  ReasoningEffortId,
  type ReasoningEffortId as ReasoningEffortIdBrand,
} from '@deepseek-ai/dsh-llm';

/** Semantic reasoning levels offered to Harness users. */
export type ReasoningLevel = 'off' | 'low' | 'medium' | 'xhigh';

/**
 * How a resolved policy is translated to llama.cpp request fields.
 * - `chat-template-kwargs`: Qwen-style `enable_thinking` / `preserve_thinking`
 *   template kwargs (only for families that declare support, or on explicit
 *   configuration);
 * - `reasoning-fields`: native per-request `reasoning_effort` /
 *   `thinking_budget_tokens` fields (llama.cpp-generic);
 * - `none`: no reasoning wire fields at all — the default for families whose
 *   template-kwargs support is unknown, so nothing Qwen-oriented is ever sent
 *   silently (issue #18).
 */
export type ReasoningWireMode = 'chat-template-kwargs' | 'reasoning-fields' | 'none';

/**
 * One semantic reasoning preset. `enabled` is the master thinking switch;
 * `effort` is the model semantic effort (a separate concept from the runtime
 * token budget); `budgetTokens` bounds runtime thinking tokens;
 * `preserveThinking` preserves historical thinking in the provider request
 * (Qwen `chat_template_kwargs.preserve_thinking`) — a request/template
 * behavior; `emitThinking` controls whether thinking is emitted as `reasoning`
 * blocks in the Harness output stream — a purely output-visibility choice.
 */
export interface ReasoningPreset {
  readonly enabled: boolean;
  readonly effort?: string;
  readonly budgetTokens?: number;
  readonly preserveThinking: boolean;
  readonly emitThinking: boolean;
}

/** The built-in preset table. Values are starting points, not ceilings. */
export const REASONING_PRESETS: Readonly<Record<ReasoningLevel, ReasoningPreset>> = {
  off: { enabled: false, preserveThinking: false, emitThinking: true },
  low: { enabled: true, effort: 'low', budgetTokens: 1_024, preserveThinking: false, emitThinking: true },
  medium: { enabled: true, effort: 'medium', budgetTokens: 4_096, preserveThinking: false, emitThinking: true },
  xhigh: { enabled: true, effort: 'xhigh', budgetTokens: 16_384, preserveThinking: false, emitThinking: true },
};

export const REASONING_LEVELS: readonly ReasoningLevel[] = ['off', 'low', 'medium', 'xhigh'];

/**
 * Explicitly named expert/advanced override surface. Overrides never rewrite
 * the preset table; they adjust one resolution. An explicit per-request
 * reasoning effort still wins over the configured preset.
 */
export interface ReasoningExpertOverride {
  readonly enabled?: boolean;
  readonly effort?: string;
  readonly budgetTokens?: number;
  /** Provider-request behavior: preserve historical thinking (Qwen `preserve_thinking`). */
  readonly preserveThinking?: boolean;
  /** Output behavior: emit `reasoning` blocks to the Harness stream (default true). */
  readonly emitThinking?: boolean;
}

/** Plugin-level reasoning configuration (semantic, not wire-level). */
export interface ReasoningPolicyConfig {
  /** Master switch; when false only `off` is advertised and allowed. */
  readonly enabled: boolean;
  /** Default semantic level when the request names none. */
  readonly preset: ReasoningLevel;
  /** Optional expert override surface. */
  readonly expert?: ReasoningExpertOverride;
  /** Which llama.cpp wire translation to use (version-dependent). */
  readonly wire: ReasoningWireMode;
  /** Optional adaptive budget adjustment (issue #6). */
  readonly adaptive?: AdaptiveReasoningConfig;
  /** Optional feedback-informed budget adjustment (issue #11). */
  readonly feedback?: FeedbackReasoningConfig;
}

/**
 * Optional feedback mode (issue #11): layer recent provider outcomes onto the
 * budget decision. Bounded by the adaptive min/max safety bounds; explicit
 * per-request effort and expert budgets always win.
 */
export interface FeedbackReasoningConfig {
  readonly enabled: boolean;
}

/**
 * Configurable adaptive-budget bounds. Adaptive mode adjusts the preset budget
 * from request context within hard `minBudgetTokens`/`maxBudgetTokens` bounds;
 * an explicit expert budget (or per-request effort) always wins over it.
 */
export interface AdaptiveReasoningConfig {
  /** Master switch for the adaptive layer. */
  readonly enabled: boolean;
  /**
   * Optional adaptive base budget that intentionally overrides the selected
   * preset budget before context adjustment (issue #6, blocker 2). Must lie
   * within the configured (or default) min/max bounds.
   */
  readonly defaultBudgetTokens?: number;
  /** Hard lower bound for the adjusted budget; default 512. */
  readonly minBudgetTokens?: number;
  /** Hard upper bound for the adjusted budget; default 65536. */
  readonly maxBudgetTokens?: number;
  /** Configurable task/profile hints applied to every request, e.g. 'short' | 'deep' | 'precise'. */
  readonly hints?: readonly string[];
}

/** Default adaptive safety bounds. */
export const DEFAULT_ADAPTIVE_MIN_BUDGET = 512;
export const DEFAULT_ADAPTIVE_MAX_BUDGET = 65_536;

/**
 * Per-request context an adaptive policy may use. Deliberately independent of
 * HTTP transport and of Harness internals: it is pure request shape.
 */
export interface ReasoningPolicyContext {
  /** Number of conversation messages. */
  readonly messages: number;
  /** Estimated prompt size in tokens (adapter-side approximation). */
  readonly estimatedPromptTokens?: number;
  /** Whether tools are offered on this request. */
  readonly toolsAvailable: boolean;
  /** Whether the current turn follows a tool result. */
  readonly followsToolResult: boolean;
  /** Configurable task/profile hints. */
  readonly hints?: readonly string[];
}

/** A fully resolved, still-semantic reasoning policy for one request. */
export interface ResolvedReasoningPolicy {
  readonly enabled: boolean;
  readonly effort?: string;
  readonly budgetTokens?: number;
  /** Provider-request behavior: preserve historical thinking in the template. */
  readonly preserveThinking: boolean;
  /** Output behavior: emit `reasoning` blocks to the Harness stream. */
  readonly emitThinking: boolean;
  readonly wire: ReasoningWireMode;
}

/**
 * A resolved policy plus an optional human-readable explanation of why that
 * budget was chosen (the debug-metadata channel for adaptive decisions).
 */
export interface ReasoningDecision extends ResolvedReasoningPolicy {
  /** Why this budget was selected; present when the adaptive layer ran. */
  readonly reason?: string;
}

/**
 * Deterministically adjust a base budget from request context within hard
 * bounds. Pure and reproducible: identical inputs always produce identical
 * output and reason.
 */
export function adaptBudget(
  base: number,
  context: ReasoningPolicyContext,
  config: AdaptiveReasoningConfig,
): { budgetTokens: number; reason: string } {
  let budget = base;
  const factors: string[] = [];
  const apply = (factor: number, label: string): void => {
    if (factor === 1) return;
    budget = Math.round(budget * factor);
    factors.push(label);
  };
  // Prompt-size scaling: +10% per 8 messages, capped at +80%.
  apply(1 + (Math.min(context.messages, 64) / 8) * 0.1, `messages=${context.messages}`);
  // Estimated prompt tokens: +20% per 16k tokens, capped at +160%.
  if (context.estimatedPromptTokens !== undefined) {
    apply(
      1 + (Math.min(context.estimatedPromptTokens, 128_000) / 16_000) * 0.2,
      `prompt≈${context.estimatedPromptTokens}t`,
    );
  }
  if (context.toolsAvailable) apply(1.25, 'tools');
  if (context.followsToolResult) apply(0.6, 'tool-result');
  for (const hint of context.hints ?? []) {
    if (hint === 'short') apply(0.5, 'hint:short');
    else if (hint === 'deep' || hint === 'precise') apply(1.5, `hint:${hint}`);
  }
  const min = config.minBudgetTokens ?? DEFAULT_ADAPTIVE_MIN_BUDGET;
  const max = config.maxBudgetTokens ?? DEFAULT_ADAPTIVE_MAX_BUDGET;
  const clamped = Math.min(Math.max(budget, min), max);
  if (clamped !== budget) factors.push(`clamp[${min},${max}]`);
  const reason = factors.length > 0 ? factors.join(', ') : 'no adjustment';
  return { budgetTokens: clamped, reason };
}

/** Parse a harness reasoning-effort id into a semantic level. */
export function parseReasoningLevel(effort: ReasoningEffortIdBrand | string): ReasoningLevel {
  const value = String(effort);
  if ((REASONING_LEVELS as readonly string[]).includes(value)) return value as ReasoningLevel;
  throw new LlmError(
    `llm-llamacpp: unsupported reasoning effort "${value}" (supported: ${REASONING_LEVELS.join(', ')})`,
    'UNSUPPORTED_REASONING_EFFORT',
  );
}

/** Validate a plugin-level reasoning config, failing clearly on bad combos. */
export function validateReasoningConfig(config: ReasoningPolicyConfig, path: string): void {
  if (!config.enabled && config.preset !== 'off') {
    throw new Error(
      `${path}: only reasoning preset "off" can be configured when reasoning is disabled`,
    );
  }
  const expert = config.expert;
  if (expert !== undefined) {
    if (expert.enabled === false && (expert.effort !== undefined || expert.budgetTokens !== undefined)) {
      throw new Error(`${path}.expert: effort/budgetTokens cannot be set when enabled is false`);
    }
    if (expert.budgetTokens !== undefined && (!Number.isSafeInteger(expert.budgetTokens) || expert.budgetTokens <= 0)) {
      throw new Error(`${path}.expert: budgetTokens must be a positive safe integer`);
    }
    if (expert.effort !== undefined && expert.effort.length === 0) {
      throw new Error(`${path}.expert: effort must not be empty`);
    }
  }
  const adaptive = config.adaptive;
  if (adaptive === undefined) return;
  if (
    !adaptive.enabled &&
    (adaptive.defaultBudgetTokens !== undefined ||
      adaptive.minBudgetTokens !== undefined ||
      adaptive.maxBudgetTokens !== undefined ||
      (adaptive.hints?.length ?? 0) > 0)
  ) {
    throw new Error(`${path}.adaptive: defaultBudgetTokens/minBudgetTokens/maxBudgetTokens/hints require enabled: true`);
  }
  const min = adaptive.minBudgetTokens ?? DEFAULT_ADAPTIVE_MIN_BUDGET;
  const max = adaptive.maxBudgetTokens ?? DEFAULT_ADAPTIVE_MAX_BUDGET;
  if (!Number.isSafeInteger(min) || min <= 0) {
    throw new Error(`${path}.adaptive: minBudgetTokens must be a positive safe integer`);
  }
  if (!Number.isSafeInteger(max) || max <= 0) {
    throw new Error(`${path}.adaptive: maxBudgetTokens must be a positive safe integer`);
  }
  if (min > max) {
    throw new Error(`${path}.adaptive: minBudgetTokens must not exceed maxBudgetTokens`);
  }
  if (adaptive.defaultBudgetTokens !== undefined) {
    if (!Number.isSafeInteger(adaptive.defaultBudgetTokens) || adaptive.defaultBudgetTokens <= 0) {
      throw new Error(`${path}.adaptive: defaultBudgetTokens must be a positive safe integer`);
    }
    if (adaptive.defaultBudgetTokens < min || adaptive.defaultBudgetTokens > max) {
      throw new Error(
        `${path}.adaptive: defaultBudgetTokens must lie within [minBudgetTokens, maxBudgetTokens]`,
      );
    }
  }
  for (const hint of adaptive.hints ?? []) {
    if (typeof hint !== 'string' || hint.length === 0) {
      throw new Error(`${path}.adaptive: hints must contain only non-empty strings`);
    }
  }
}

/** One policy resolution input: explicit effort, purpose, and adaptive context. */
export interface ReasoningPolicyInput {
  /** Explicit per-request effort, when the harness selected one. */
  readonly effort?: ReasoningEffortIdBrand;
  /** Optional call purpose; `session-title` disables thinking. */
  readonly purpose?: 'compaction' | 'session-title';
  /** Request context consumed by the adaptive layer; optional for static use. */
  readonly context?: ReasoningPolicyContext;
}

/**
 * The inference-policy seam (issue #6, blocker 1). A policy turns one request
 * shape into a reasoning decision; it is independent of HTTP transport and of
 * Harness internals. The adapter depends on this seam only — it has no
 * knowledge of whether the underlying policy is static or adaptive.
 */
export interface ReasoningPolicy {
  resolve(input: ReasoningPolicyInput): ReasoningDecision;
}

/**
 * Pure static preset resolution: the default policy. Applies per-request
 * effort > preset > expert-override precedence without any adaptive branch.
 */
export function resolveStaticPolicy(
  effort: ReasoningEffortIdBrand | undefined,
  config: ReasoningPolicyConfig,
  purpose?: 'compaction' | 'session-title',
): ResolvedReasoningPolicy {
  if (purpose === 'session-title') {
    return { enabled: false, preserveThinking: false, emitThinking: true, wire: config.wire };
  }
  const level = effort !== undefined ? parseReasoningLevel(effort) : config.preset;
  if (!config.enabled && level !== 'off') {
    throw new LlmError(
      `llm-llamacpp: reasoning is disabled but effort "${level}" was requested`,
      'UNSUPPORTED_REASONING_EFFORT',
    );
  }
  const preset = REASONING_PRESETS[level];
  const expert = config.expert;
  const enabled = expert?.enabled ?? preset.enabled;
  const effortValue = expert?.effort ?? preset.effort;
  const budgetTokens = expert?.budgetTokens ?? preset.budgetTokens;
  const preserveThinking = expert?.preserveThinking ?? preset.preserveThinking;
  const emitThinking = expert?.emitThinking ?? preset.emitThinking;
  if (!enabled && (effortValue !== undefined || budgetTokens !== undefined)) {
    throw new LlmError(
      'llm-llamacpp: expert override disables reasoning but sets effort/budgetTokens',
      'INVALID_REASONING_CONFIG',
    );
  }
  return {
    enabled,
    ...(effortValue !== undefined ? { effort: effortValue } : {}),
    ...(budgetTokens !== undefined ? { budgetTokens } : {}),
    preserveThinking,
    emitThinking,
    wire: config.wire,
  };
}

/** The default policy: static preset resolution. */
export class StaticReasoningPolicy implements ReasoningPolicy {
  readonly config: ReasoningPolicyConfig;

  constructor(config: ReasoningPolicyConfig) {
    this.config = config;
  }

  resolve(input: ReasoningPolicyInput): ReasoningDecision {
    return resolveStaticPolicy(input.effort, this.config, input.purpose);
  }
}

/**
 * Optional adaptive policy: a decorator over a base (static) policy that
 * adjusts the base budget from request context within hard bounds and explains
 * the choice. Precedence: an explicit expert `budgetTokens` and any explicit
 * per-request effort win (no adjustment); otherwise the base budget is either
 * the selected preset budget or the configured adaptive `defaultBudgetTokens`
 * when that intentionally overrides the preset base; then context adjustment;
 * then the min/max clamp.
 */
export class AdaptiveReasoningPolicy implements ReasoningPolicy {
  readonly base: ReasoningPolicy;
  readonly config: ReasoningPolicyConfig;

  constructor(base: ReasoningPolicy, config: ReasoningPolicyConfig) {
    this.base = base;
    this.config = config;
  }

  resolve(input: ReasoningPolicyInput): ReasoningDecision {
    const decision = this.base.resolve(input);
    const adaptive = this.config.adaptive;
    if (adaptive?.enabled !== true) return decision;
    // An explicit per-request effort always wins: no adaptive adjustment.
    if (input.effort !== undefined) return decision;
    if (!decision.enabled || decision.budgetTokens === undefined) return decision;
    // An explicit expert budget always wins over the adaptive layer.
    if (this.config.expert?.budgetTokens !== undefined) return decision;
    if (input.context === undefined) return decision;
    const base = adaptive.defaultBudgetTokens ?? decision.budgetTokens;
    const adjusted = adaptBudget(base, input.context, adaptive);
    return {
      ...decision,
      budgetTokens: adjusted.budgetTokens,
      reason: `adaptive: ${adjusted.reason}`,
    };
  }
}

/** Build the policy implied by the plugin-level reasoning configuration. */
export function buildReasoningPolicy(config: ReasoningPolicyConfig): ReasoningPolicy {
  const base = new StaticReasoningPolicy(config);
  return config.adaptive?.enabled === true ? new AdaptiveReasoningPolicy(base, config) : base;
}

/**
 * Convenience: resolve one request through the policy implied by the config.
 * Precedence (tested): explicit per-request effort > selected preset (with
 * expert adjustments) > adaptive adjustment > provider defaults / safety
 * bounds. Static resolution is unchanged when the adaptive layer is off.
 * @param effort - explicit per-request effort, when the harness selected one.
 * @param config - plugin-level reasoning configuration.
 * @param purpose - optional call purpose; `session-title` disables thinking.
 * @param context - request context consumed by the adaptive layer (issue #6).
 * @returns the resolved semantic policy plus an adaptive explanation, if any.
 */
export function resolveReasoningPolicy(
  effort: ReasoningEffortIdBrand | undefined,
  config: ReasoningPolicyConfig,
  purpose?: 'compaction' | 'session-title',
  context?: ReasoningPolicyContext,
): ReasoningDecision {
  return buildReasoningPolicy(config).resolve({ effort, purpose, context });
}

/** The harness-visible reasoning efforts for one provider/model route. */
export function reasoningEfforts(config: ReasoningPolicyConfig): {
  efforts: Array<{ id: ReasoningEffortIdBrand; name: string }>;
  defaultEffort: ReasoningEffortIdBrand;
} {
  if (!config.enabled) {
    return {
      efforts: [{ id: ReasoningEffortId('off'), name: 'Off' }],
      defaultEffort: ReasoningEffortId('off'),
    };
  }
  return {
    efforts: REASONING_LEVELS.map((level) => ({
      id: ReasoningEffortId(level),
      name: level === 'xhigh' ? 'XHigh' : level[0]!.toUpperCase() + level.slice(1),
    })),
    defaultEffort: ReasoningEffortId(config.preset),
  };
}

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
 * Issue #6 extends this module with an adaptive `ReasoningPolicy` that can
 * adjust budget/effort from request context; static preset resolution remains
 * the default.
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

/** How a resolved policy is translated to llama.cpp request fields. */
export type ReasoningWireMode = 'chat-template-kwargs' | 'reasoning-fields';

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
  if (expert === undefined) return;
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

/**
 * Resolve the reasoning policy for one request. Precedence: an explicit
 * per-request effort wins over the configured preset; the expert override then
 * adjusts individual fields without touching the preset table; a
 * `session-title` purpose always disables thinking (bounded output for titles).
 * @param effort - explicit per-request effort, when the harness selected one.
 * @param config - plugin-level reasoning configuration.
 * @param purpose - optional call purpose; `session-title` disables thinking.
 * @returns the resolved semantic policy.
 */
export function resolveReasoningPolicy(
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

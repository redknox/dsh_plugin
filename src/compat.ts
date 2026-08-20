/**
 * Model-family compatibility profile seam (issue #18).
 *
 * The core adapter owns Harness ↔ llama.cpp semantics; model-family-specific
 * behavior — above all Qwen's chat-template reasoning kwargs — is isolated
 * behind explicit compatibility profiles so the adapter can credibly serve a
 * *generic* llama.cpp provider.
 *
 * Resolution precedence (documented and enforced):
 *
 *   explicit configuration → server/provider capability metadata → model
 *   family compatibility profile → model-name heuristic (last resort)
 *
 * The current implementation deliberately introduces **no model-name
 * heuristic**: an unconfigured or unrecognized model family resolves to the
 * `unknown` profile, which never silently inherits Qwen-only behavior
 * ("unknown != false"). Qwen remains the strongest validated profile and is
 * selected explicitly (`modelFamily: 'qwen'`) or implicitly through the
 * shared default — never by guessing from the model string.
 *
 * @module llm-llamacpp/compat
 */
import type { ReasoningWireMode } from './reasoning.ts';

/** Explicit model-family selector; `'auto'` resolves without a heuristic. */
export type ModelFamilyId = 'auto' | 'qwen';

/**
 * One model family's compatibility facts. Families are the replaceable layer
 * beneath the llama.cpp wire layer: a new family (Llama, Gemma, DeepSeek, …)
 * declares its own profile instead of editing the adapter.
 */
export interface ModelFamilyProfile {
  /** Stable id; the `unknown` family is the non-inheriting default. */
  readonly id: 'qwen' | 'unknown';
  /** Human-readable label for diagnostics/docs. */
  readonly label: string;
  /** Reasoning wire behavior of this family. */
  readonly reasoning: {
    /**
     * The family's default wire translation when the user did not configure
     * `reasoning.wire` explicitly. Explicit configuration always wins.
     */
    readonly defaultWire: ReasoningWireMode;
    /**
     * Whether the family's chat template honors the `enable_thinking` /
     * `preserve_thinking` template kwargs. `undefined` means *unknown* — the
     * adapter must not assume either way (it keeps the configured behavior).
     */
    readonly supportsThinkingKwargs?: boolean;
  };
}

/** The Qwen family: strongest validated compatibility profile (issue #18). */
export const QWEN_PROFILE: ModelFamilyProfile = {
  id: 'qwen',
  label: 'Qwen (validated model family)',
  reasoning: {
    defaultWire: 'chat-template-kwargs',
    supportsThinkingKwargs: true,
  },
};

/**
 * The unknown/default family. It shares llama.cpp's *generic* template-kwargs
 * mechanism (a llama.cpp build forwards `chat_template_kwargs` to any
 * template) but inherits nothing Qwen-specific: capabilities the family does
 * not declare stay unknown, and the wire default is the generic one, always
 * overridable by explicit `reasoning.wire`.
 */
export const UNKNOWN_PROFILE: ModelFamilyProfile = {
  id: 'unknown',
  label: 'Unknown model family',
  reasoning: {
    defaultWire: 'chat-template-kwargs',
  },
};

/** Resolve a family id to its profile; anything unrecognized is unknown. */
export function familyProfileFor(id: ModelFamilyId | undefined): ModelFamilyProfile {
  if (id === 'qwen') return QWEN_PROFILE;
  return UNKNOWN_PROFILE;
}

/**
 * The effective default wire mode for one family: the profile's default,
 * which explicit `reasoning.wire` configuration overrides upstream.
 */
export function defaultReasoningWire(profile: ModelFamilyProfile): ReasoningWireMode {
  return profile.reasoning.defaultWire;
}

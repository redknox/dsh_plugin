/**
 * Reasoning policy tests for issue #4: preset resolution, expert overrides,
 * config validation, and the version-dependent wire translation produced by
 * the request builder. Policy decisions must be deterministic.
 */
import { describe, expect, it } from 'vitest';
import { LlmError, ReasoningEffortId } from '@deepseek-ai/dsh-llm';
import { resolveAdapterOptions } from '../src/config.ts';
import {
  REASONING_PRESETS,
  parseReasoningLevel,
  reasoningEfforts,
  resolveReasoningPolicy,
  validateReasoningConfig,
  type ReasoningPolicyConfig,
} from '../src/reasoning.ts';
import { serializeRequest } from '../src/serialize.ts';
import type { GenerateOptions } from '@deepseek-ai/dsh-llm';
import type { LlamaCppChatCompletionRequest } from '../src/protocol.ts';

/** Assert a function throws an LlmError with the given code. */
function throwsCode(fn: () => unknown, code: string): void {
  try {
    fn();
  } catch (error) {
    expect((error as LlmError).code).toBe(code);
    return;
  }
  throw new Error(`expected an LlmError with code ${code}`);
}

const baseConfig: ReasoningPolicyConfig = {
  enabled: true,
  preset: 'medium',
  wire: 'chat-template-kwargs',
};

const baseOptions: GenerateOptions = {
  provider: 'llamacpp-local',
  model: 'qwen3',
  messages: [],
};

function wire(config: ReasoningPolicyConfig, effort?: string, purpose?: 'compaction' | 'session-title'): LlamaCppChatCompletionRequest {
  const policy = resolveReasoningPolicy(effort !== undefined ? ReasoningEffortId(effort) : undefined, config, purpose);
  return serializeRequest(baseOptions, policy);
}

describe('preset resolution', () => {
  it('keeps effort and budgetTokens as separate semantic concepts', () => {
    expect(REASONING_PRESETS.medium.effort).toBe('medium');
    expect(REASONING_PRESETS.medium.budgetTokens).toBe(4096);
    expect(REASONING_PRESETS.off.enabled).toBe(false);
  });

  it('resolves each semantic level deterministically', () => {
    for (const level of ['off', 'low', 'medium', 'xhigh'] as const) {
      const policy = resolveReasoningPolicy(undefined, { ...baseConfig, preset: level });
      expect(policy.enabled).toBe(REASONING_PRESETS[level].enabled);
      expect(policy.effort).toBe(REASONING_PRESETS[level].effort);
      expect(policy.budgetTokens).toBe(REASONING_PRESETS[level].budgetTokens);
    }
  });

  it('lets an explicit per-request effort win over the configured preset', () => {
    const policy = resolveReasoningPolicy(ReasoningEffortId('off'), { ...baseConfig, preset: 'xhigh' });
    expect(policy.enabled).toBe(false);
    const low = resolveReasoningPolicy(ReasoningEffortId('low'), { ...baseConfig, preset: 'xhigh' });
    expect(low.effort).toBe('low');
    expect(low.budgetTokens).toBe(REASONING_PRESETS.low.budgetTokens);
  });

  it('parses only known levels and rejects others', () => {
    expect(parseReasoningLevel('medium')).toBe('medium');
    throwsCode(() => parseReasoningLevel('high'), 'UNSUPPORTED_REASONING_EFFORT');
  });

  it('rejects a non-off effort when reasoning is disabled', () => {
    const disabled: ReasoningPolicyConfig = { enabled: false, preset: 'off', wire: 'chat-template-kwargs' };
    expect(resolveReasoningPolicy(ReasoningEffortId('off'), disabled).enabled).toBe(false);
    throwsCode(() => resolveReasoningPolicy(ReasoningEffortId('medium'), disabled), 'UNSUPPORTED_REASONING_EFFORT');
  });

  it('disables thinking for session-title purposes', () => {
    const policy = resolveReasoningPolicy(undefined, baseConfig, 'session-title');
    expect(policy.enabled).toBe(false);
  });
});

describe('expert overrides', () => {
  it('adjusts budget without changing the preset table', () => {
    const policy = resolveReasoningPolicy(undefined, {
      ...baseConfig,
      preset: 'medium',
      expert: { budgetTokens: 8192 },
    });
    expect(policy.budgetTokens).toBe(8192);
    expect(policy.effort).toBe('medium');
    expect(REASONING_PRESETS.medium.budgetTokens).toBe(4096);
  });

  it('applies expert fields over any preset', () => {
    const policy = resolveReasoningPolicy(undefined, {
      ...baseConfig,
      preset: 'low',
      expert: { enabled: true, effort: 'custom', budgetTokens: 512, preserveThinking: false },
    });
    expect(policy).toMatchObject({ enabled: true, effort: 'custom', budgetTokens: 512, preserveThinking: false });
  });

  it('rejects an expert override that disables reasoning but keeps effort/budget', () => {
    throwsCode(() => resolveReasoningPolicy(undefined, {
      ...baseConfig,
      expert: { enabled: false, budgetTokens: 100 },
    }), 'INVALID_REASONING_CONFIG');
  });
});

describe('config validation', () => {
  it('rejects reasoning disabled with a non-off preset', () => {
    expect(() => validateReasoningConfig({ enabled: false, preset: 'low', wire: 'chat-template-kwargs' }, 'cfg')).toThrow(/only reasoning preset "off"/);
    expect(() => resolveAdapterOptions({ reasoning: { enabled: false, preset: 'low' } })).toThrow(/reasoning/);
  });

  it('rejects non-positive expert budgetTokens', () => {
    expect(() => validateReasoningConfig({ ...baseConfig, expert: { budgetTokens: 0 } }, 'cfg')).toThrow(/budgetTokens/);
    expect(() => resolveAdapterOptions({ reasoning: { expert: { budgetTokens: 0 } } })).toThrow(/budgetTokens/);
  });

  it('accepts a valid expert override at load', () => {
    const options = resolveAdapterOptions({ reasoning: { expert: { enabled: true, effort: 'high', budgetTokens: 2048, preserveThinking: false } } });
    expect(options.reasoning.expert).toEqual({ enabled: true, effort: 'high', budgetTokens: 2048, preserveThinking: false });
  });
});

describe('wire translation', () => {
  it('maps an enabled preset to enable_thinking template kwarg + thinking_budget_tokens', () => {
    const request = wire({ ...baseConfig, preset: 'medium' });
    expect(request.chat_template_kwargs).toEqual({ enable_thinking: true });
    expect(request.thinking_budget_tokens).toBe(4096);
  });

  it('maps off to enable_thinking: false with no stale budget/preserve fields', () => {
    const request = wire({ ...baseConfig, preset: 'off' });
    expect(request.chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(request.thinking_budget_tokens).toBeUndefined();
    expect(request.reasoning_effort).toBeUndefined();
  });

  it('sends preserve_thinking template kwarg only when preserveThinking is set', () => {
    const base = wire({ ...baseConfig, preset: 'medium' });
    expect(base.chat_template_kwargs).toEqual({ enable_thinking: true });

    const preserved = wire({ ...baseConfig, preset: 'medium', expert: { preserveThinking: true } });
    expect(preserved.chat_template_kwargs).toEqual({ enable_thinking: true, preserve_thinking: true });
    expect(preserved.thinking_budget_tokens).toBe(4096);

    const notPreserved = wire({ ...baseConfig, preset: 'medium', expert: { preserveThinking: false } });
    expect(notPreserved.chat_template_kwargs).toEqual({ enable_thinking: true });
  });

  it('sends an expert budgetTokens through the per-request thinking_budget_tokens field', () => {
    const request = wire({ ...baseConfig, preset: 'medium', expert: { budgetTokens: 8192 } });
    expect(request.thinking_budget_tokens).toBe(8192);
    expect(request.chat_template_kwargs).toEqual({ enable_thinking: true });
  });

  it('keeps the preset budget when the expert override does not name one', () => {
    const request = wire({ ...baseConfig, expert: { enabled: true } });
    expect(request.chat_template_kwargs).toEqual({ enable_thinking: true });
    expect(request.thinking_budget_tokens).toBe(4096);
  });

  it('uses native reasoning fields in reasoning-fields mode', () => {
    const request = wire({ ...baseConfig, wire: 'reasoning-fields', preset: 'xhigh' });
    expect(request.reasoning_effort).toBe('xhigh');
    expect(request.thinking_budget_tokens).toBe(16384);
    expect(request.chat_template_kwargs).toBeUndefined();
  });

  it('sends preserve_thinking alongside native fields in reasoning-fields mode', () => {
    const request = wire({
      ...baseConfig,
      wire: 'reasoning-fields',
      preset: 'xhigh',
      expert: { preserveThinking: true },
    });
    expect(request.chat_template_kwargs).toEqual({ preserve_thinking: true });
    expect(request.reasoning_effort).toBe('xhigh');
    expect(request.thinking_budget_tokens).toBe(16384);
  });

  it('never leaves preserve_thinking on the request when reasoning is off', () => {
    const request = wire({
      ...baseConfig,
      wire: 'reasoning-fields',
      preset: 'off',
      expert: { preserveThinking: true },
    });
    expect(request.reasoning_effort).toBe('none');
    expect(request.chat_template_kwargs).toBeUndefined();
    expect(request.thinking_budget_tokens).toBeUndefined();
  });

  it('uses reasoning_effort: none for disabled reasoning in reasoning-fields mode', () => {
    const request = wire({ ...baseConfig, wire: 'reasoning-fields', preset: 'off' });
    expect(request.reasoning_effort).toBe('none');
    expect(request.thinking_budget_tokens).toBeUndefined();
  });
});

describe('harness-facing reasoning metadata', () => {
  it('exposes ordered efforts with a configured default', () => {
    const { efforts, defaultEffort } = reasoningEfforts({ ...baseConfig, preset: 'low' });
    expect(efforts.map((e) => e.id)).toEqual(['off', 'low', 'medium', 'xhigh']);
    expect(efforts.map((e) => e.name)).toEqual(['Off', 'Low', 'Medium', 'XHigh']);
    expect(defaultEffort).toBe('low');
  });

  it('exposes only off when disabled', () => {
    const { efforts, defaultEffort } = reasoningEfforts({ enabled: false, preset: 'off', wire: 'chat-template-kwargs' });
    expect(efforts.map((e) => e.id)).toEqual(['off']);
    expect(defaultEffort).toBe('off');
  });
});

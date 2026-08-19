/**
 * Adaptive reasoning budget tests for issue #6: the optional policy layer must
 * adjust the preset budget deterministically from request context within hard
 * bounds, keep static resolution unchanged when disabled, honor explicit
 * expert/per-request precedence, and explain its choices in debug metadata.
 */
import { describe, expect, it, vi } from 'vitest';
import { ReasoningEffortId, type ToolResultBlock } from '@deepseek-ai/dsh-llm';
import {
  adaptBudget,
  resolveReasoningPolicy,
  type AdaptiveReasoningConfig,
  type ReasoningPolicyConfig,
  type ReasoningPolicyContext,
} from '../src/reasoning.ts';
import { resolveAdapterOptions } from '../src/config.ts';
import { baseOptions, collect, harness, lastRequest } from './helpers.ts';

const baseConfig: ReasoningPolicyConfig = {
  enabled: true,
  preset: 'medium',
  wire: 'chat-template-kwargs',
  adaptive: { enabled: true },
};

const adaptive: AdaptiveReasoningConfig = { enabled: true };

/** A minimal context with everything neutral (no adjustment factors). */
function context(partial: Partial<ReasoningPolicyContext> = {}): ReasoningPolicyContext {
  return {
    messages: 1,
    estimatedPromptTokens: 100,
    toolsAvailable: false,
    followsToolResult: false,
    ...partial,
  };
}

describe('adaptBudget (deterministic rule)', () => {
  it('is deterministic: identical inputs produce identical output', () => {
    const input = context({ messages: 20, toolsAvailable: true });
    const a = adaptBudget(4096, input, adaptive);
    const b = adaptBudget(4096, input, adaptive);
    expect(a).toEqual(b);
    expect(a.budgetTokens).toBe(b.budgetTokens);
  });

  it('grows the budget with the message count', () => {
    expect(adaptBudget(4096, context({ messages: 0, estimatedPromptTokens: 0 }), adaptive).budgetTokens).toBe(4096);
    expect(adaptBudget(4096, context({ messages: 40 }), adaptive).budgetTokens).toBeGreaterThan(4096);
    expect(adaptBudget(4096, context({ messages: 64 }), adaptive).budgetTokens).toBeGreaterThan(
      adaptBudget(4096, context({ messages: 8 }), adaptive).budgetTokens,
    );
  });

  it('clamps to the configured max budget', () => {
    const tight: AdaptiveReasoningConfig = { enabled: true, maxBudgetTokens: 5000 };
    const result = adaptBudget(4096, context({ messages: 64, toolsAvailable: true }), tight);
    expect(result.budgetTokens).toBe(5000);
    expect(result.reason).toContain('clamp[512,5000]');
  });

  it('never drops below the configured min budget', () => {
    const floored: AdaptiveReasoningConfig = { enabled: true, minBudgetTokens: 2000 };
    // base 1024 * 0.6 (tool-result) * 0.5 (hint:short) would go below 2000.
    const result = adaptBudget(1024, context({ followsToolResult: true, hints: ['short'] }), floored);
    expect(result.budgetTokens).toBe(2000);
  });

  it('reduces the budget after a tool result', () => {
    const plain = adaptBudget(4096, context({}), adaptive);
    const afterTool = adaptBudget(4096, context({ followsToolResult: true }), adaptive);
    expect(afterTool.budgetTokens).toBeLessThan(plain.budgetTokens);
  });

  it('increases the budget when tools are available', () => {
    const plain = adaptBudget(4096, context({}), adaptive);
    const withTools = adaptBudget(4096, context({ toolsAvailable: true }), adaptive);
    expect(withTools.budgetTokens).toBeGreaterThan(plain.budgetTokens);
  });

  it('honors short/deep task hints', () => {
    const short = adaptBudget(4096, context({ hints: ['short'] }), adaptive);
    const deep = adaptBudget(4096, context({ hints: ['deep'] }), adaptive);
    const plain = adaptBudget(4096, context({}), adaptive);
    expect(short.budgetTokens).toBeLessThan(plain.budgetTokens);
    expect(deep.budgetTokens).toBeGreaterThan(plain.budgetTokens);
  });

  it('explains the adjustment in the reason string', () => {
    const result = adaptBudget(4096, context({ messages: 16, toolsAvailable: true }), adaptive);
    expect(result.reason).toContain('messages=16');
    expect(result.reason).toContain('tools');
  });
});

describe('resolveReasoningPolicy with adaptive', () => {
  it('keeps static resolution unchanged when adaptive is disabled', () => {
    const staticConfig: ReasoningPolicyConfig = { ...baseConfig, adaptive: { enabled: false } };
    const withContext = resolveReasoningPolicy(undefined, staticConfig, undefined, context({ messages: 40 }));
    const without = resolveReasoningPolicy(undefined, staticConfig);
    expect(withContext.budgetTokens).toBe(4096);
    expect(without.budgetTokens).toBe(4096);
    expect(withContext.reason).toBeUndefined();
  });

  it('adjusts the preset budget from context when adaptive is enabled', () => {
    const decision = resolveReasoningPolicy(undefined, baseConfig, undefined, context({ messages: 40 }));
    expect(decision.budgetTokens).toBeGreaterThan(4096);
    expect(decision.reason).toMatch(/^adaptive: /);
  });

  it('still lets an explicit per-request effort win', () => {
    const decision = resolveReasoningPolicy(
      ReasoningEffortId('off'),
      { ...baseConfig, preset: 'xhigh' },
      undefined,
      context({ messages: 64 }),
    );
    expect(decision.enabled).toBe(false);
    expect(decision.budgetTokens).toBeUndefined();
    expect(decision.reason).toBeUndefined();
  });

  it('never lets the adaptive layer override an explicit expert budget', () => {
    const config: ReasoningPolicyConfig = {
      ...baseConfig,
      preset: 'low',
      expert: { budgetTokens: 777 },
    };
    const decision = resolveReasoningPolicy(undefined, config, undefined, context({ messages: 64, toolsAvailable: true }));
    expect(decision.budgetTokens).toBe(777);
    expect(decision.reason).toBeUndefined();
  });

  it('ignores the adaptive layer for session-title purposes', () => {
    const decision = resolveReasoningPolicy(undefined, baseConfig, 'session-title', context({ messages: 64 }));
    expect(decision.enabled).toBe(false);
    expect(decision.budgetTokens).toBeUndefined();
  });
});

describe('config validation for adaptive', () => {
  it('rejects min greater than max', () => {
    expect(() => resolveAdapterOptions({ reasoning: { adaptive: { enabled: true, minBudgetTokens: 9000, maxBudgetTokens: 5000 } } }))
      .toThrow(/minBudgetTokens must not exceed maxBudgetTokens/);
  });

  it('rejects non-positive bounds', () => {
    expect(() => resolveAdapterOptions({ reasoning: { adaptive: { enabled: true, minBudgetTokens: 0 } } }))
      .toThrow(/minBudgetTokens/);
    expect(() => resolveAdapterOptions({ reasoning: { adaptive: { enabled: true, maxBudgetTokens: -1 } } }))
      .toThrow(/maxBudgetTokens/);
  });

  it('rejects bounds/hints configured while adaptive is disabled', () => {
    expect(() => resolveAdapterOptions({ reasoning: { adaptive: { enabled: false, maxBudgetTokens: 9000 } } }))
      .toThrow(/require enabled: true/);
    expect(() => resolveAdapterOptions({ reasoning: { adaptive: { hints: ['deep'] } } }))
      .toThrow(/require enabled: true/);
  });

  it('accepts a valid adaptive config at load', () => {
    const options = resolveAdapterOptions({
      reasoning: { adaptive: { enabled: true, minBudgetTokens: 1024, maxBudgetTokens: 8192, hints: ['deep'] } },
    });
    expect(options.reasoning.adaptive).toEqual({ enabled: true, minBudgetTokens: 1024, maxBudgetTokens: 8192, hints: ['deep'] });
  });
});

describe('adapter integration', () => {
  it('extracts policy context and emits a debug log explaining the decision', async () => {
    const debug = vi.fn();
    const { adapter, fakeChat } = harness(
      { reasoning: { adaptive: { enabled: true }, expert: { emitThinking: false } } },
      [],
      { debug },
    );
    await collect(adapter.stream({
      ...baseOptions,
      messages: [
        {
          id: 'm1' as never,
          role: 'user',
          content: [{ type: 'text', text: 'hi' }],
          source: { kind: 'user' },
        },
        {
          id: 'm2' as never,
          role: 'user',
          content: [{ type: 'text', text: 'tell me the time please' }],
          source: { kind: 'user' },
        },
      ],
      tools: [{ name: 'get_time', description: 'Get time', parameters: { type: 'object' } }],
    }));

    expect(debug).toHaveBeenCalledTimes(1);
    const log = debug.mock.calls[0]?.[0] as string;
    expect(log).toContain('llm-llamacpp reasoning decision: adaptive:');
    expect(log).toContain('budget=');

    // Deterministic: 4096 * 1.025 (2 messages) * 1.25 (tools) = 5248.
    const wire = lastRequest(fakeChat);
    expect(wire.thinking_budget_tokens).toBe(5248);
  });

  it('does not log an adaptive reason when the layer is off', async () => {
    const debug = vi.fn();
    const { adapter } = harness({}, [], { debug });
    await collect(adapter.stream(baseOptions));
    const log = debug.mock.calls[0]?.[0] as string;
    expect(log).toContain('static preset');
  });
});

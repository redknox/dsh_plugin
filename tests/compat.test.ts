/**
 * Issue #18: the core adapter must not depend on undocumented Qwen
 * assumptions. The compatibility seam is **behaviorally effective**: the
 * unknown family sends no Qwen-oriented thinking kwargs by default
 * (`reasoning.wire: 'none'`), and only explicit configuration, capability
 * metadata, or an explicit family profile (Qwen) opts into them. Tests prove
 * the generic path works with a non-Qwen model name without any Qwen wire
 * parameters, and that the validated Qwen behavior is preserved when
 * explicitly selected.
 */
import { describe, expect, it, vi } from 'vitest';
import type { GenerateOptions } from '@deepseek-ai/dsh-llm';
import { LlamacppAdapter } from '../src/adapter.ts';
import { familyProfileFor, defaultReasoningWire, QWEN_PROFILE, UNKNOWN_PROFILE } from '../src/compat.ts';
import { resolveAdapterOptions } from '../src/config.ts';
import { serializeRequest } from '../src/serialize.ts';
import { harness, collect, msg, baseOptions } from './helpers.ts';

describe('model-family compatibility profile (issue #18)', () => {
  it('resolves an unconfigured or "auto" family to the unknown profile, never Qwen', () => {
    expect(familyProfileFor(undefined).id).toBe('unknown');
    expect(familyProfileFor('auto').id).toBe('unknown');
    expect(familyProfileFor('qwen').id).toBe('qwen');
  });

  it('the unknown profile inherits nothing Qwen-specific and defaults to no wire kwargs', () => {
    expect(UNKNOWN_PROFILE.reasoning.supportsThinkingKwargs).toBeUndefined();
    expect(QWEN_PROFILE.reasoning.supportsThinkingKwargs).toBe(true);
    // Behaviorally effective: only a profile declaring template-kwargs
    // support defaults to sending them; unknown/unsupported -> 'none'.
    expect(defaultReasoningWire(UNKNOWN_PROFILE)).toBe('none');
    expect(defaultReasoningWire(QWEN_PROFILE)).toBe('chat-template-kwargs');
  });

  it('resolveAdapterOptions never guesses the family from the model name', () => {
    // A model name that looks like Qwen must NOT flip the profile: no heuristic.
    const auto = resolveAdapterOptions({ model: 'qwen3-27b' });
    expect(auto.family.id).toBe('unknown');
    expect(auto.reasoning.wire).toBe('none');
    // Explicit selection is the only way to the Qwen profile.
    const qwen = resolveAdapterOptions({ model: 'custom-model', modelFamily: 'qwen' });
    expect(qwen.family.id).toBe('qwen');
    expect(qwen.family.reasoning.supportsThinkingKwargs).toBe(true);
    expect(qwen.reasoning.wire).toBe('chat-template-kwargs');
  });

  it('explicit reasoning.wire beats the profile default', () => {
    const qwen = resolveAdapterOptions({
      modelFamily: 'qwen',
      reasoning: { wire: 'reasoning-fields' },
    });
    expect(qwen.family.id).toBe('qwen');
    expect(qwen.reasoning.wire).toBe('reasoning-fields');
    const unknown = resolveAdapterOptions({ reasoning: { wire: 'chat-template-kwargs' } });
    expect(unknown.family.id).toBe('unknown');
    expect(unknown.reasoning.wire).toBe('chat-template-kwargs');
  });
});

describe('unknown family sends no Qwen wire kwargs by default (issue #18)', () => {
  const NON_QWEN_MODEL = 'granite-3.3';
  const policy = (wire: 'none' | 'chat-template-kwargs' | 'reasoning-fields', over: Partial<{ preserveThinking: boolean; enabled: boolean; budgetTokens: number }> = {}) => ({
    enabled: over.enabled ?? true,
    budgetTokens: over.budgetTokens,
    preserveThinking: over.preserveThinking ?? false,
    emitThinking: true,
    wire,
  });

  it('default config (unknown family) emits no reasoning wire fields at all', () => {
    const request = serializeRequest(
      { ...baseOptions, model: NON_QWEN_MODEL, messages: [msg('user', 'hello')] },
      policy('none', { budgetTokens: 4096 }),
    );
    expect(request.chat_template_kwargs).toBeUndefined();
    expect(request.thinking_budget_tokens).toBeUndefined();
    expect(request.reasoning_effort).toBeUndefined();
  });

  it('enable_thinking is absent for unknown families unless explicitly configured', () => {
    // Unknown + explicit chat-template-kwargs opt-in -> kwargs are sent.
    const opted = serializeRequest(
      { ...baseOptions, model: NON_QWEN_MODEL, messages: [] },
      policy('chat-template-kwargs', { budgetTokens: 4096 }),
    );
    expect(opted.chat_template_kwargs).toEqual({ enable_thinking: true });
    // preserve_thinking stays absent unless explicitly configured.
    expect(opted.chat_template_kwargs).not.toHaveProperty('preserve_thinking');
    const preserved = serializeRequest(
      { ...baseOptions, model: NON_QWEN_MODEL, messages: [] },
      policy('chat-template-kwargs', { preserveThinking: true }),
    );
    expect(preserved.chat_template_kwargs).toEqual({ enable_thinking: true, preserve_thinking: true });
  });

  it('the Qwen profile (explicit modelFamily) preserves the validated kwargs behavior', () => {
    const request = serializeRequest(
      { ...baseOptions, model: 'qwen3', messages: [] },
      policy('chat-template-kwargs', { budgetTokens: 4096 }),
    );
    expect(request.chat_template_kwargs).toEqual({ enable_thinking: true });
    expect(request.thinking_budget_tokens).toBe(4096);
  });

  it('reasoning-fields mode stays available for unknown families (native, non-Qwen)', () => {
    const request = serializeRequest(
      { ...baseOptions, model: NON_QWEN_MODEL, messages: [] },
      policy('reasoning-fields', { budgetTokens: 4096 }),
    );
    expect(request.reasoning_effort).toBeUndefined(); // no effort set in this policy
    expect(request.thinking_budget_tokens).toBe(4096);
    expect(request.chat_template_kwargs).toBeUndefined();
  });
});

describe('generic llama.cpp path with a non-Qwen model name (issue #18)', () => {
  const NON_QWEN_MODEL = 'granite-3.3';

  it('serializes a request without any model-name branch', () => {
    const request = serializeRequest(
      { ...baseOptions, model: NON_QWEN_MODEL, messages: [msg('user', 'hello')] },
      { enabled: true, effort: 'medium', budgetTokens: 4096, preserveThinking: false, emitThinking: true, wire: 'chat-template-kwargs' },
    );
    expect(request.model).toBe(NON_QWEN_MODEL);
    expect(request.chat_template_kwargs).toEqual({ enable_thinking: true });
    expect(request.thinking_budget_tokens).toBe(4096);
  });

  it('lists, resolves, and streams through the adapter with a non-Qwen model name', async () => {
    // Unknown family ('auto'), non-Qwen model: full adapter path works.
    const { adapter } = harness(
      { baseURL: 'http://a', model: NON_QWEN_MODEL, modelFamily: 'auto', discovery: { enabled: true, ttlMs: 60000 } },
      [{
        id: 'c1',
        model: NON_QWEN_MODEL,
        choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: null }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      }],
    );
    const models = await adapter.listModels('llamacpp-local');
    expect(models.some((m) => m.id === NON_QWEN_MODEL)).toBe(true);
    const info = await adapter.resolveModel('llamacpp-local', NON_QWEN_MODEL);
    expect(info.id).toBe(NON_QWEN_MODEL);
    const { chunks } = await collect(adapter.stream({ ...baseOptions, model: NON_QWEN_MODEL, messages: [msg('user', 'hi')] }));
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } });
  });
});

describe('config regression guards for issue #18', () => {
  it('keeps the validated Qwen default behavior when the Qwen profile is selected', () => {
    const qwen = resolveAdapterOptions({ modelFamily: 'qwen' });
    expect(qwen.family.id).toBe('qwen');
    expect(qwen.reasoning).toMatchObject({ enabled: true, preset: 'medium', wire: 'chat-template-kwargs' });
    expect(qwen.model).toBe('qwen3');
  });
});

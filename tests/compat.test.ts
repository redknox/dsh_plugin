/**
 * Issue #18: the core adapter must not depend on undocumented Qwen
 * assumptions. Tests the model-family compatibility profile seam (explicit
 * selector only, never a model-name heuristic) and proves the generic path
 * works with a non-Qwen model name.
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

  it('the unknown profile inherits nothing Qwen-specific', () => {
    // supportsThinkingKwargs is unknown (undefined), not assumed true.
    expect(UNKNOWN_PROFILE.reasoning.supportsThinkingKwargs).toBeUndefined();
    expect(QWEN_PROFILE.reasoning.supportsThinkingKwargs).toBe(true);
    // The unknown default is the generic llama.cpp template-kwargs mechanism,
    // which is not a Qwen-only capability.
    expect(defaultReasoningWire(UNKNOWN_PROFILE)).toBe('chat-template-kwargs');
    expect(defaultReasoningWire(QWEN_PROFILE)).toBe('chat-template-kwargs');
  });

  it('resolveAdapterOptions surfaces the family and never guesses from the model name', () => {
    // A model name that looks like Qwen must NOT flip the profile: no heuristic.
    const auto = resolveAdapterOptions({ model: 'qwen3-27b' });
    expect(auto.family.id).toBe('unknown');
    expect(auto.reasoning.wire).toBe('chat-template-kwargs');
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
    const unknown = resolveAdapterOptions({ reasoning: { wire: 'reasoning-fields' } });
    expect(unknown.family.id).toBe('unknown');
    expect(unknown.reasoning.wire).toBe('reasoning-fields');
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
    const { adapter, createClient } = harness(
      { baseURL: 'http://a', model: NON_QWEN_MODEL, discovery: { enabled: true, ttlMs: 60000 } },
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
    const stream = adapter.stream({ ...baseOptions, model: NON_QWEN_MODEL, messages: [msg('user', 'hi')] });
    const { chunks } = await collect(stream);
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } });
    const wire = createClient.mock.calls[0]?.[1] as { model?: string };
    expect((wire as unknown as { model?: string })?.model).toBeUndefined(); // model rides the request body, not client options
  });

  it('never sends Qwen-only wire parameters for an unknown family unless explicitly configured', () => {
    // Unknown family, default wire: only the generic enable_thinking kwarg is
    // sent (llama.cpp's generic template-kwargs mechanism); preserve_thinking
    // — a Qwen chat-template kwarg — is absent unless configured.
    const plain = serializeRequest(
      { ...baseOptions, model: NON_QWEN_MODEL, messages: [] },
      { enabled: true, preserveThinking: false, emitThinking: true, wire: 'chat-template-kwargs' },
    );
    expect(plain.chat_template_kwargs).toEqual({ enable_thinking: true });
    // Explicit configuration may opt into it (Qwen templates, expert knob).
    const explicit = serializeRequest(
      { ...baseOptions, model: NON_QWEN_MODEL, messages: [] },
      { enabled: true, preserveThinking: true, emitThinking: true, wire: 'chat-template-kwargs' },
    );
    expect(explicit.chat_template_kwargs).toEqual({ enable_thinking: true, preserve_thinking: true });
  });
});

describe('config regression guards for issue #18', () => {
  it('keeps the validated Qwen default behavior unchanged', () => {
    // The default (no modelFamily) resolves exactly as before: unknown family,
    // wire 'chat-template-kwargs', preset medium, reasoning enabled.
    const opts = resolveAdapterOptions({});
    expect(opts.family.id).toBe('unknown');
    expect(opts.reasoning).toMatchObject({ enabled: true, preset: 'medium', wire: 'chat-template-kwargs' });
    expect(opts.model).toBe('qwen3'); // DEFAULT_MODEL untouched (validated default)
  });
});

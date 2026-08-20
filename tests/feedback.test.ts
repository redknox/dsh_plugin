/**
 * Adaptive reasoning feedback tests (issue #11): bounded outcome history with
 * decay/reset, deterministic summary and adjustment rules, the feedback
 * policy layering over the #6 seam with precedence and safety bounds, and the
 * adapter recording outcomes and applying them to the next request.
 */
import { describe, expect, it, vi } from 'vitest';
import { ReasoningEffortId, type GenerateOptions } from '@deepseek-ai/dsh-llm';
import { LlamacppAdapter, type LlamaCppChatHandle } from '../src/adapter.ts';
import { resolveAdapterOptions } from '../src/config.ts';
import {
  DEFAULT_FEEDBACK_WINDOW,
  FeedbackHistory,
  FeedbackReasoningPolicy,
  buildFeedbackPolicy,
  feedbackBudgetAdjustment,
  type ReasoningFeedback,
} from '../src/feedback.ts';
import {
  StaticReasoningPolicy,
  type ReasoningPolicy,
  type ReasoningPolicyConfig,
} from '../src/reasoning.ts';
import { collect, contentDelta } from './helpers.ts';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from '../src/protocol.ts';

function feedback(partial: Partial<ReasoningFeedback> = {}): ReasoningFeedback {
  return {
    outcome: 'success',
    retried: false,
    toolCallRetried: false,
    latencyMs: 1000,
    ...partial,
  };
}

const baseConfig: ReasoningPolicyConfig = {
  enabled: true,
  preset: 'medium',
  wire: 'chat-template-kwargs',
  feedback: { enabled: true },
};

function policyWith(history: FeedbackHistory, config: ReasoningPolicyConfig = baseConfig): ReasoningPolicy {
  return new FeedbackReasoningPolicy(new StaticReasoningPolicy(config), config, history);
}

const baseOptions: GenerateOptions = { provider: 'llamacpp-local', model: 'qwen3', messages: [] };

describe('FeedbackHistory', () => {
  it('is bounded to the window and drops stale entries (decay)', () => {
    const history = new FeedbackHistory(5);
    for (let i = 0; i < 8; i += 1) history.record(feedback({ latencyMs: i }));
    expect(history.size).toBe(5);
    // Oldest three dropped: the remaining window starts at latency 3.
    const summary = history.summarize();
    expect(summary.avgLatencyMs).toBe((3 + 4 + 5 + 6 + 7) / 5);
  });

  it('resets entirely and defaults to the documented window', () => {
    const history = new FeedbackHistory();
    expect(history.windowSize).toBe(DEFAULT_FEEDBACK_WINDOW);
    for (let i = 0; i < 25; i += 1) history.record(feedback());
    expect(history.size).toBe(DEFAULT_FEEDBACK_WINDOW);
    history.reset();
    expect(history.size).toBe(0);
    expect(history.summarize()).toMatchObject({ count: 0, failureRatio: 0, timeoutRatio: 0 });
  });

  it('summarizes deterministic ratios and reasoning averages', () => {
    const history = new FeedbackHistory();
    history.record(feedback({ outcome: 'success', reasoningTokens: 1000 }));
    history.record(feedback({ outcome: 'timeout', retried: true }));
    history.record(feedback({ outcome: 'failure', failureCode: 'SERVER', retried: true }));
    history.record(feedback({ outcome: 'success', reasoningTokens: 3000 }));
    const summary = history.summarize();
    expect(summary.count).toBe(4);
    expect(summary.failureRatio).toBe(0.5);
    expect(summary.timeoutRatio).toBe(0.25);
    expect(summary.retriedRatio).toBe(0.5);
    expect(summary.avgReasoningTokens).toBe(2000);
  });
});

describe('feedbackBudgetAdjustment', () => {
  it('reduces the budget under heavy timeouts, aborts, or failures', () => {
    const timeouts = feedbackBudgetAdjustment(4096, { count: 10, failureRatio: 1, timeoutRatio: 0.6, abortedRatio: 0, retriedRatio: 1, avgLatencyMs: 5000 });
    expect(timeouts.budgetTokens).toBe(Math.round(4096 * 0.8));
    expect(timeouts.reason).toContain('timeouts');

    const aborted = feedbackBudgetAdjustment(4096, { count: 10, failureRatio: 1, timeoutRatio: 0, abortedRatio: 0.6, retriedRatio: 0, avgLatencyMs: 1000 });
    expect(aborted.budgetTokens).toBe(Math.round(4096 * 0.85));

    const failures = feedbackBudgetAdjustment(4096, { count: 10, failureRatio: 0.8, timeoutRatio: 0, abortedRatio: 0, retriedRatio: 0, avgLatencyMs: 1000 });
    expect(failures.budgetTokens).toBe(Math.round(4096 * 0.9));
  });

  it('increases the budget when reasoning consumption is near the cap', () => {
    const result = feedbackBudgetAdjustment(4096, { count: 10, failureRatio: 0, timeoutRatio: 0, abortedRatio: 0, retriedRatio: 0, avgLatencyMs: 2000, avgReasoningTokens: 4000 });
    expect(result.budgetTokens).toBe(Math.round(4096 * 1.25));
    expect(result.reason).toContain('reasoning near budget');
  });

  it('is neutral with no history and purely deterministic', () => {
    const empty = feedbackBudgetAdjustment(4096, { count: 0, failureRatio: 0, timeoutRatio: 0, abortedRatio: 0, retriedRatio: 0, avgLatencyMs: 0 });
    expect(empty).toEqual({ budgetTokens: 4096, reason: 'no feedback history' });

    const summary = { count: 10, failureRatio: 0, timeoutRatio: 0, abortedRatio: 0, retriedRatio: 0, avgLatencyMs: 1000, avgReasoningTokens: 3000 };
    expect(feedbackBudgetAdjustment(4096, summary)).toEqual(feedbackBudgetAdjustment(4096, summary));
  });
});

describe('FeedbackReasoningPolicy', () => {
  it('behaves exactly like the base policy with no history', () => {
    const policy = policyWith(new FeedbackHistory());
    const decision = policy.resolve({});
    expect(decision.budgetTokens).toBe(4096);
    expect(decision.reason).toBeUndefined();
  });

  it('applies a deterministic reduction from a timeout-heavy history', () => {
    const history = new FeedbackHistory(10);
    for (let i = 0; i < 5; i += 1) history.record(feedback({ outcome: 'timeout', retried: true }));
    const first = policyWith(history).resolve({});
    const second = policyWith(history).resolve({});
    expect(first.budgetTokens).toBe(Math.round(4096 * 0.8));
    expect(first.reason).toMatch(/^feedback: /);
    expect(second).toEqual(first); // deterministic for identical inputs+history
  });

  it('never adjusts when an explicit per-request effort is given', () => {
    const history = new FeedbackHistory();
    for (let i = 0; i < 10; i += 1) history.record(feedback({ outcome: 'timeout' }));
    const decision = policyWith(history).resolve({ effort: ReasoningEffortId('low') });
    expect(decision.budgetTokens).toBe(1024); // low preset, untouched
    expect(decision.reason).toBeUndefined();
  });

  it('never adjusts an explicit expert budget', () => {
    const history = new FeedbackHistory();
    for (let i = 0; i < 10; i += 1) history.record(feedback({ outcome: 'timeout' }));
    const config: ReasoningPolicyConfig = { ...baseConfig, expert: { budgetTokens: 777 } };
    const decision = new FeedbackReasoningPolicy(new StaticReasoningPolicy(config), config, history).resolve({});
    expect(decision.budgetTokens).toBe(777);
    expect(decision.reason).toBeUndefined();
  });

  it('clamps feedback adjustments to the adaptive safety bounds', () => {
    const history = new FeedbackHistory();
    for (let i = 0; i < 10; i += 1) history.record(feedback({ outcome: 'timeout' }));
    const config: ReasoningPolicyConfig = {
      ...baseConfig,
      adaptive: { enabled: true, minBudgetTokens: 2000, maxBudgetTokens: 3000 },
    };
    const decision = new FeedbackReasoningPolicy(new StaticReasoningPolicy(config), config, history).resolve({});
    expect(decision.budgetTokens).toBe(3000); // 4096*0.8=3276 clamped to max 3000
    expect(decision.reason).toContain('clamped[2000,3000]');
  });

  it('builds the feedback policy only when enabled', () => {
    const history = new FeedbackHistory();
    expect(buildFeedbackPolicy({ ...baseConfig, feedback: { enabled: false } }, history)).toBeInstanceOf(StaticReasoningPolicy);
    expect(buildFeedbackPolicy(baseConfig, history)).toBeInstanceOf(FeedbackReasoningPolicy);
  });
});

describe('adapter feedback loop integration', () => {
  function clientWithReasoningUsage(reasoningTokens: number, requests: LlamaCppChatCompletionRequest[]): LlamaCppChatHandle {
    return {
      chat: vi.fn().mockImplementation(async function* (req: LlamaCppChatCompletionRequest) {
        requests.push(req);
        yield contentDelta('answer');
        yield contentDelta('', 'stop');
        yield {
          id: 'c',
          model: 'qwen3',
          choices: [],
          usage: { prompt_tokens: 100, completion_tokens: 50, completion_tokens_details: { reasoning_tokens: reasoningTokens } },
        } as LlamaCppChatCompletionChunk;
      }),
    };
  }

  it('records a success with reasoning near the budget and raises the next request', async () => {
    const options = resolveAdapterOptions({
      baseURL: 'http://127.0.0.1:8080',
      reasoning: { preset: 'medium', feedback: { enabled: true } },
    });
    const requests: LlamaCppChatCompletionRequest[] = [];
    const adapter = new LlamacppAdapter({
      options: () => options,
      resolveApiKey: async () => undefined,
      createClient: vi.fn(() => clientWithReasoningUsage(4000, requests)),
    });

    await collect(adapter.stream(baseOptions)); // records success, reasoningTokens 4000
    expect(adapter.history.size).toBe(1);
    await collect(adapter.stream(baseOptions)); // second: budget raised x1.25
    expect(adapter.history.size).toBe(2);
    // 4096 * 1.25 = 5120, within the default [512, 65536] bounds.
    expect(requests[1]?.thinking_budget_tokens).toBe(Math.round(4096 * 1.25));
  });

  it('does not record outcomes when feedback is disabled', async () => {
    const options = resolveAdapterOptions({ baseURL: 'http://127.0.0.1:8080' });
    const requests: LlamaCppChatCompletionRequest[] = [];
    const adapter = new LlamacppAdapter({
      options: () => options,
      resolveApiKey: async () => undefined,
      createClient: vi.fn(() => clientWithReasoningUsage(4000, requests)),
    });
    await collect(adapter.stream(baseOptions));
    expect(adapter.history.size).toBe(0);
  });

  it('applies a seeded timeout-heavy history to the next request budget', async () => {
    const history = new FeedbackHistory();
    for (let i = 0; i < 10; i += 1) history.record(feedback({ outcome: 'timeout' }));
    const options = resolveAdapterOptions({
      baseURL: 'http://127.0.0.1:8080',
      reasoning: { preset: 'medium', feedback: { enabled: true } },
    });
    const requests: LlamaCppChatCompletionRequest[] = [];
    const adapter = new LlamacppAdapter({
      options: () => options,
      resolveApiKey: async () => undefined,
      createClient: vi.fn(() => clientWithReasoningUsage(0, requests)),
      history,
    });
    await collect(adapter.stream(baseOptions));
    // 4096 * 0.8 = 3277, within bounds.
    expect(requests[0]?.thinking_budget_tokens).toBe(Math.round(4096 * 0.8));
    expect(adapter.history.size).toBe(11); // seeded 10 + the new success
  });
});

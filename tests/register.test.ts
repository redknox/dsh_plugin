/**
 * Registration/lifecycle tests for issue #1: the plugin must load on a real
 * Cordis context, register the `llamacpp-local` provider route and
 * configurable-provider directory entry through the public `ctx.llm` seam, and
 * unregister both when the plugin scope is disposed — without touching any
 * agent-loop internals.
 */
import { Context, type Plugin } from '@deepseek-ai/cordis';
import LlmRuntime from '@deepseek-ai/dsh-llm';
import { afterEach, describe, expect, it } from 'vitest';
import {
  Config,
  DEFAULT_PROVIDER_NAME,
  PLUGIN_NAME,
  PROVIDER,
  apply,
  type ConfigType,
} from '../src/index.ts';

/** The plugin as a Cordis `Plugin.Object`, exactly as the harness loader mounts it. */
const mountPlugin: Plugin = {
  name: PLUGIN_NAME,
  inject: ['llm'],
  Config,
  apply,
};

/** Fresh harness: one context with the public `llm` service mounted. */
function harness() {
  const ctx = new Context();
  const llmScope = ctx.plugin(LlmRuntime);
  return { ctx, llmScope };
}

const contexts: Context[] = [];

function newContext() {
  const { ctx, llmScope } = harness();
  contexts.push(ctx);
  return { ctx, llmScope };
}

afterEach(async () => {
  // Tear down every context created in this file, in reverse order.
  for (const ctx of contexts.splice(0).reverse()) {
    const task = ctx.fiber.dispose();
    if (task !== undefined && typeof task.then === 'function') await task;
  }
});

describe('llm-llamacpp plugin registration', () => {
  it('registers the provider route and directory entry', async () => {
    const { ctx, llmScope } = newContext();
    await llmScope.await();
    const scope = ctx.plugin(mountPlugin, { baseURL: 'http://127.0.0.1:8080' });
    await scope.await();

    const providers = ctx.llm.listProviders();
    expect(providers.map((p) => p.id)).toEqual([PROVIDER]);
    expect(providers[0]?.name).toBe(DEFAULT_PROVIDER_NAME);

    const configurable = ctx.llm.listConfigurableProviders();
    expect(configurable.map((p) => p.provider)).toEqual([PROVIDER]);
    expect(configurable[0]?.settingsNs).toBe(PLUGIN_NAME);
    expect(configurable[0]?.settingsPath).toEqual([]);
  });

  it('reports the resolved retry policy through the public API', async () => {
    const { ctx, llmScope } = newContext();
    await llmScope.await();
    const scope = ctx.plugin(mountPlugin, { retryPolicy: { mode: 'always' } });
    await scope.await();

    const policy = ctx.llm.providerRetryPolicy(PROVIDER);
    expect(policy.mode).toBe('always');
  });

  it('publishes the configured model through listModels and resolveModelInfo', async () => {
    const { ctx, llmScope } = newContext();
    await llmScope.await();
    const scope = ctx.plugin(mountPlugin, { model: 'qwen3-14b' });
    await scope.await();

    const models = await ctx.llm.listModels(PROVIDER);
    expect(models.map((m) => m.id)).toEqual(['qwen3-14b']);

    const info = await ctx.llm.resolveModelInfo(PROVIDER, 'qwen3-14b');
    expect(info.id).toBe('qwen3-14b');
    expect(info.provider).toBe(PROVIDER);
  });

  it('fails the plugin load clearly on an invalid baseURL', async () => {
    const { ctx, llmScope } = newContext();
    await llmScope.await();
    const scope = ctx.plugin(mountPlugin, { baseURL: 'not-a-url' });
    await expect(scope.await()).rejects.toThrow(/baseURL/);
    expect(ctx.llm.listProviders()).toHaveLength(0);
  });

  it('unregisters the provider and directory entry on dispose', async () => {
    const { ctx, llmScope } = newContext();
    await llmScope.await();
    const scope = ctx.plugin(mountPlugin, {});
    await scope.await();
    expect(ctx.llm.listProviders()).toHaveLength(1);

    await scope.dispose();
    expect(ctx.llm.listProviders()).toHaveLength(0);
    expect(ctx.llm.listConfigurableProviders()).toHaveLength(0);
  });

  it('normalizes a transport failure into a terminal error finish', async () => {
    const { ctx, llmScope } = newContext();
    await llmScope.await();
    const scope = ctx.plugin(mountPlugin, {});
    await scope.await();

    // No llama.cpp server is running: the real client fails at the fetch, and
    // LlmRuntime must turn it into a terminal finish rather than a throw.
    const chunks: unknown[] = [];
    for await (const chunk of ctx.llm.stream({
      provider: PROVIDER,
      model: 'qwen3',
      messages: [],
    })) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    const finish = chunks[0] as { type: string; reason: { kind: string; failure: { code: string } } };
    expect(finish.type).toBe('finish');
    expect(finish.reason.kind).toBe('error');
    expect(finish.reason.failure.code).toBe('TRANSPORT');
  });
});

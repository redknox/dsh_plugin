#!/usr/bin/env node
/**
 * Production diagnostics example (issue #12): mount the plugin, run one real
 * request against a llama.cpp server, and print the machine-readable snapshot
 * plus the human-readable rendering exposed via the context service
 * `llm-llamacpp/diagnostics`.
 *
 * Usage:
 *   npm run build
 *   node examples/diagnostics.mjs
 *   LLAMACPP_BASE_URL=http://10.60.84.212:8040 \
 *   LLAMACPP_MODEL=/models/Qwen3.8-27B-Q8_0.gguf \
 *   LLAMA_API_TOKEN=<token> node examples/diagnostics.mjs
 *
 * Requires a running llama.cpp server. Troubleshooting flow: inspect the
 * endpoint health/backoff rows, request counters, latency windows, and recent
 * failures without reading raw application logs.
 */
import { Context } from '@deepseek-ai/cordis';
import LlmRuntime, { createUserMessage } from '@deepseek-ai/dsh-llm';
import * as plugin from '../dist/index.js';

const baseURL = process.env.LLAMACPP_BASE_URL ?? 'http://127.0.0.1:8080';
const model = process.env.LLAMACPP_MODEL ?? 'qwen3';
const apiKeyEnv = process.env.LLAMACPP_API_KEY_ENV ?? 'LLAMA_API_TOKEN';

const ctx = new Context();
const llmScope = ctx.plugin(LlmRuntime);
await llmScope.await();
const scope = ctx.plugin(
  { name: plugin.name, inject: plugin.inject, Config: plugin.Config, apply: plugin.apply },
  { baseURL, model, apiKeyEnv },
);
await scope.await();

const diagnostics = ctx.get('llm-llamacpp/diagnostics');

try {
  console.log(`llama.cpp diagnostics example: ${baseURL} model=${model}\n`);
  for await (const chunk of ctx.llm.stream({
    provider: plugin.PROVIDER,
    model,
    messages: [createUserMessage({ content: [{ type: 'text', text: 'Say "diagnostics ok" in one short sentence.' }], source: { kind: 'user' } })],
  })) {
    if (chunk.type === 'text-delta') process.stdout.write(chunk.text);
  }
  console.log('\n');

  // Machine-readable snapshot.
  console.log('=== snapshot (machine-readable) ===');
  console.log(JSON.stringify(diagnostics.snapshot(), null, 2).slice(0, 1200));

  // Human-readable rendering.
  console.log('\n=== rendered ===');
  console.log(diagnostics.render());
} catch (error) {
  console.error(`\nfailed: ${error?.message ?? error}`);
  console.error('Is a llama.cpp server running?');
  process.exitCode = 1;
} finally {
  try {
    scope.dispose();
  } catch {
    // teardown noise ignored
  }
}

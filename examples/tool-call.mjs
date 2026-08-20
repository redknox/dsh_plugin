#!/usr/bin/env node
/**
 * End-to-end example (issue #5): a Harness-style turn with trivial local
 * tools (`get_time` / `echo`) against a real llama.cpp server.
 *
 * The provider only translates the LLM protocol: tool schemas go out, streamed
 * tool-call blocks come back, and this script executes the calls locally —
 * standing in for Harness `ctx.tools`. Tool execution itself stays out of the
 * provider.
 *
 * Usage:
 *   npm run build
 *   node examples/tool-call.mjs
 *   LLAMACPP_BASE_URL=http://127.0.0.1:8081 LLAMACPP_MODEL=qwen3 node examples/tool-call.mjs
 *
 * Requires a running llama.cpp server with a tool-capable Qwen model.
 */
import { Context } from '@deepseek-ai/cordis';
import LlmRuntime, {
  BlockAssembler,
  CallId,
  createAssistantMessage,
  createToolResultMessage,
  createUserMessage,
} from '@deepseek-ai/dsh-llm';
import * as plugin from '../dist/index.js';

const baseURL = process.env.LLAMACPP_BASE_URL ?? 'http://127.0.0.1:8080';
const model = process.env.LLAMACPP_MODEL ?? 'qwen3';
// Optional API key: name of an environment variable holding the key (e.g.
// LLAMA_API_TOKEN for llama.cpp servers started with --api-key). The plugin
// resolves it per request via apiKeyEnv.
const apiKeyEnv = process.env.LLAMACPP_API_KEY_ENV ?? 'LLAMA_API_TOKEN';

/** Trivial local tools, exactly as Harness would register them. */
const tools = [
  {
    name: 'get_time',
    description: 'Get the current local time, optionally in a given IANA timezone.',
    parameters: {
      type: 'object',
      properties: { tz: { type: 'string', description: 'IANA timezone, e.g. Asia/Shanghai' } },
      required: [],
    },
  },
  {
    name: 'echo',
    description: 'Echo the given text back verbatim.',
    parameters: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] },
  },
];

/** Execute one tool call locally, standing in for Harness ctx.tools. */
function runTool(name, args) {
  if (name === 'get_time') {
    const now = new Date();
    if (args.tz === undefined) return `The local time is ${now.toLocaleTimeString()}.`;
    return `The time in ${args.tz} is ${now.toLocaleTimeString('en-US', { timeZone: args.tz })}.`;
  }
  if (name === 'echo') return args.text;
  return `unknown tool: ${name}`;
}

/** Stream one turn through the provider, assembling the assistant message. */
async function streamTurn(messages, offeredTools) {
  const assembler = new BlockAssembler();
  for await (const chunk of ctx.llm.stream({
    provider: plugin.PROVIDER,
    model,
    messages,
    ...(offeredTools !== undefined ? { tools: offeredTools } : {}),
  })) {
    assembler.push(chunk);
    if (chunk.type === 'text-delta') process.stdout.write(chunk.text);
    else if (chunk.type === 'reasoning-delta') process.stdout.write(`\x1b[2m${chunk.text}\x1b[0m`);
  }
  return {
    message: assembler.message({ kind: 'model', provider: plugin.PROVIDER, model }),
    finish: assembler.finish,
    usage: assembler.usage,
  };
}

let ctx;
let scope;
try {
  ctx = new Context();
  const llmScope = ctx.plugin(LlmRuntime);
  await llmScope.await();
  scope = ctx.plugin(
    { name: plugin.name, inject: plugin.inject, Config: plugin.Config, apply: plugin.apply },
    { baseURL, model, apiKeyEnv, modelFamily: 'qwen' }, // issue #18: these examples validate the Qwen profile explicitly
  );
  await scope.await();

  console.log(`llama.cpp e2e tool example: ${baseURL} model=${model}\n`);

  const history = [createUserMessage({
    content: [{ type: 'text', text: 'What time is it in Asia/Shanghai? Also echo the word "hello".' }],
    source: { kind: 'user' },
  })];

  const first = await streamTurn(history, tools);
  console.log(`\nfinish: ${first.finish.kind}${first.usage ? ` | usage: ${JSON.stringify(first.usage)}` : ''}`);
  if (first.finish.kind === 'error' || first.finish.kind === 'aborted') {
    console.error(`\nturn failed: ${JSON.stringify(first.finish)}`);
    console.error('Is a llama.cpp server running with a tool-capable Qwen model?');
    process.exitCode = 1;
    process.exit();
  }

  const toolCalls = first.message.content.filter((block) => block.type === 'tool-call');
  if (toolCalls.length === 0) {
    console.log('The model answered directly without calling a tool.');
    process.exit(0);
  }

  history.push(createAssistantMessage({ content: first.message.content, source: { provider: plugin.PROVIDER, model } }));
  for (const call of toolCalls) {
    const args = JSON.parse(call.arguments);
    const result = runTool(call.name, args);
    console.log(`\ntool ${call.name}(${JSON.stringify(args)}) -> ${result}`);
    history.push(createToolResultMessage({
      callId: CallId(call.id),
      content: [{ type: 'text', text: result }],
      isError: false,
    }));
  }

  console.log('\n--- second turn: tool results fed back to the model ---');
  const second = await streamTurn(history, tools);
  console.log(`\nfinish: ${second.finish.kind}`);
} catch (error) {
  console.error(`\nfailed: ${error?.message ?? error}`);
  console.error('Is a llama.cpp server running with a tool-capable Qwen model?');
  process.exitCode = 1;
} finally {
  try {
    scope?.dispose();
  } catch {
    // teardown noise ignored
  }
}

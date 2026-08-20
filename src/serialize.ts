/**
 * Serialize Harness `GenerateOptions` into llama.cpp OpenAI-compatible chat
 * completions requests. Harness messages become wire messages; user text is
 * joined, assistant text becomes `content` with `tool_calls` preserved, and
 * tool results become separate `role: 'tool'` messages. Options that this
 * milestone does not support (tools, reasoning effort) are rejected explicitly
 * instead of being silently dropped.
 *
 * @module llm-llamacpp/serialize
 */
import {
  LlmError,
  contentHasImage,
  type GenerateOptions,
  type Message,
} from '@deepseek-ai/dsh-llm';
import type { ResolvedReasoningPolicy } from './reasoning.ts';
import type {
  LlamaCppChatCompletionRequest,
  LlamaCppChatMessage,
  LlamaCppToolCall,
} from './protocol.ts';

/** Join the text blocks of a message (used for user/tool-result content). */
function flattenText(blocks: Message['content']): string {
  return blocks
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('');
}

/**
 * Reject content this text-only wire route cannot represent, before any
 * flattening can silently erase it. Images (including inside tool results)
 * fail via the public `contentHasImage` helper; unknown declaration-merged
 * block types fail explicitly too, since they have no wire representation
 * here. Reasoning, tool-call, and tool-result blocks are handled by the
 * serializer itself and pass.
 */
function assertSupportedContent(blocks: readonly Message['content'][number][]): void {
  if (contentHasImage(blocks)) {
    throw new LlmError(
      'llm-llamacpp: image content is not supported (text-only wire route)',
      'UNSUPPORTED_CONTENT',
    );
  }
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
      case 'reasoning':
      case 'tool-call':
      case 'tool-result':
        break;
      default:
        throw new LlmError(
          `llm-llamacpp: unsupported content block type "${block.type}"`,
          'UNSUPPORTED_CONTENT',
        );
    }
  }
}

/**
 * Serialize one assistant message. Text becomes `content`; tool-call blocks
 * become OpenAI `tool_calls`; reasoning blocks are intentionally not replayed
 * (the llama.cpp/Qwen chat template regenerates thinking).
 */
function serializeAssistant(content: Message['content']): LlamaCppChatMessage {
  const text = flattenText(content);
  const toolCalls: LlamaCppToolCall[] = content
    .filter((block) => block.type === 'tool-call')
    .map((block) => ({
      id: block.id,
      type: 'function',
      function: { name: block.name, arguments: block.arguments },
    }));
  return {
    role: 'assistant',
    content: text,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };
}

/**
 * Serialize the conversation. `tool-result` blocks become standalone
 * `{ role: 'tool' }` messages; a mixed user message contributes its text first
 * and its tool results as separate wire messages after.
 * @param messages - the harness conversation, in order.
 * @returns the wire messages; order preserved, each tool result expanded into its own entry.
 */
export function serializeMessages(messages: readonly Message[]): LlamaCppChatMessage[] {
  const wire: LlamaCppChatMessage[] = [];
  for (const message of messages) {
    assertSupportedContent(message.content);
    if (message.role === 'system') {
      wire.push({ role: 'system', content: flattenText(message.content) });
      continue;
    }
    if (message.role === 'assistant') {
      wire.push(serializeAssistant(message.content));
      continue;
    }
    const toolResults = message.content.filter((block) => block.type === 'tool-result');
    const text = flattenText(message.content);
    if (text.length > 0 || toolResults.length === 0) {
      wire.push({ role: 'user', content: text });
    }
    for (const result of toolResults) {
      wire.push({
        role: 'tool',
        tool_call_id: result.toolCallId,
        content: flattenText(result.content) || '(no output)',
      });
    }
  }
  return wire;
}

/**
 * Translate a resolved reasoning policy into llama.cpp request fields. This is
 * the only layer that maps semantic reasoning to wire fields.
 *
 * Wire-mode semantics (issue #18): `chat-template-kwargs` sends the Qwen-style
 * `enable_thinking` / `preserve_thinking` template kwargs — only for a model
 * family that declares support (the Qwen profile) or when the user explicitly
 * configured that wire mode; `reasoning-fields` sends the llama.cpp-native
 * `reasoning_effort` / `thinking_budget_tokens` per-request fields; `none`
 * sends no reasoning wire fields at all — the default for families whose
 * template-kwargs support is unknown, so nothing Qwen-oriented is ever sent
 * silently.
 *
 * Version dependence: `chat_template_kwargs.enable_thinking` /
 * `chat_template_kwargs.preserve_thinking` are Qwen chat-template kwargs,
 * honored by llama.cpp builds with the per-request template-kwargs hook
 * (llama.cpp PR #13196). The runtime thinking budget is a separate llama.cpp
 * inference control and travels as the top-level `thinking_budget_tokens`
 * per-request field (the server's OpenAI-compatible spelling; newer builds
 * also accept `reasoning_budget_tokens` as an alias, PRs #22336/#23116).
 * Newer builds additionally honor `reasoning_effort` (including `"none"`,
 * PR #26045); select the `reasoning-fields` wire mode on those builds.
 * `preserve_thinking` is merged from `chat_template_kwargs` independently of
 * those native fields, so it can ride alongside them in either mode.
 */
function applyReasoningToRequest(request: LlamaCppChatCompletionRequest, policy: ResolvedReasoningPolicy): void {
  if (policy.wire === 'none') return;
  if (policy.wire === 'chat-template-kwargs') {
    request.chat_template_kwargs = policy.enabled
      ? { enable_thinking: true, ...(policy.preserveThinking ? { preserve_thinking: true } : {}) }
      : { enable_thinking: false };
    if (policy.enabled && policy.budgetTokens !== undefined) {
      request.thinking_budget_tokens = policy.budgetTokens;
    }
    return;
  }
  // 'reasoning-fields' wire mode: native per-request reasoning fields.
  if (!policy.enabled) {
    request.reasoning_effort = 'none';
    return;
  }
  if (policy.effort !== undefined) request.reasoning_effort = policy.effort;
  if (policy.budgetTokens !== undefined) request.thinking_budget_tokens = policy.budgetTokens;
  // `preserve_thinking` is a Qwen chat-template kwarg that llama.cpp merges
  // from `chat_template_kwargs` independently of the native reasoning fields,
  // so the expert knob can never be a silent no-op in this mode either.
  if (policy.preserveThinking) {
    request.chat_template_kwargs = { preserve_thinking: true };
  }
}

/**
 * Build the full wire request. Always streaming with usage reporting on;
 * optional fields are omitted rather than sent as null so provider defaults
 * apply. This milestone explicitly rejects unsupported options (tools).
 * @param options - the harness request (model, history, system, sampling).
 * @param reasoning - the resolved reasoning policy for this request.
 * @returns the chat-completions request body.
 */
export function serializeRequest(
  options: GenerateOptions,
  reasoning: ResolvedReasoningPolicy,
): LlamaCppChatCompletionRequest {
  const messages: LlamaCppChatMessage[] = [];
  if (options.system !== undefined) messages.push({ role: 'system', content: options.system });
  messages.push(...serializeMessages(options.messages));
  const tools = options.tools !== undefined && options.tools.length > 0
    ? options.tools.map((tool) => ({
        type: 'function' as const,
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      }))
    : undefined;
  const request: LlamaCppChatCompletionRequest = {
    model: options.model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    ...(tools !== undefined ? { tools } : {}),
    ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
    ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
    ...(options.stop !== undefined ? { stop: options.stop } : {}),
  };
  applyReasoningToRequest(request, reasoning);
  return request;
}

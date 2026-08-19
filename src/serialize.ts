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
import { LlmError, type GenerateOptions, type Message } from '@deepseek-ai/dsh-llm';
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
 * Version dependence: `chat_template_kwargs.enable_thinking` /
 * `chat_template_kwargs.thinking_budget` are honored by llama.cpp builds that
 * ship the per-request template-kwargs hook for Qwen3 templates (PR #13196),
 * across Qwen3-era versions; semantic `effort` has no template knob there, so
 * it maps to the token budget instead. Newer builds additionally honor
 * top-level `reasoning_effort` (including `"none"`) and
 * `reasoning_budget_tokens` (PRs #22336/#23116/#26045); select the
 * `reasoning-fields` wire mode on those builds.
 */
function applyReasoningToRequest(request: LlamaCppChatCompletionRequest, policy: ResolvedReasoningPolicy): void {
  if (policy.wire === 'chat-template-kwargs') {
    request.chat_template_kwargs = policy.enabled
      ? { enable_thinking: true, ...(policy.budgetTokens !== undefined ? { thinking_budget: policy.budgetTokens } : {}) }
      : { enable_thinking: false };
    return;
  }
  // 'reasoning-fields' wire mode: native per-request reasoning fields.
  if (!policy.enabled) {
    request.reasoning_effort = 'none';
    return;
  }
  if (policy.effort !== undefined) request.reasoning_effort = policy.effort;
  if (policy.budgetTokens !== undefined) request.reasoning_budget_tokens = policy.budgetTokens;
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

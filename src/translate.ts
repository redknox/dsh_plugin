/**
 * Translate llama.cpp wire chunks into the Harness `StreamChunk` protocol.
 * One stateful harness block per text, reasoning, or tool-call delta;
 * fragmented tool-call ids, function names, and JSON argument fragments are
 * accumulated per wire `call.index`; finish reason and the latest usage are
 * deferred to stream end so no chunk follows `finish`.
 *
 * @module llm-llamacpp/translate
 */
import {
  CallId,
  EMPTY_RESPONSE_CODE,
  LlmError,
  type FinishReason,
  type StreamChunk,
  type TokenUsage,
} from '@deepseek-ai/dsh-llm';
import type { LlamaCppChatCompletionChunk, LlamaCppUsage } from './protocol.ts';

/** Map the wire finish_reason vocabulary to the harness FinishReason. */
function mapFinishReason(reason: string): FinishReason {
  switch (reason) {
    case 'stop':
      return { kind: 'stop' };
    case 'tool_calls':
      return { kind: 'tool-calls' };
    case 'length':
      return { kind: 'max-tokens' };
    default:
      return {
        kind: 'error',
        failure: {
          message: `model stopped: ${reason}`,
          code: reason.toUpperCase(),
        },
      };
  }
}

/**
 * Map wire usage fields to the harness TokenUsage convention (DISJOINT counts:
 * cached input is reported separately and subtracted from `inputTokens`).
 */
function mapUsage(usage: LlamaCppUsage): TokenUsage {
  const cacheRead = usage.prompt_tokens_details?.cached_tokens;
  const reasoning = usage.completion_tokens_details?.reasoning_tokens;
  return {
    inputTokens: usage.prompt_tokens - (cacheRead ?? 0),
    outputTokens: usage.completion_tokens,
    ...(cacheRead !== undefined ? { cacheReadTokens: cacheRead } : {}),
    ...(reasoning !== undefined ? { reasoningTokens: reasoning } : {}),
  };
}

/** One open harness block, closed at stream end. */
type OpenBlock =
  | { readonly index: number; readonly kind: 'text'; text: string }
  | { readonly index: number; readonly kind: 'reasoning'; text: string }
  | {
      readonly index: number;
      readonly kind: 'tool-call';
      /** Accumulated provider-issued call id (may arrive in a later frame). */
      callId?: string;
      /** Accumulated function name (may arrive in a later frame). */
      name?: string;
      /** Accumulated raw JSON argument string. */
      text: string;
    };

/** One open tool-call harness block. */
type ToolCallOpenBlock = Extract<OpenBlock, { kind: 'tool-call' }>;

/**
 * Consume parsed wire chunks and yield StreamChunks as they arrive; block-end,
 * usage, and finish are deferred to stream end. A `stop` (or absent) finish
 * with no opened blocks is a degenerate completion and maps to an
 * `EMPTY_RESPONSE` error finish. Tool-call argument strings must parse as JSON
 * by stream end, otherwise the stream fails with `INVALID_TOOL_ARGUMENTS`.
 * @param chunks - parsed wire chunks from the client (already JSON-decoded).
 * @param options - `preserveThinking: false` consumes reasoning deltas without
 *   emitting reasoning blocks (the `preserveThinking` expert knob).
 */
export async function* translate(
  chunks: AsyncIterable<LlamaCppChatCompletionChunk>,
  options: { preserveThinking?: boolean } = {},
): AsyncIterable<StreamChunk> {
  const preserveThinking = options.preserveThinking ?? true;
  let nextIndex = 0;
  let textBlock: OpenBlock | undefined;
  let reasoningBlock: OpenBlock | undefined;
  const toolBlocks = new Map<number, ToolCallOpenBlock>();
  const order: OpenBlock[] = [];
  let pendingFinish: FinishReason | undefined;
  let pendingUsage: TokenUsage | undefined;

  for await (const chunk of chunks) {
    for (const choice of chunk.choices ?? []) {
      const delta = choice.delta;
      const reasoning = delta.reasoning_content;
      if (typeof reasoning === 'string' && reasoning.length > 0) {
        if (preserveThinking) {
          if (reasoningBlock === undefined) {
            reasoningBlock = { index: nextIndex++, kind: 'reasoning', text: '' };
            order.push(reasoningBlock);
            yield { type: 'block-start', index: reasoningBlock.index, blockType: 'reasoning' };
          }
          reasoningBlock.text += reasoning;
          yield { type: 'reasoning-delta', index: reasoningBlock.index, text: reasoning };
        }
      }
      const content = delta.content;
      if (typeof content === 'string' && content.length > 0) {
        if (textBlock === undefined) {
          textBlock = { index: nextIndex++, kind: 'text', text: '' };
          order.push(textBlock);
          yield { type: 'block-start', index: textBlock.index, blockType: 'text' };
        }
        textBlock.text += content;
        yield { type: 'text-delta', index: textBlock.index, text: content };
      }
      for (const call of delta.tool_calls ?? []) {
        const wireIndex = call.index ?? 0;
        let block = toolBlocks.get(wireIndex);
        if (block === undefined) {
          block = { index: nextIndex++, kind: 'tool-call', text: '' };
          toolBlocks.set(wireIndex, block);
          order.push(block);
          yield { type: 'block-start', index: block.index, blockType: 'tool-call' };
        }
        if (call.id !== undefined) block.callId = call.id;
        if (call.function?.name !== undefined) {
          // Function names arrive as fragments like arguments: accumulate.
          block.name = (block.name ?? '') + call.function.name;
        }
        const fragment = call.function?.arguments ?? '';
        block.text += fragment;
        yield {
          type: 'tool-call-delta',
          index: block.index,
          id: CallId(block.callId ?? ''),
          ...(block.name !== undefined ? { name: block.name } : {}),
          argumentsDelta: fragment,
        };
      }
      if (typeof choice.finish_reason === 'string') {
        pendingFinish = mapFinishReason(choice.finish_reason);
      }
    }
    if (chunk.usage !== undefined) pendingUsage = mapUsage(chunk.usage);
  }

  for (const block of order) {
    if (block.kind === 'tool-call') {
      try {
        JSON.parse(block.text);
      } catch {
        throw new LlmError(
          `llama.cpp streamed malformed tool arguments for "${block.name ?? ''}": ${block.text.slice(0, 120)}`,
          'INVALID_TOOL_ARGUMENTS',
        );
      }
      yield {
        type: 'block-end',
        index: block.index,
        block: {
          type: 'tool-call',
          id: CallId(block.callId ?? ''),
          name: block.name ?? '',
          arguments: block.text,
        },
      };
    } else {
      yield {
        type: 'block-end',
        index: block.index,
        block: block.kind === 'text'
          ? { type: 'text', text: block.text }
          : { type: 'reasoning', text: block.text },
      };
    }
  }
  if (pendingUsage !== undefined) yield { type: 'usage', usage: pendingUsage };
  const reason = pendingFinish ?? { kind: 'stop' };
  yield {
    type: 'finish',
    reason: reason.kind === 'stop' && order.length === 0
      ? {
          kind: 'error',
          failure: {
            message: 'llama.cpp returned a completed response with no content',
            code: EMPTY_RESPONSE_CODE,
          },
        }
      : reason,
  };
}

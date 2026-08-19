/**
 * Translate llama.cpp wire chunks into the Harness `StreamChunk` protocol.
 * One stateful harness block per content or reasoning delta; finish reason and
 * the latest usage are deferred to stream end so no chunk follows `finish`.
 *
 * @module llm-llamacpp/translate
 */
import {
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

interface OpenBlock {
  readonly index: number;
  readonly kind: 'text' | 'reasoning';
  text: string;
}

/**
 * Consume parsed wire chunks and yield StreamChunks as they arrive; block-end,
 * usage, and finish are deferred to stream end. A `stop` (or absent) finish
 * with no opened blocks is a degenerate completion and maps to an
 * `EMPTY_RESPONSE` error finish. Streamed tool-call deltas are rejected
 * explicitly until issue #5 implements them.
 * @param chunks - parsed wire chunks from the client (already JSON-decoded).
 */
export async function* translate(
  chunks: AsyncIterable<LlamaCppChatCompletionChunk>,
): AsyncIterable<StreamChunk> {
  let nextIndex = 0;
  let textBlock: OpenBlock | undefined;
  let reasoningBlock: OpenBlock | undefined;
  const order: OpenBlock[] = [];
  let pendingFinish: FinishReason | undefined;
  let pendingUsage: TokenUsage | undefined;

  for await (const chunk of chunks) {
    for (const choice of chunk.choices ?? []) {
      const delta = choice.delta;
      const reasoning = delta.reasoning_content;
      if (typeof reasoning === 'string' && reasoning.length > 0) {
        if (reasoningBlock === undefined) {
          reasoningBlock = { index: nextIndex++, kind: 'reasoning', text: '' };
          order.push(reasoningBlock);
          yield { type: 'block-start', index: reasoningBlock.index, blockType: 'reasoning' };
        }
        reasoningBlock.text += reasoning;
        yield { type: 'reasoning-delta', index: reasoningBlock.index, text: reasoning };
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
      if (delta.tool_calls !== undefined && delta.tool_calls.length > 0) {
        throw new LlmError(
          'llm-llamacpp: streamed tool calls are not supported yet (issue #5)',
          'UNSUPPORTED_OPTION',
        );
      }
      if (typeof choice.finish_reason === 'string') {
        pendingFinish = mapFinishReason(choice.finish_reason);
      }
    }
    if (chunk.usage !== undefined) pendingUsage = mapUsage(chunk.usage);
  }

  for (const block of order) {
    yield {
      type: 'block-end',
      index: block.index,
      block: block.kind === 'text'
        ? { type: 'text', text: block.text }
        : { type: 'reasoning', text: block.text },
    };
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

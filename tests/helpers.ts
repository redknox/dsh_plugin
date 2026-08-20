/**
 * Shared vitest helpers for adapter/tools suites: a harness that constructs a
 * `LlamacppAdapter` with a mocked transport client (capturing requests and
 * client options), wire-chunk builders, and stream collection.
 */
import { vi } from 'vitest';
import type { GenerateOptions, Message, StreamChunk } from '@deepseek-ai/dsh-llm';
import { LlamacppAdapter, type LlamaCppChatHandle } from '../src/adapter.ts';
import { resolveAdapterOptions, type ResolvedAdapterOptions } from '../src/config.ts';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from '../src/protocol.ts';

/** Build one wire chunk from a partial. */
export function chunk(partial: Partial<LlamaCppChatCompletionChunk> & Pick<LlamaCppChatCompletionChunk, 'choices'>): LlamaCppChatCompletionChunk {
  return { id: 'chatcmpl-1', model: 'qwen3', ...partial };
}

export function contentDelta(content: string, finishReason: string | null = null): LlamaCppChatCompletionChunk {
  return chunk({ choices: [{ index: 0, delta: { content }, finish_reason: finishReason }] });
}

export function usageChunk(): LlamaCppChatCompletionChunk {
  return chunk({
    choices: [],
    usage: {
      prompt_tokens: 12,
      completion_tokens: 5,
      prompt_tokens_details: { cached_tokens: 4 },
      completion_tokens_details: { reasoning_tokens: 2 },
    },
  });
}

/** Collect a full chunk stream into an array, capturing throws. */
export async function collect(stream: AsyncIterable<StreamChunk>): Promise<{ chunks: StreamChunk[]; error?: unknown }> {
  const chunks: StreamChunk[] = [];
  try {
    for await (const chunk of stream) chunks.push(chunk);
    return { chunks };
  } catch (error) {
    return { chunks, error };
  }
}

export type ChunkSource =
  | Iterable<LlamaCppChatCompletionChunk>
  | AsyncIterable<LlamaCppChatCompletionChunk>
  | (() => Iterable<LlamaCppChatCompletionChunk> | AsyncIterable<LlamaCppChatCompletionChunk>);

/** Normalize an iterable or async-iterable chunk source into an async iterable. */
export async function* toAsync(source: Iterable<LlamaCppChatCompletionChunk> | AsyncIterable<LlamaCppChatCompletionChunk>): AsyncIterable<LlamaCppChatCompletionChunk> {
  if (Symbol.asyncIterator in source) yield* source as AsyncIterable<LlamaCppChatCompletionChunk>;
  else for (const item of source as Iterable<LlamaCppChatCompletionChunk>) yield item;
}

export interface HarnessResult {
  adapter: LlamacppAdapter;
  fakeChat: ReturnType<typeof vi.fn>;
  createClient: ReturnType<typeof vi.fn>;
  calls: Array<{ baseURL: string; clientOptions: unknown }>;
}

/**
 * Build an adapter whose transport client is a recording fake. The harness
 * defaults to the validated Qwen family profile (`modelFamily: 'qwen'`) so
 * adapter-level tests keep exercising the Qwen chat-template wire behavior;
 * issue #18 generic-path tests override it with `modelFamily: 'auto'`.
 */
export function harness(
  config: Record<string, unknown> = {},
  chunks?: ChunkSource,
  logger?: { debug: (message: string) => void },
): HarnessResult {
  const options = () => resolveAdapterOptions({ modelFamily: 'qwen', ...config }) as ResolvedAdapterOptions;
  const fakeChat = vi.fn().mockImplementation(() => toAsync(typeof chunks === 'function' ? chunks() : (chunks ?? [])));
  const fakeClient: LlamaCppChatHandle = { chat: fakeChat };
  const calls: Array<{ baseURL: string; clientOptions: unknown }> = [];
  const createClient = vi.fn((baseURL: string, clientOptions: unknown) => {
    calls.push({ baseURL, clientOptions });
    return fakeClient;
  });
  const resolveApiKey = async () => {
    const opts = options();
    if (opts.apiKeyEnv === undefined) return undefined;
    return process.env[opts.apiKeyEnv];
  };
  const adapter = new LlamacppAdapter({
    options,
    resolveApiKey,
    createClient,
    ...(logger !== undefined ? { logger } : {}),
  });
  return { adapter, fakeChat, createClient, calls };
}

export const baseOptions: GenerateOptions = {
  provider: 'llamacpp-local',
  model: 'qwen3',
  messages: [],
};

export function msg(role: 'system' | 'user' | 'assistant', text: string): Message {
  return {
    id: `msg-${role}-${text}` as Message['id'],
    role,
    content: [{ type: 'text', text }],
    source: role === 'assistant'
      ? { kind: 'model', provider: 'llamacpp-local', model: 'qwen3' }
      : { kind: 'user' },
  };
}

/** The last request the fake client received. */
export function lastRequest(fakeChat: ReturnType<typeof vi.fn>): LlamaCppChatCompletionRequest {
  return fakeChat.mock.calls.at(-1)?.[0] as LlamaCppChatCompletionRequest;
}

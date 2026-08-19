/**
 * Adapter translation tests for issue #3: `GenerateOptions` → llama.cpp wire
 * request, wire chunks → Harness `StreamChunk`s, and explicit rejection of
 * unsupported options. The transport client is mocked via `createClient`, so
 * no HTTP happens here; the plugin-level path is covered by integration.test.ts.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  CallId,
  LlmError,
  type GenerateOptions,
  type Message,
  type StreamChunk,
} from '@deepseek-ai/dsh-llm';
import { LlamacppAdapter, type LlamaCppChatHandle } from '../src/adapter.ts';
import { resolveAdapterOptions, type ResolvedAdapterOptions } from '../src/config.ts';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from '../src/protocol.ts';

/** Build one wire chunk from a partial. */
function chunk(partial: Partial<LlamaCppChatCompletionChunk> & Pick<LlamaCppChatCompletionChunk, 'choices'>): LlamaCppChatCompletionChunk {
  return { id: 'chatcmpl-1', model: 'qwen3', ...partial };
}

function contentDelta(content: string, finishReason: string | null = null): LlamaCppChatCompletionChunk {
  return chunk({ choices: [{ index: 0, delta: { content }, finish_reason: finishReason }] });
}

function usageChunk(): LlamaCppChatCompletionChunk {
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
async function collect(stream: AsyncIterable<StreamChunk>): Promise<{ chunks: StreamChunk[]; error?: unknown }> {
  const chunks: StreamChunk[] = [];
  try {
    for await (const chunk of stream) chunks.push(chunk);
    return { chunks };
  } catch (error) {
    return { chunks, error };
  }
}

const KEY_ENV = 'LLAMACPP_TEST_KEY';
const originalEnv = process.env[KEY_ENV];

afterEach(() => {
  vi.restoreAllMocks();
  if (originalEnv === undefined) delete process.env[KEY_ENV];
  else process.env[KEY_ENV] = originalEnv;
});

type ChunkSource =
  | Iterable<LlamaCppChatCompletionChunk>
  | AsyncIterable<LlamaCppChatCompletionChunk>
  | (() => Iterable<LlamaCppChatCompletionChunk> | AsyncIterable<LlamaCppChatCompletionChunk>);

/** Normalize an iterable or async-iterable chunk source into an async iterable. */
async function* toAsync(source: Iterable<LlamaCppChatCompletionChunk> | AsyncIterable<LlamaCppChatCompletionChunk>): AsyncIterable<LlamaCppChatCompletionChunk> {
  if (Symbol.asyncIterator in source) yield* source as AsyncIterable<LlamaCppChatCompletionChunk>;
  else for (const item of source as Iterable<LlamaCppChatCompletionChunk>) yield item;
}

function harness(config: Record<string, unknown> = {}, chunks?: ChunkSource) {
  const options = () => resolveAdapterOptions(config) as ResolvedAdapterOptions;
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
  const adapter = new LlamacppAdapter({ options, resolveApiKey, createClient });
  return { adapter, fakeChat, createClient, calls };
}

const baseOptions: GenerateOptions = {
  provider: 'llamacpp-local',
  model: 'qwen3',
  messages: [],
};

function msg(role: 'system' | 'user' | 'assistant', text: string): Message {
  return {
    id: `msg-${role}-${text}` as Message['id'],
    role,
    content: [{ type: 'text', text }],
    source: role === 'assistant'
      ? { kind: 'model', provider: 'llamacpp-local', model: 'qwen3' }
      : { kind: 'user' },
  };
}

describe('LlamacppAdapter request serialization', () => {
  it('maps model, system, history, and sampling options onto the wire request', async () => {
    const { adapter, fakeChat } = harness({ baseURL: 'http://127.0.0.1:8080' });
    const request: GenerateOptions = {
      ...baseOptions,
      model: 'qwen3-14b',
      system: 'You are helpful.',
      messages: [msg('user', 'Hello'), msg('assistant', 'Hi!'), msg('user', 'Again')],
      temperature: 0.7,
      maxTokens: 512,
      stop: ['\n\n', 'END'],
    };
    await collect(adapter.stream(request));
    const wire = fakeChat.mock.calls[0]?.[0] as LlamaCppChatCompletionRequest;
    expect(wire).toMatchObject({
      model: 'qwen3-14b',
      stream: true,
      stream_options: { include_usage: true },
      temperature: 0.7,
      max_tokens: 512,
      stop: ['\n\n', 'END'],
      messages: [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi!' },
        { role: 'user', content: 'Again' },
      ],
    });
  });

  it('serializes tool results as role:tool messages and assistant tool_calls', async () => {
    const { adapter, fakeChat } = harness({});
    const toolResultMessage: Message = {
      id: 'msg-tool' as Message['id'],
      role: 'user',
      content: [{ type: 'tool-result', toolCallId: CallId('call_1'), content: [{ type: 'text', text: '42' }] }],
      source: { kind: 'tool', callId: CallId('call_1') },
    };
    const assistantWithTools: Message = {
      id: 'msg-asst-tools' as Message['id'],
      role: 'assistant',
      content: [
        { type: 'text', text: 'Let me check.' },
        { type: 'tool-call', id: CallId('call_1'), name: 'get_time', arguments: '{}' },
      ],
      source: { kind: 'model', provider: 'llamacpp-local', model: 'qwen3' },
    };
    await collect(adapter.stream({ ...baseOptions, messages: [assistantWithTools, toolResultMessage] }));
    const wire = fakeChat.mock.calls[0]?.[0] as LlamaCppChatCompletionRequest;
    expect(wire.messages).toEqual([
      { role: 'assistant', content: 'Let me check.', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_time', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: 'call_1', content: '42' },
    ]);
  });

  it('attaches Bearer auth on the authorization header when a key is configured', async () => {
    process.env[KEY_ENV] = 'secret-key';
    const { adapter, createClient } = harness({ apiKeyEnv: KEY_ENV });
    await collect(adapter.stream(baseOptions));
    expect(createClient).toHaveBeenCalledWith(
      'http://127.0.0.1:8080',
      expect.objectContaining({ auth: { name: 'authorization', value: 'Bearer secret-key' } }),
    );
  });

  it('attaches the raw key on a custom auth header', async () => {
    process.env[KEY_ENV] = 'sk-raw';
    const { adapter, createClient } = harness({ apiKeyEnv: KEY_ENV, apiKeyHeader: 'x-api-key' });
    await collect(adapter.stream(baseOptions));
    expect(createClient).toHaveBeenCalledWith(
      'http://127.0.0.1:8080',
      expect.objectContaining({ auth: { name: 'x-api-key', value: 'sk-raw' } }),
    );
  });

  it('sends no auth when no key is configured', async () => {
    const { adapter, createClient } = harness({});
    await collect(adapter.stream(baseOptions));
    expect(createClient).toHaveBeenCalledWith('http://127.0.0.1:8080', { streamIdleTimeoutMs: 300_000 });
  });

  it('forwards the caller signal to the client', async () => {
    const { adapter, fakeChat } = harness({});
    const controller = new AbortController();
    await collect(adapter.stream({ ...baseOptions, signal: controller.signal }));
    expect(fakeChat).toHaveBeenCalledWith(expect.anything(), { signal: controller.signal });
  });

  it('serializes tool schemas onto the wire (issue #5)', async () => {
    const { adapter, fakeChat } = harness({});
    await collect(adapter.stream({
      ...baseOptions,
      tools: [{ name: 'get_time', description: 'Get the time', parameters: { type: 'object' } }],
    }));
    const wire = fakeChat.mock.calls[0]?.[0] as LlamaCppChatCompletionRequest;
    expect(wire.tools).toEqual([
      { type: 'function', function: { name: 'get_time', description: 'Get the time', parameters: { type: 'object' } } },
    ]);
  });

  it('rejects image content explicitly instead of silently erasing it', async () => {
    const { adapter, fakeChat } = harness({});
    const imageMessage: Message = {
      id: 'msg-img' as Message['id'],
      role: 'user',
      content: [
        { type: 'text', text: 'look at this: ' },
        { type: 'image', attachment: { ref: 'attachment://img' } as never },
      ],
      source: { kind: 'user' },
    };
    const { error } = await collect(adapter.stream({ ...baseOptions, messages: [imageMessage] }));
    expect((error as LlmError).code).toBe('UNSUPPORTED_CONTENT');
    expect(fakeChat).not.toHaveBeenCalled();
  });

  it('rejects unknown content block types explicitly', async () => {
    const { adapter, fakeChat } = harness({});
    const oddMessage: Message = {
      id: 'msg-odd' as Message['id'],
      role: 'user',
      content: [{ type: 'audio', whatever: true } as never],
      source: { kind: 'user' },
    };
    const { error } = await collect(adapter.stream({ ...baseOptions, messages: [oddMessage] }));
    expect((error as LlmError).code).toBe('UNSUPPORTED_CONTENT');
    expect(fakeChat).not.toHaveBeenCalled();
  });

  it('rejects an unsupported reasoning effort explicitly', async () => {
    const { adapter, fakeChat } = harness({});
    const { error } = await collect(adapter.stream({ ...baseOptions, reasoningEffort: 'high' as never }));
    expect((error as LlmError).code).toBe('UNSUPPORTED_REASONING_EFFORT');
    expect(fakeChat).not.toHaveBeenCalled();
  });

  it('translates the default reasoning preset onto the wire (chat-template-kwargs)', async () => {
    const { adapter, fakeChat } = harness({});
    await collect(adapter.stream(baseOptions));
    const wire = fakeChat.mock.calls[0]?.[0] as LlamaCppChatCompletionRequest;
    expect(wire.chat_template_kwargs).toEqual({ enable_thinking: true });
    expect(wire.thinking_budget_tokens).toBe(4096);
  });

  it('honors an explicit per-request reasoning effort of off', async () => {
    const { adapter, fakeChat } = harness({});
    await collect(adapter.stream({ ...baseOptions, reasoningEffort: 'off' as never }));
    const wire = fakeChat.mock.calls[0]?.[0] as LlamaCppChatCompletionRequest;
    expect(wire.chat_template_kwargs).toEqual({ enable_thinking: false });
  });

  it('uses reasoning-fields wire mode when configured', async () => {
    const { adapter, fakeChat } = harness({ reasoning: { wire: 'reasoning-fields', preset: 'xhigh' } });
    await collect(adapter.stream(baseOptions));
    const wire = fakeChat.mock.calls[0]?.[0] as LlamaCppChatCompletionRequest;
    expect(wire.reasoning_effort).toBe('xhigh');
    expect(wire.thinking_budget_tokens).toBe(16384);
    expect(wire.chat_template_kwargs).toBeUndefined();
  });

  it('advertises reasoning efforts and default through resolveModel', async () => {
    const { adapter } = harness({ reasoning: { preset: 'low' } });
    const info = await adapter.resolveModel('llamacpp-local', 'qwen3');
    expect(info.reasoning?.efforts.map((e) => e.id)).toEqual(['off', 'low', 'medium', 'xhigh']);
    expect(info.reasoning?.defaultEffort).toBe('low');
  });

  it('advertises only off when reasoning is disabled', async () => {
    const { adapter } = harness({ reasoning: { enabled: false, preset: 'off' } });
    const info = await adapter.resolveModel('llamacpp-local', 'qwen3');
    expect(info.reasoning?.efforts.map((e) => e.id)).toEqual(['off']);
  });
});

describe('LlamacppAdapter stream translation', () => {
  it('translates text deltas into block-start/text-delta/block-end/finish', async () => {
    const { adapter } = harness({}, [
      contentDelta('Hel'),
      contentDelta('lo'),
      contentDelta('', 'stop'),
    ]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'Hel' },
      { type: 'text-delta', index: 0, text: 'lo' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ]);
  });

  it('translates reasoning deltas into a separate reasoning block', async () => {
    const { adapter } = harness({}, [
      chunk({ choices: [{ index: 0, delta: { reasoning_content: 'Hmm, ' }, finish_reason: null }] }),
      contentDelta('Answer!', 'stop'),
    ]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'reasoning' },
      { type: 'reasoning-delta', index: 0, text: 'Hmm, ' },
      { type: 'block-start', index: 1, blockType: 'text' },
      { type: 'text-delta', index: 1, text: 'Answer!' },
      { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'Hmm, ' } },
      { type: 'block-end', index: 1, block: { type: 'text', text: 'Answer!' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ]);
  });

  it('maps finish reasons (length -> max-tokens, tool_calls, unknown -> error)', async () => {
    const { adapter: lengthAdapter } = harness({}, [contentDelta('x', 'length')]);
    const { chunks: lengthChunks } = await collect(lengthAdapter.stream(baseOptions));
    expect(lengthChunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'max-tokens' } });

    const { adapter: toolAdapter } = harness({}, [contentDelta('x', 'tool_calls')]);
    const { chunks: toolChunks } = await collect(toolAdapter.stream(baseOptions));
    expect(toolChunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'tool-calls' } });

    const { adapter: unknownAdapter } = harness({}, [contentDelta('x', 'content_filter')]);
    const { chunks: unknownChunks } = await collect(unknownAdapter.stream(baseOptions));
    expect(unknownChunks.at(-1)).toEqual({
      type: 'finish',
      reason: { kind: 'error', failure: { message: 'model stopped: content_filter', code: 'CONTENT_FILTER' } },
    });
  });

  it('maps usage with disjoint cached/reasoning token counts before finish', async () => {
    const { adapter } = harness({}, [
      contentDelta('hi', 'stop'),
      usageChunk(),
    ]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    expect(chunks).toContainEqual({
      type: 'usage',
      usage: { inputTokens: 8, outputTokens: 5, cacheReadTokens: 4, reasoningTokens: 2 },
    });
    expect(chunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'stop' } });
  });

  it('maps a degenerate empty completion to EMPTY_RESPONSE', async () => {
    const { adapter } = harness({}, [contentDelta('', 'stop')]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    expect(chunks).toEqual([
      {
        type: 'finish',
        reason: { kind: 'error', failure: { message: 'llama.cpp returned a completed response with no content', code: 'EMPTY_RESPONSE' } },
      },
    ]);
  });

  it('drops reasoning blocks when emitThinking is disabled (output-only)', async () => {
    const { adapter } = harness({ reasoning: { expert: { emitThinking: false } } }, [
      chunk({ choices: [{ index: 0, delta: { reasoning_content: 'hidden' }, finish_reason: null }] }),
      contentDelta('visible', 'stop'),
    ]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'visible' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'visible' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ]);
  });

  it('propagates client failures (caller cancellation -> ABORTED)', async () => {
    const { adapter } = harness({}, () => {
      throw new LlmError('llama.cpp request aborted by caller', 'ABORTED');
    });
    const { error } = await collect(adapter.stream(baseOptions));
    expect((error as LlmError).code).toBe('ABORTED');
  });
});

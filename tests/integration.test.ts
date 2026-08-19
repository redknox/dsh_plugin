/**
 * Plugin-level integration tests for issue #3: with the plugin mounted on a
 * real Cordis context, a Harness agent can stream plain-text responses from a
 * mocked llama.cpp server through `ctx.llm` — the full public contract, no
 * agent-loop internals.
 */
import { Context, type Plugin } from '@deepseek-ai/cordis';
import LlmRuntime, {
  CallId,
  createAssistantMessage,
  createToolResultMessage,
  createUserMessage,
  type StreamChunk,
} from '@deepseek-ai/dsh-llm';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Config, PLUGIN_NAME, PROVIDER, apply, type ConfigType } from '../src/index.ts';
import { SSE_DONE } from '../src/protocol.ts';

const mountPlugin: Plugin = { name: PLUGIN_NAME, inject: ['llm'], Config, apply };

function sseResponse(frames: string[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(new TextEncoder().encode(frame));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } });
}

function data(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

const fetchMock = vi.fn();
const contexts: Context[] = [];

afterEach(async () => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
  for (const ctx of contexts.splice(0).reverse()) {
    const task = ctx.fiber.dispose();
    if (task !== undefined && typeof task.then === 'function') await task;
  }
});

async function mounted(config: ConfigType = {}) {
  const ctx = new Context();
  contexts.push(ctx);
  const llmScope = ctx.plugin(LlmRuntime);
  await llmScope.await();
  const scope = ctx.plugin(mountPlugin, config);
  await scope.await();
  return ctx;
}

describe('llm-llamacpp end-to-end through ctx.llm', () => {
  it('streams plain text, usage, and a stop finish from a mocked server', async () => {
    fetchMock.mockResolvedValue(
      sseResponse([
        data({ id: '1', model: 'qwen3', choices: [{ index: 0, delta: { content: 'Hel' }, finish_reason: null }] }),
        data({ id: '1', model: 'qwen3', choices: [{ index: 0, delta: { content: 'lo' }, finish_reason: null }] }),
        data({ id: '1', model: 'qwen3', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
        data({ id: '1', model: 'qwen3', choices: [], usage: { prompt_tokens: 10, completion_tokens: 2 } }),
        `data: ${SSE_DONE}\n\n`,
      ]),
    );
    vi.stubGlobal('fetch', fetchMock);
    const ctx = await mounted({ baseURL: 'http://127.0.0.1:8080' });

    const chunks: StreamChunk[] = [];
    for await (const chunk of ctx.llm.stream({
      provider: PROVIDER,
      model: 'qwen3',
      system: 'You are helpful.',
      messages: [createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })],
    })) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'Hel' },
      { type: 'text-delta', index: 0, text: 'lo' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello' } },
      { type: 'usage', usage: { inputTokens: 10, outputTokens: 2 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ]);

    // The wire request carries attribution and hits /v1/chat/completions.
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8080/v1/chat/completions');
    const headers = init.headers as Record<string, string>;
    expect(headers['user-agent']).toMatch(/^deepseek-harness\//);
    const body = JSON.parse(init.body as string) as { model: string; stream: boolean; messages: unknown[] };
    expect(body.model).toBe('qwen3');
    expect(body.stream).toBe(true);
    expect(body.messages).toHaveLength(2); // system + user
  });

  it('normalizes a server error into a terminal error finish', async () => {
    // A fresh Response per attempt (a consumed body must not leak across
    // reliability retries); maxRetries 0 keeps this test single-attempt.
    fetchMock.mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({ error: { message: 'model not loaded' } }), {
        status: 503,
        headers: { 'content-type': 'application/json' },
      }),
    ));
    vi.stubGlobal('fetch', fetchMock);
    const ctx = await mounted({ retryPolicy: { mode: 'normal', maxRetries: 0 } });

    const chunks: StreamChunk[] = [];
    for await (const chunk of ctx.llm.stream({
      provider: PROVIDER,
      model: 'qwen3',
      messages: [createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })],
    })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(1);
    const finish = chunks[0] as Extract<StreamChunk, { type: 'finish' }>;
    expect(finish.type).toBe('finish');
    expect(finish.reason.kind).toBe('error');
    if (finish.reason.kind === 'error') {
      expect(finish.reason.failure).toMatchObject({ code: 'SERVER', status: 503, message: 'model not loaded' });
    }
  });

  it('round-trips tool schemas and streamed tool calls through ctx.llm (issue #5)', async () => {
    // First turn: fragmented tool-call frames from a mocked llama.cpp server.
    fetchMock.mockResolvedValueOnce(
      sseResponse([
        data({ id: '1', model: 'qwen3', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_7', function: { name: 'get_' } }] }, finish_reason: null }] }),
        data({ id: '1', model: 'qwen3', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { name: 'time', arguments: '{"tz":' } }] }, finish_reason: null }] }),
        data({ id: '1', model: 'qwen3', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"Asia/Shanghai"}' } }] }, finish_reason: null }] }),
        data({ id: '1', model: 'qwen3', choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
        `data: ${SSE_DONE}\n\n`,
      ]),
    );
    // Second turn: the tool result is fed back as a role:tool message.
    fetchMock.mockResolvedValueOnce(
      sseResponse([
        data({ id: '2', model: 'qwen3', choices: [{ index: 0, delta: { content: 'The time is 14:30.' }, finish_reason: null }] }),
        data({ id: '2', model: 'qwen3', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
        `data: ${SSE_DONE}\n\n`,
      ]),
    );
    vi.stubGlobal('fetch', fetchMock);
    const ctx = await mounted({ baseURL: 'http://127.0.0.1:8080' });

    const tools = [{
      name: 'get_time',
      description: 'Get the current local time',
      parameters: { type: 'object', properties: { tz: { type: 'string' } }, required: [] },
    }];

    // Turn 1: offer the tool; the provider must surface the assembled call.
    const firstChunks: StreamChunk[] = [];
    for await (const chunk of ctx.llm.stream({
      provider: PROVIDER,
      model: 'qwen3',
      messages: [createUserMessage({ content: [{ type: 'text', text: 'What time is it in Asia/Shanghai?' }], source: { kind: 'user' } })],
      tools,
    })) {
      firstChunks.push(chunk);
    }
    const toolEnds = firstChunks
      .filter((c): c is Extract<StreamChunk, { type: 'block-end' }> => c.type === 'block-end')
      .map((c) => c.block);
    expect(toolEnds).toEqual([
      { type: 'tool-call', id: CallId('call_7'), name: 'get_time', arguments: '{"tz":"Asia/Shanghai"}' },
    ]);
    expect(firstChunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'tool-calls' } });

    // The first wire request carried the tool schema.
    const firstWire = JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string) as {
      tools?: unknown[];
    };
    expect(firstWire.tools).toEqual([{
      type: 'function',
      function: {
        name: 'get_time',
        description: 'Get the current local time',
        parameters: { type: 'object', properties: { tz: { type: 'string' } }, required: [] },
      },
    }]);

    // Turn 2: assistant tool call + tool result back to the model.
    const history = [
      createUserMessage({ content: [{ type: 'text', text: 'What time is it in Asia/Shanghai?' }], source: { kind: 'user' } }),
      createAssistantMessage({
        content: [{ type: 'tool-call', id: CallId('call_7'), name: 'get_time', arguments: '{"tz":"Asia/Shanghai"}' }],
        source: { provider: PROVIDER, model: 'qwen3' },
      }),
      createToolResultMessage({
        callId: CallId('call_7'),
        content: [{ type: 'text', text: '14:30' }],
        isError: false,
      }),
    ];
    const secondChunks: StreamChunk[] = [];
    for await (const chunk of ctx.llm.stream({ provider: PROVIDER, model: 'qwen3', messages: history, tools })) {
      secondChunks.push(chunk);
    }
    expect(secondChunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'stop' } });

    // The second wire request carried the tool result as a role:tool message.
    const secondWire = JSON.parse((fetchMock.mock.calls[1]?.[1] as RequestInit).body as string) as {
      messages: Array<{ role: string; tool_call_id?: string; content?: string }>;
    };
    const toolMessage = secondWire.messages.find((m) => m.role === 'tool');
    expect(toolMessage).toEqual({ role: 'tool', tool_call_id: 'call_7', content: '14:30' });
  });
});

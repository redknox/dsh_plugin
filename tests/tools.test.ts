/**
 * Tool-call tests for issue #5: Harness tool schemas → OpenAI-compatible
 * `tools`, streamed `tool_calls` deltas → Harness tool-call blocks with
 * correct fragmentation handling, multiple calls, and clear errors for
 * malformed argument JSON.
 */
import { describe, expect, it } from 'vitest';
import { CallId, type StreamChunk, type ToolCallBlock } from '@deepseek-ai/dsh-llm';
import {
  baseOptions,
  chunk,
  collect,
  contentDelta,
  harness,
  lastRequest,
  msg,
} from './helpers.ts';
import type { LlamaCppToolCallDelta } from '../src/protocol.ts';

type ToolCallDeltaChunk = Extract<StreamChunk, { type: 'tool-call-delta' }>;

/** A streamed tool-call delta frame for one wire call index. */
function toolDelta(partial: LlamaCppToolCallDelta & { index: number }): ReturnType<typeof chunk> {
  return chunk({ choices: [{ index: 0, delta: { tool_calls: [partial] }, finish_reason: null }] });
}

describe('tool schema serialization', () => {
  it('converts Harness tool schemas into OpenAI-compatible tools', async () => {
    const { adapter, fakeChat } = harness({});
    await collect(adapter.stream({
      ...baseOptions,
      tools: [
        {
          name: 'get_time',
          description: 'Get the current local time',
          parameters: { type: 'object', properties: { tz: { type: 'string' } }, required: [] },
        },
        {
          name: 'echo',
          description: 'Echo text back',
          parameters: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] },
        },
      ],
    }));
    const wire = lastRequest(fakeChat);
    expect(wire.tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'get_time',
          description: 'Get the current local time',
          parameters: { type: 'object', properties: { tz: { type: 'string' } }, required: [] },
        },
      },
      {
        type: 'function',
        function: {
          name: 'echo',
          description: 'Echo text back',
          parameters: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] },
        },
      },
    ]);
  });

  it('omits the tools field when no tools are offered', async () => {
    const { adapter, fakeChat } = harness({});
    await collect(adapter.stream(baseOptions));
    expect(lastRequest(fakeChat).tools).toBeUndefined();
  });
});

describe('streamed tool-call translation', () => {
  it('accumulates fragmented ids, names, and argument fragments', async () => {
    const { adapter } = harness({}, [
      toolDelta({ index: 0, id: 'call_7', function: { name: 'get_' } }),
      toolDelta({ index: 0, function: { name: 'time', arguments: '{"tz":' } }),
      toolDelta({ index: 0, function: { arguments: '"UTC"}' } }),
      chunk({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
    ]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();

    const deltas = chunks.filter((c): c is ToolCallDeltaChunk => c.type === 'tool-call-delta');
    // The first frame carries the id/name with no argument fragment yet; its
    // delta (empty argumentsDelta) must still be emitted so id/name reach the
    // consumer, exactly as the provider sent them.
    expect(deltas.map((d) => d.argumentsDelta)).toEqual(['', '{"tz":', '"UTC"}']);
    expect(deltas[0]?.id).toBe(CallId('call_7'));
    expect(deltas[0]?.name).toBe('get_');

    const ends = chunks.filter((c) => c.type === 'block-end');
    const toolEnd = ends[0]?.block as ToolCallBlock;
    expect(toolEnd).toEqual({
      type: 'tool-call',
      id: CallId('call_7'),
      name: 'get_time',
      arguments: '{"tz":"UTC"}',
    });
    expect(chunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'tool-calls' } });
  });

  it('supports multiple tool calls in one response, preserving order and ids', async () => {
    const { adapter } = harness({}, [
      toolDelta({ index: 0, id: 'call_a', function: { name: 'get_time', arguments: '{}' } }),
      toolDelta({ index: 1, id: 'call_b', function: { name: 'echo', arguments: '{"text":"hi"}' } }),
      chunk({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
    ]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();

    const starts = chunks.filter((c) => c.type === 'block-start');
    expect(starts.map((s) => s.blockType)).toEqual(['tool-call', 'tool-call']);

    const toolEnds = chunks
      .filter((c): c is Extract<typeof c, { type: 'block-end' }> => c.type === 'block-end')
      .map((c) => c.block as ToolCallBlock);
    expect(toolEnds).toEqual([
      { type: 'tool-call', id: CallId('call_a'), name: 'get_time', arguments: '{}' },
      { type: 'tool-call', id: CallId('call_b'), name: 'echo', arguments: '{"text":"hi"}' },
    ]);
  });

  it('handles mixed text and tool-call output in one response', async () => {
    const { adapter } = harness({}, [
      contentDelta('I will check.'),
      toolDelta({ index: 0, id: 'call_1', function: { name: 'get_time', arguments: '{}' } }),
      contentDelta(' Done.'),
      chunk({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
    ]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();

    const blockTypes = chunks
      .filter((c) => c.type === 'block-start')
      .map((c) => c.blockType);
    expect(blockTypes).toEqual(['text', 'tool-call']);
    const texts = chunks
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.text);
    expect(texts).toEqual(['I will check.', ' Done.']);
  });

  it('emits a placeholder id until the provider sends one', async () => {
    const { adapter } = harness({}, [
      toolDelta({ index: 0, function: { name: 'get_time', arguments: '{}' } }),
      toolDelta({ index: 0, id: 'call_late' }),
      chunk({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
    ]);
    const { chunks, error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeUndefined();
    const deltas = chunks.filter((c) => c.type === 'tool-call-delta');
    expect(deltas[0]?.id).toBe(CallId(''));
    expect(deltas[1]?.id).toBe(CallId('call_late'));
  });

  it('fails with a clear error on malformed or incomplete argument JSON', async () => {
    const { adapter } = harness({}, [
      toolDelta({ index: 0, id: 'call_1', function: { name: 'get_time', arguments: '{"tz":' } }),
      chunk({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
    ]);
    const { error } = await collect(adapter.stream(baseOptions));
    expect(error).toBeInstanceOf(Error);
    expect((error as Error & { code: string }).code).toBe('INVALID_TOOL_ARGUMENTS');
  });

  it('rejects empty arguments as malformed', async () => {
    const { adapter } = harness({}, [
      toolDelta({ index: 0, id: 'call_1', function: { name: 'get_time' } }),
      chunk({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
    ]);
    const { error } = await collect(adapter.stream(baseOptions));
    expect((error as Error & { code: string }).code).toBe('INVALID_TOOL_ARGUMENTS');
  });
});

describe('tool round-trip serialization', () => {
  it('sends the tool result back as a role:tool message', async () => {
    const { adapter, fakeChat } = harness({}, [
      toolDelta({ index: 0, id: 'call_1', function: { name: 'get_time', arguments: '{}' } }),
      chunk({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
    ]);
    await collect(adapter.stream(baseOptions));

    // Second turn: assistant tool call + user tool result.
    await collect(adapter.stream({
      ...baseOptions,
      messages: [
        {
          id: 'asst' as never,
          role: 'assistant',
          content: [{ type: 'tool-call', id: CallId('call_1'), name: 'get_time', arguments: '{}' }],
          source: { kind: 'model', provider: 'llamacpp-local', model: 'qwen3' },
        },
        {
          id: 'toolres' as never,
          role: 'user',
          content: [{ type: 'tool-result', toolCallId: CallId('call_1'), content: [{ type: 'text', text: '14:30 UTC' }] }],
          source: { kind: 'tool', callId: CallId('call_1') },
        },
        msg('user', 'Thanks'),
      ],
    }));

    const wire = lastRequest(fakeChat);
    expect(wire.messages).toEqual([
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_time', arguments: '{}' } }],
      },
      { role: 'tool', tool_call_id: 'call_1', content: '14:30 UTC' },
      { role: 'user', content: 'Thanks' },
    ]);
  });
});

/**
 * llama.cpp OpenAI-compatible streaming client.
 *
 * Transport only: HTTP request construction, SSE streaming, cancellation, and
 * protocol parsing against `/v1/chat/completions`. Harness message/tool
 * conversion belongs to the adapter (`adapter.ts`), which depends on this
 * client — never the other way around. The client imports no DeepSeek Harness
 * core code, so it can be unit-tested with mocked HTTP/SSE responses alone.
 *
 * Implemented in issue #2. Until then this class is a structural placeholder:
 * every method fails with a clear `NOT_IMPLEMENTED` error.
 *
 * @module llm-llamacpp/client
 */
import { LlmError } from '@deepseek-ai/dsh-llm';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from './protocol.ts';

/** Per-request client options. */
export interface LlamaCppClientOptions {
  /** Timeout for one outstanding provider read, in milliseconds. */
  readonly streamIdleTimeoutMs: number;
  /** Optional `{ header: value }` to attach when an API key is configured. */
  readonly authHeader?: string;
}

/**
 * Minimal health probe against the llama.cpp server.
 * @returns true when the endpoint answers with a 2xx status.
 */
export async function checkHealth(baseURL: string, signal?: AbortSignal): Promise<boolean> {
  throw new LlmError('llm-llamacpp: client health check not implemented yet (issue #2)', 'NOT_IMPLEMENTED');
}

/**
 * Streaming client for llama.cpp's OpenAI-compatible chat completions.
 * One instance is bound to one endpoint; issue #7 adds endpoint selection.
 */
export class LlamaCppClient {
  constructor(
    readonly baseURL: string,
    readonly options: LlamaCppClientOptions = { streamIdleTimeoutMs: 300_000 },
  ) {}

  /**
   * POST one chat completion and yield its SSE data payloads.
   * @param request - the wire request; `stream` is forced true.
   * @param signal - cancellation for the whole request, including body reads.
   */
  async *chat(
    request: LlamaCppChatCompletionRequest,
    signal?: AbortSignal,
  ): AsyncIterable<LlamaCppChatCompletionChunk> {
    throw new LlmError('llm-llamacpp: client streaming not implemented yet (issue #2)', 'NOT_IMPLEMENTED');
  }
}

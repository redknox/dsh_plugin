/**
 * Typed wire structures for llama.cpp's OpenAI-compatible
 * `/v1/chat/completions` endpoint. These are the *transport* shapes only:
 * Harness message/tool conversion belongs to the adapter, and reasoning
 * translation to `reasoning.ts`; this module stays free of Harness imports so
 * the client can be tested without any DeepSeek Harness core code.
 *
 * The reasoning-related fields (`chat_template_kwargs`, `reasoning_effort`,
 * `reasoning_budget_tokens`) depend on the installed llama.cpp/Qwen version;
 * see the adapter/reasoning documentation for which versions accept which.
 *
 * @module llm-llamacpp/protocol
 */

/** One chat message exactly as llama.cpp's OpenAI-compatible server reads it. */
export interface LlamaCppChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  /** `null` is the documented OpenAI shape for assistant messages carrying tool calls. */
  content: string | null;
  /** Assistant message tool calls, when the model emitted them. */
  tool_calls?: LlamaCppToolCall[];
  /** Tool-result correlation, present only on `role: 'tool'` messages. */
  tool_call_id?: string;
}

/** A fully assembled tool call inside an assistant message. */
export interface LlamaCppToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    /** Raw JSON argument string exactly as produced by the model. */
    arguments: string;
  };
}

/** A tool schema offered to the model. */
export interface LlamaCppTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    /** JSON Schema object for the arguments. */
    parameters: Record<string, unknown>;
  };
}

/** Per-choice `tool_calls` delta inside a streaming chunk. */
export interface LlamaCppToolCallDelta {
  /** Accumulator index for fragmented tool-call frames; optional in some servers. */
  index?: number;
  id?: string;
  type?: 'function';
  function?: {
    name?: string;
    /** A fragment of the JSON argument string. */
    arguments?: string;
  };
}

/** Request body for `POST /v1/chat/completions`. */
export interface LlamaCppChatCompletionRequest {
  model: string;
  messages: LlamaCppChatMessage[];
  /** Always true for this plugin: responses are consumed as an SSE stream. */
  stream: true;
  stream_options?: { include_usage?: boolean };
  temperature?: number;
  max_tokens?: number;
  stop?: string[];
  tools?: LlamaCppTool[];
  tool_choice?: 'auto' | 'none' | 'required';
  /**
   * Qwen3 template kwargs (`enable_thinking`, `preserve_thinking`) supported
   * by llama.cpp builds that ship the per-request template-kwargs hook. The
   * runtime thinking budget is NOT a template kwarg; it travels as the
   * top-level `thinking_budget_tokens` inference control below.
   */
  chat_template_kwargs?: Record<string, unknown>;
  /** Newer llama.cpp per-request reasoning effort, when the server supports it. */
  reasoning_effort?: string;
  /**
   * Per-request runtime thinking budget in tokens (llama.cpp OpenAI-compatible
   * field; the server also accepts `reasoning_budget_tokens` as its newer
   * alias, but `thinking_budget_tokens` is the broader-compatibility spelling).
   */
  thinking_budget_tokens?: number;
}

/** Token accounting reported by the server (non-streamed usage or trailing usage chunk). */
export interface LlamaCppUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

/** One choice of one streaming chunk. */
export interface LlamaCppChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: string;
    content?: string | null;
    /** Qwen-style reasoning/thinking deltas, when the server streams them. */
    reasoning_content?: string;
    tool_calls?: LlamaCppToolCallDelta[];
  };
  finish_reason: string | null;
}

/** One SSE data payload of a streamed chat completion. */
export interface LlamaCppChatCompletionChunk {
  id: string;
  model: string;
  choices: LlamaCppChatCompletionChunkChoice[];
  usage?: LlamaCppUsage;
}

/** The OpenAI `[DONE]` sentinel terminating a streamed completion. */
export const SSE_DONE = '[DONE]';

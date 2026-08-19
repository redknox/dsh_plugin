/**
 * The `LlmAdapter` implementation bridging DeepSeek Harness and the llama.cpp
 * transport client.
 *
 * The adapter owns Harness-specific orchestration only: resolve connection
 * facts and credentials, serialize `GenerateOptions` into llama.cpp request
 * bodies (`serialize.ts`), and translate streamed responses back into Harness
 * `StreamChunk` values (`translate.ts`). Transport stays in `client.ts`;
 * reasoning policy stays in `reasoning.ts`. No package-internal agent-loop code
 * is imported; only the public `ctx.llm` contract is used.
 *
 * @module llm-llamacpp/adapter
 */
import {
  LlmAdapter,
  type GenerateOptions,
  type LlmModelInfo,
  type LlmProviderInfo,
  type LlmResolvedModelInfo,
  type Message,
  type StreamChunk,
} from '@deepseek-ai/dsh-llm';
import {
  LlamaCppClient,
  type LlamaCppClientOptions,
} from './client.ts';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from './protocol.ts';
import type { ResolvedAdapterOptions } from './config.ts';
import {
  buildReasoningPolicy,
  reasoningEfforts,
  type ReasoningPolicyContext,
} from './reasoning.ts';
import { serializeRequest } from './serialize.ts';
import { translate } from './translate.ts';

/** The streaming surface an adapter needs from a transport client. */
export interface LlamaCppChatHandle {
  chat(
    request: LlamaCppChatCompletionRequest,
    options?: { signal?: AbortSignal },
  ): AsyncIterable<LlamaCppChatCompletionChunk>;
}

/** Minimal logger surface for adaptive-policy debug metadata. */
export interface LlamacppLogger {
  debug(message: string): void;
}

/** The adapter's dependency surface, injected by the registering plugin. */
export interface LlamacppAdapterDeps {
  /** Per-operation re-read of validated connection facts. */
  readonly options: () => ResolvedAdapterOptions;
  /** Per-request API key resolution; `undefined` means no auth header. */
  readonly resolveApiKey: () => Promise<string | undefined>;
  /** Optional client factory for tests; defaults to a real `LlamaCppClient`. */
  readonly createClient?: (baseURL: string, options: LlamaCppClientOptions) => LlamaCppChatHandle;
  /** Optional debug logger (e.g. the plugin's `ctx.logger`) for policy decisions. */
  readonly logger?: LlamacppLogger;
}

/** Rough prompt-size estimate in tokens (~4 chars/token) for policy context. */
function estimatePromptTokens(system: string | undefined, messages: readonly Message[]): number {
  let chars = system?.length ?? 0;
  for (const message of messages) {
    for (const block of message.content) {
      if (block.type === 'text') chars += block.text.length;
    }
  }
  return Math.floor(chars / 4);
}

/** Extract the adaptive policy context from one request (issue #6). */
function policyContext(
  options: GenerateOptions,
  hints: readonly string[] | undefined,
): ReasoningPolicyContext {
  const last = options.messages.at(-1);
  return {
    messages: options.messages.length,
    estimatedPromptTokens: estimatePromptTokens(options.system, options.messages),
    toolsAvailable: (options.tools?.length ?? 0) > 0,
    followsToolResult: last !== undefined && last.content.some((block) => block.type === 'tool-result'),
    ...(hints !== undefined && hints.length > 0 ? { hints } : {}),
  };
}

function defaultCreateClient(baseURL: string, options: LlamaCppClientOptions): LlamaCppChatHandle {
  return new LlamaCppClient(baseURL, options);
}

/**
 * Adapter for the `llamacpp-local` provider route. One instance serves every
 * model name: the harness model id IS the wire model id.
 */
export class LlamacppAdapter extends LlmAdapter {
  readonly deps: LlamacppAdapterDeps;

  constructor(deps: LlamacppAdapterDeps) {
    super();
    this.deps = deps;
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: this.deps.options().providerName };
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const { model } = this.deps.options();
    return Promise.resolve([
      {
        provider,
        id: model,
        name: model,
        inputModalities: ['text'],
      },
    ]);
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const { reasoning } = this.deps.options();
    const { efforts, defaultEffort } = reasoningEfforts(reasoning);
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      inputModalities: ['text'],
      reasoning: { efforts, defaultEffort },
    });
  }

  /**
   * Stream one model call. Each call resolves connection facts, the API key,
   * and the reasoning policy afresh (a changed base URL, key, or preset
   * reaches the very next request), builds a fresh client for the resolved
   * endpoint, and translates the wire stream. Thrown failures (client
   * `LlmError`s included) are normalized by `LlmRuntime` into a terminal
   * `finish` chunk.
   */
  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const opts = this.deps.options();
    const context = policyContext(options, opts.reasoning.adaptive?.hints);
    // The adapter depends only on the inference-policy seam; it has no
    // knowledge of whether the resolved policy is static or adaptive.
    const reasoning = buildReasoningPolicy(opts.reasoning).resolve({
      effort: options.reasoningEffort,
      purpose: options.purpose,
      context,
    });
    this.deps.logger?.debug(
      `llm-llamacpp reasoning decision: ${reasoning.reason ?? 'static preset'} ` +
        `(enabled=${reasoning.enabled}, effort=${reasoning.effort ?? '-'}, budget=${reasoning.budgetTokens ?? '-'})`,
    );
    const request = serializeRequest(options, reasoning);
    const apiKey = await this.deps.resolveApiKey();
    const auth = apiKey !== undefined
      ? {
          name: opts.apiKeyHeader,
          value: opts.apiKeyHeader === 'authorization' ? `Bearer ${apiKey}` : apiKey,
        }
      : undefined;
    const createClient = this.deps.createClient ?? defaultCreateClient;
    const client = createClient(opts.baseURL, {
      streamIdleTimeoutMs: opts.streamIdleTimeoutMs,
      ...(auth !== undefined ? { auth } : {}),
    });
    yield* translate(client.chat(request, { signal: options.signal }), {
      emitReasoning: reasoning.emitThinking,
    });
  }
}

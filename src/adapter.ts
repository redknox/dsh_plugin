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
  isTokenDelta,
  type FinishReason,
  type GenerateOptions,
  type LlmModelInfo,
  type LlmProviderInfo,
  type LlmResolvedModelInfo,
  type Message,
  type StreamChunk,
  type TokenUsage,
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
import {
  EndpointPool,
  streamReliably,
  type ReliabilityEndpoint,
} from './reliability.ts';
import {
  deriveWorkload,
  routeEndpoints,
  type RoutingRequest,
} from './routing.ts';
import { serializeRequest } from './serialize.ts';
import {
  NoopTelemetry,
  newRequestId,
  type TelemetrySink,
} from './telemetry.ts';
import { translate } from './translate.ts';

/** The streaming surface an adapter needs from a transport client. */
export interface LlamaCppChatHandle {
  chat(
    request: LlamaCppChatCompletionRequest,
    options?: { signal?: AbortSignal },
  ): AsyncIterable<LlamaCppChatCompletionChunk>;
}

/** Minimal logger surface for reliability/adaptive debug metadata. */
export interface LlamacppLogger {
  debug(message: string): void;
  warn?(message: string): void;
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
  /** Optional structured telemetry sink factory (issue #8); defaults to no-op. */
  readonly telemetry?: () => TelemetrySink;
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
  /** Persistent endpoint health state across requests (reliability layer). */
  readonly pool: EndpointPool;

  constructor(deps: LlamacppAdapterDeps) {
    super();
    this.deps = deps;
    this.pool = new EndpointPool();
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: this.deps.options().providerName };
  }

  override providerRetryPolicy(provider: string): ReturnType<LlmAdapter['providerRetryPolicy']> {
    void provider;
    return this.deps.options().retryPolicy;
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
   * reaches the very next request) and drives the request through the
   * reliability layer (ordered fallback + bounded retry/backoff; issue #7)
   * before translating the wire stream. Thrown failures (client `LlmError`s
   * included) are normalized by `LlmRuntime` into a terminal `finish` chunk.
   */
  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    // Structured observability (issue #8): the request lifecycle starts at
    // adapter entry, BEFORE any work, so pre-transport failures (reasoning
    // resolution, serialization, credential resolution) converge into the
    // same trace with true end-to-end latency and a stable requestId. The
    // sink is per-operation so telemetry can be disabled without touching the
    // adapter.
    const sink = (this.deps.telemetry ?? (() => NoopTelemetry))();
    const requestId = newRequestId();
    const startedAt = Date.now();
    sink.emit({
      type: 'started',
      requestId,
      at: startedAt,
      context: {
        model: options.model,
        ...(options.purpose !== undefined ? { purpose: options.purpose } : {}),
        toolsAvailable: (options.tools?.length ?? 0) > 0,
      },
    });

    let retryCount = 0;
    let fallbackCount = 0;
    let lastEndpoint: string | undefined;
    let ttftMs: number | undefined;
    let chunkCount = 0;
    let toolCallCount = 0;
    let finish: FinishReason | undefined;
    let usage: TokenUsage | undefined;

    const finishWith = (failureCode?: string): void => {
      const totalMs = Date.now() - startedAt;
      sink.emit({
        type: 'finished',
        requestId,
        at: startedAt + totalMs,
        outcome: {
          endpoint: lastEndpoint ?? 'unresolved',
          retryCount,
          fallbackCount,
          ...(ttftMs !== undefined ? { ttftMs } : {}),
          totalMs,
          ...(ttftMs !== undefined ? { completionMs: totalMs - ttftMs } : {}),
          streamChunkCount: chunkCount,
          ...(finish !== undefined ? { finishReason: finish } : {}),
          ...(usage !== undefined ? { usage } : {}),
          ...(toolCallCount > 0 ? { toolCallCount } : {}),
          ...(failureCode !== undefined ? { failureCode } : {}),
        },
      });
    };

    try {
      const opts = this.deps.options();
      const context = policyContext(options, opts.reasoning.adaptive?.hints);
      // The adapter depends only on the inference-policy seam; it has no
      // knowledge of whether the resolved policy is static or adaptive.
      const reasoning = buildReasoningPolicy(opts.reasoning).resolve({
        effort: options.reasoningEffort,
        purpose: options.purpose,
        context,
      });
      sink.emit({
        type: 'reasoning',
        requestId,
        at: Date.now(),
        decision: {
          enabled: reasoning.enabled,
          ...(reasoning.effort !== undefined ? { effort: reasoning.effort } : {}),
          ...(reasoning.budgetTokens !== undefined ? { budgetTokens: reasoning.budgetTokens } : {}),
          ...(reasoning.reason !== undefined ? { reason: reasoning.reason } : {}),
        },
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

      // Capability-aware routing (issue #9) before reliability fallback: pick
      // eligible endpoints from request/model capabilities; a plain endpoint
      // list without metadata routes exactly as #7 (all eligible, config
      // order). Reliability still owns transient failures afterwards.
      const routingRequest: RoutingRequest = {
        model: options.model,
        toolsAvailable: (options.tools?.length ?? 0) > 0,
        reasoningEnabled: reasoning.enabled,
        workload: deriveWorkload(options.purpose, reasoning.enabled),
        ...(context.estimatedPromptTokens !== undefined ? { estimatedPromptTokens: context.estimatedPromptTokens } : {}),
      };
      const routing = routeEndpoints(routingRequest, opts.endpointProfiles);
      sink.emit({
        type: 'routing',
        requestId,
        at: Date.now(),
        decision: { candidates: routing.candidates, rationale: routing.rationale },
      });
      const endpoints: readonly ReliabilityEndpoint[] = routing.candidates.map((baseURL) => ({
        baseURL,
        ...(auth !== undefined ? { auth } : {}),
      }));
      const createClient = this.deps.createClient ?? defaultCreateClient;

      for await (const chunk of translate(streamReliably(request, {
        endpoints,
        retryPolicy: opts.retryPolicy,
        streamIdleTimeoutMs: opts.streamIdleTimeoutMs,
        ...(opts.requestTimeoutMs !== undefined ? { requestTimeoutMs: opts.requestTimeoutMs } : {}),
        createClient,
        pool: this.pool,
        logger: this.deps.logger,
        signal: options.signal,
        onAttempt: (report) => {
          lastEndpoint = report.baseURL;
          if (report.outcome === 'retry') retryCount += 1;
          else if (report.outcome === 'fallback') fallbackCount += 1;
          sink.emit({ type: 'attempt', requestId, at: Date.now(), attempt: report });
        },
      }), {
        emitReasoning: reasoning.emitThinking,
      })) {
        if (chunk.type === 'block-start' && chunk.blockType === 'tool-call') toolCallCount += 1;
        if (chunk.type === 'usage') usage = chunk.usage;
        if (chunk.type === 'finish') finish = chunk.reason;
        if (isTokenDelta(chunk) && ttftMs === undefined) ttftMs = Date.now() - startedAt;
        chunkCount += 1;
        yield chunk;
      }
      finishWith();
    } catch (error) {
      finishWith(errorCodeOf(error));
      throw error;
    }
  }
}

/** Stable machine code of a thrown failure, when it has one. */
function errorCodeOf(error: unknown): string | undefined {
  if (error === null || typeof error !== 'object') return undefined;
  const code = (error as { code?: unknown }).code;
  return typeof code === 'string' ? code : undefined;
}

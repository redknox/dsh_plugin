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
  type LlamaCppAuth,
  type LlamaCppClientOptions,
} from './client.ts';
import type { LlamaCppChatCompletionChunk, LlamaCppChatCompletionRequest } from './protocol.ts';
import type { ResolvedAdapterOptions } from './config.ts';
import {
  EndpointDiscovery,
  mergeCapabilities,
  type DiscoveredModel,
  type DiscoveryOptions,
} from './discovery.ts';
import {
  FeedbackHistory,
  buildFeedbackPolicy,
  type ReasoningFeedback,
} from './feedback.ts';
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
  CapabilityRoutingPolicy,
  deriveWorkload,
  type EndpointCapabilities,
  type EndpointRoutingProfile,
  type RoutingPolicy,
  type RoutingRequest,
} from './routing.ts';
import { serializeRequest } from './serialize.ts';
import type { DiagnosticsModel } from './diagnostics.ts';
import {
  NoopTelemetry,
  newRequestId,
  type RequestOutcome,
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
  /** Optional routing policy (issue #9); defaults to capability routing. */
  readonly routing?: RoutingPolicy;
  /** Optional discovery factory (issue #10); defaults to the real probe client. */
  readonly discovery?: (baseURL: string, options: DiscoveryOptions) => EndpointDiscovery;
  /** Optional feedback history (issue #11); defaults to a fresh bounded store. */
  readonly history?: FeedbackHistory;
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

/** Default routing policy: capability filtering + deterministic ordering. */
const defaultRoutingPolicy: RoutingPolicy = new CapabilityRoutingPolicy();

/**
 * Configured endpoint capabilities that apply to one model: the first profile
 * in configuration order that explicitly serves the model, else the first
 * profile with no model restriction. `undefined` means no configured facts.
 */
function configuredCapabilitiesFor(
  model: string,
  profiles: readonly EndpointRoutingProfile[],
): EndpointCapabilities | undefined {
  const unrestricted = profiles.find((profile) => profile.capabilities?.models === undefined || profile.capabilities.models.length === 0);
  const serving = profiles.find((profile) => profile.capabilities?.models?.includes(model));
  const chosen = serving ?? (unrestricted?.capabilities !== undefined ? unrestricted : undefined);
  return chosen?.capabilities;
}

/**
 * Adapter for the `llamacpp-local` provider route. One instance serves every
 * model name: the harness model id IS the wire model id.
 */
export class LlamacppAdapter extends LlmAdapter {
  readonly deps: LlamacppAdapterDeps;
  /** Persistent endpoint health state across requests (reliability layer). */
  readonly pool: EndpointPool;
  /** Bounded reasoning-outcome history (issue #11 feedback loop). */
  readonly history: FeedbackHistory;
  /** Per-endpoint discovery managers (issue #10), cached for TTL persistence. */
  private readonly discoveryByUrl = new Map<string, EndpointDiscovery>();

  constructor(deps: LlamacppAdapterDeps) {
    super();
    this.deps = deps;
    this.pool = new EndpointPool();
    this.history = deps.history ?? new FeedbackHistory();
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: this.deps.options().providerName };
  }

  override providerRetryPolicy(provider: string): ReturnType<LlmAdapter['providerRetryPolicy']> {
    void provider;
    return this.deps.options().retryPolicy;
  }

  private discoveryFor(baseURL: string, auth?: LlamaCppAuth): EndpointDiscovery {
    let discovery = this.discoveryByUrl.get(baseURL);
    if (discovery === undefined) {
      const opts = this.deps.options();
      const create = this.deps.discovery ?? ((url: string, options: DiscoveryOptions) => new EndpointDiscovery(url, options));
      discovery = create(baseURL, {
        ...(opts.discovery.ttlMs !== undefined ? { ttlMs: opts.discovery.ttlMs } : {}),
        ...(opts.discovery.timeoutMs !== undefined ? { timeoutMs: opts.discovery.timeoutMs } : {}),
        ...(auth !== undefined ? { auth } : {}),
      });
      this.discoveryByUrl.set(baseURL, discovery);
    }
    return discovery;
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const opts = this.deps.options();
    if (opts.discovery.enabled) {
      const discovered = await this.discoveredModels();
      if (discovered.length > 0) {
        return discovered.map((entry) => ({
          provider,
          id: entry.id,
          name: entry.id,
          inputModalities: ['text'],
        }));
      }
    }
    const { model } = opts;
    return [
      {
        provider,
        id: model,
        name: model,
        inputModalities: ['text'],
      },
    ];
  }

  override async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const opts = this.deps.options();
    const { efforts, defaultEffort } = reasoningEfforts(opts.reasoning);
    // User-configured endpoint capabilities win over discovered facts
    // (issue #10 precedence); discovery fills gaps when enabled.
    const configured = configuredCapabilitiesFor(model, opts.endpointProfiles);
    let discovered: DiscoveredModel | undefined;
    if (opts.discovery.enabled) {
      discovered = await this.findDiscovered(model, opts);
    }
    const merged = mergeCapabilities(configured, discovered);
    return {
      provider,
      id: model,
      name: model,
      inputModalities: ['text'],
      ...(merged.contextWindow !== undefined ? { context: { contextWindow: merged.contextWindow } } : {}),
      reasoning: { efforts, defaultEffort },
    };
  }

  /**
   * Union of cached discovered model ids across endpoints (issue #12
   * diagnostics). Empty when discovery is disabled or no fresh cache exists.
   */
  cachedDiscoveredModels(): readonly string[] {
    const opts = this.deps.options();
    if (!opts.discovery.enabled) return [];
    const ids = new Set<string>();
    for (const profile of opts.endpointProfiles) {
      const result = this.discoveryFor(profile.baseURL).discoverCached();
      for (const entry of result?.models ?? []) ids.add(entry.id);
    }
    return [...ids];
  }

  /**
   * Structured model/capability facts for diagnostics (issue #12): the
   * configured model and every model named by configured endpoint
   * capabilities (authoritative), merged with cached discovered facts when
   * discovery is enabled (configured wins per field; `source` records which
   * layer provided the row).
   */
  diagnosticModels(): DiagnosticsModel[] {
    const opts = this.deps.options();
    const facts = new Map<string, { contextWindow?: number; tools?: boolean; reasoning?: boolean; configured: boolean }>();
    const configuredModel = opts.model;
    const configured = configuredCapabilitiesFor(configuredModel, opts.endpointProfiles);
    facts.set(configuredModel, { ...(configured !== undefined ? { contextWindow: configured.contextWindow, tools: configured.tools, reasoning: configured.reasoning } : {}), configured: true });
    for (const profile of opts.endpointProfiles) {
      const capabilities = profile.capabilities;
      if (capabilities?.models === undefined) continue;
      for (const id of capabilities.models) {
        const existing = facts.get(id);
        facts.set(id, {
          contextWindow: capabilities.contextWindow ?? existing?.contextWindow,
          tools: capabilities.tools ?? existing?.tools,
          reasoning: capabilities.reasoning ?? existing?.reasoning,
          configured: true,
        });
      }
    }
    if (opts.discovery.enabled) {
      for (const profile of opts.endpointProfiles) {
        const result = this.discoveryFor(profile.baseURL).discoverCached();
        for (const entry of result?.models ?? []) {
          // Per-field precedence (same as #10 routing/resolveModel):
          // configured explicit values win, discovered values fill gaps —
          // even when a configured row already exists for this model.
          const existing = facts.get(entry.id);
          facts.set(entry.id, {
            contextWindow: existing?.contextWindow ?? entry.contextWindow,
            tools: existing?.tools ?? entry.supportsTools,
            reasoning: existing?.reasoning ?? entry.supportsReasoning,
            configured: existing?.configured ?? false,
          });
        }
      }
    }
    return [...facts.entries()].map(([id, fact]) => ({
      id,
      ...(fact.contextWindow !== undefined ? { contextWindow: fact.contextWindow } : {}),
      ...(fact.tools !== undefined ? { supportsTools: fact.tools } : {}),
      ...(fact.reasoning !== undefined ? { supportsReasoning: fact.reasoning } : {}),
      source: fact.configured ? 'configured' : 'discovered',
    }));
  }

  /** Union of discovered model ids across configured endpoints (deduped). */
  private async discoveredModels(): Promise<readonly DiscoveredModel[]> {
    const opts = this.deps.options();
    const byId = new Map<string, DiscoveredModel>();
    for (const profile of opts.endpointProfiles) {
      const discovery = this.discoveryFor(profile.baseURL);
      const result = await discovery.discover(); // graceful: never throws
      for (const entry of result.models) {
        if (byId.has(entry.id)) continue;
        byId.set(entry.id, entry);
      }
    }
    return [...byId.values()];
  }

  /** First discovered facts matching the model across endpoints. */
  private async findDiscovered(model: string, opts: ResolvedAdapterOptions): Promise<DiscoveredModel | undefined> {
    for (const profile of opts.endpointProfiles) {
      const discovery = this.discoveryFor(profile.baseURL);
      const result = await discovery.discover(); // graceful: never throws
      const match = result.models.find((entry) => entry.id === model);
      if (match !== undefined) return match;
    }
    return undefined;
  }

  /**
   * Routing profiles for one request: configured endpoint capabilities merged
   * with DISCOVERED facts that are already in the fresh cache (non-blocking —
   * routing never stalls on a probe; a stale cache simply means no enrichment
   * this request). Configured capabilities win per field.
   */
  private routingProfiles(
    opts: ResolvedAdapterOptions,
    auth: LlamaCppAuth | undefined,
    model: string,
  ): readonly EndpointRoutingProfile[] {
    if (!opts.discovery.enabled) return opts.endpointProfiles;
    return opts.endpointProfiles.map((profile) => {
      const cached = this.discoveryFor(profile.baseURL, auth).discoverCached();
      const facts = cached?.models.find((entry) => entry.id === model);
      const merged = mergeCapabilities(profile.capabilities, facts);
      const capabilities: EndpointCapabilities = {
        ...(profile.capabilities?.models !== undefined ? { models: profile.capabilities.models } : {}),
        ...(profile.capabilities?.workload !== undefined ? { workload: profile.capabilities.workload } : {}),
        ...(merged.contextWindow !== undefined ? { contextWindow: merged.contextWindow } : {}),
        ...(merged.tools !== undefined ? { tools: merged.tools } : {}),
        ...(merged.reasoning !== undefined ? { reasoning: merged.reasoning } : {}),
      };
      const hasCapabilities = Object.keys(capabilities).length > 0;
      return { baseURL: profile.baseURL, ...(hasCapabilities ? { capabilities } : {}) };
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
    let feedbackEnabled = false;

    const finishWith = (failureCode?: string): void => {
      const totalMs = Date.now() - startedAt;
      const outcome = {
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
      };
      sink.emit({ type: 'finished', requestId, at: startedAt + totalMs, outcome });
      // Feedback loop (issue #11): record the bounded provider-observable
      // outcome so the next reasoning decision can learn from it.
      if (feedbackEnabled) {
        this.history.record(feedbackFrom(outcome));
      }
    };

    try {
      const opts = this.deps.options();
      feedbackEnabled = opts.reasoning.feedback?.enabled === true;
      const context = policyContext(options, opts.reasoning.adaptive?.hints);
      // The adapter depends only on the inference-policy seam; it has no
      // knowledge of whether the resolved policy is static, adaptive, or
      // feedback-informed.
      const reasoning = buildFeedbackPolicy(opts.reasoning, this.history).resolve({
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
      // The adapter depends only on the routing-policy seam; it has no
      // knowledge of the capability-filtering implementation.
      const routingPolicy = this.deps.routing ?? defaultRoutingPolicy;
      const routing = routingPolicy.route(routingRequest, this.routingProfiles(opts, auth, options.model));
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

/**
 * Derive the bounded feedback record from a request outcome (issue #11).
 * Provider-observable signals only: outcome class, failure code, retry/
 * fallback use, reasoning tokens, latency, and finish reason. Tool-call
 * retries execute outside the provider (Harness `ctx.tools`), so the provider
 * cannot observe them: the field is deliberately OMITTED (unknown), never
 * recorded as a known `false`.
 */
function feedbackFrom(outcome: RequestOutcome): ReasoningFeedback {
  const failureCode = outcome.failureCode;
  const kind: ReasoningFeedback['outcome'] = failureCode === 'TIMEOUT'
    ? 'timeout'
    : failureCode === 'ABORTED'
      ? 'aborted'
      : failureCode !== undefined
        ? 'failure'
        : 'success';
  return {
    outcome: kind,
    ...(failureCode !== undefined ? { failureCode } : {}),
    retried: outcome.retryCount > 0 || outcome.fallbackCount > 0,
    ...(outcome.usage?.reasoningTokens !== undefined ? { reasoningTokens: outcome.usage.reasoningTokens } : {}),
    latencyMs: outcome.totalMs,
    ...(outcome.finishReason !== undefined ? { finishReason: outcome.finishReason.kind } : {}),
  };
}

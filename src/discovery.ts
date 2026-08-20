/**
 * llama.cpp model and capability discovery (issue #10).
 *
 * Probes `/v1/models` (model ids, context-window metadata in `data[].meta`)
 * and `/props` (loaded model alias, slot context) through a small discovery
 * client, with a bounded TTL cache, single-flight dedup, cancellation, and
 * graceful degradation: a probe failure never breaks an otherwise valid
 * explicitly configured deployment — it yields an empty/degraded result that
 * callers fall back to configured facts.
 *
 * Precedence: user-configured endpoint capabilities (issue #9) win over
 * discovered values; discovered values fill gaps. Tool/reasoning support is
 * only set when the server states it explicitly (markers in a capabilities
 * list); absence is "unknown", which routing treats as assumed supported —
 * discovery must never cause a false negative that routes a request away from
 * a capable endpoint.
 *
 * @module llm-llamacpp/discovery
 */
import { LlmError } from '@deepseek-ai/dsh-llm';
import type { LlamaCppAuth } from './client.ts';

/** Discovered capability facts about one model on one endpoint. */
export interface DiscoveredModel {
  readonly id: string;
  /** Context window in tokens, when the server exposes it. */
  readonly contextWindow?: number;
  /** Tool-call support, only when the server states it explicitly. */
  readonly supportsTools?: boolean;
  /** Reasoning/thinking support, only when the server states it explicitly. */
  readonly supportsReasoning?: boolean;
}

/** One discovery round: models plus optional degradation reason. */
export interface DiscoveryResult {
  /** Epoch ms when the discovery was performed. */
  readonly at: number;
  readonly models: readonly DiscoveredModel[];
  /** Degradation reason when a probe failed or was aborted. */
  readonly error?: string;
}

/** Discovery tuning: bounded cache TTL and per-probe timeout. */
export interface DiscoveryOptions {
  /** Per-probe timeout in ms; default 5000. */
  readonly timeoutMs?: number;
  /** Cache TTL in ms; default 300000. */
  readonly ttlMs?: number;
  /** Auth header for probes when the server requires one. */
  readonly auth?: LlamaCppAuth;
}

export const DEFAULT_DISCOVERY_TIMEOUT_MS = 5_000;
export const DEFAULT_DISCOVERY_TTL_MS = 300_000;

/** Extract a stable model id from the many shapes llama.cpp/Ollama return. */
function modelIdOf(entry: Record<string, unknown>): string | undefined {
  const id = entry.id;
  if (typeof id === 'string' && id.length > 0) return id;
  const model = entry.model;
  if (typeof model === 'string' && model.length > 0) return model;
  const name = entry.name;
  if (typeof name === 'string' && name.length > 0) return name;
  return undefined;
}

/** Context window from `meta` when the server exposes it. */
function contextOf(meta: unknown): number | undefined {
  if (meta === null || typeof meta !== 'object') return undefined;
  const record = meta as Record<string, unknown>;
  for (const key of ['n_ctx', 'n_ctx_train']) {
    const value = record[key];
    if (typeof value === 'number' && Number.isSafeInteger(value) && value > 0) return value;
  }
  return undefined;
}

/** Explicit capability markers -> support facts; absence stays unknown. */
function supportOf(capabilities: unknown): { supportsTools?: boolean; supportsReasoning?: boolean } {
  if (!Array.isArray(capabilities)) return {};
  const values = capabilities.filter((v): v is string => typeof v === 'string').map((v) => v.toLowerCase());
  const tools = values.some((v) => v === 'tools' || v === 'tool_calls' || v === 'function' || v === 'functions');
  const reasoning = values.some((v) => v === 'reasoning' || v === 'thinking');
  return {
    ...(tools ? { supportsTools: true } : {}),
    ...(reasoning ? { supportsReasoning: true } : {}),
  };
}

/** Parse the OpenAI-compatible /v1/models payload into discovered models. */
function parseModelList(payload: unknown): DiscoveredModel[] {
  if (payload === null || typeof payload !== 'object') return [];
  const root = payload as { data?: unknown; models?: unknown };
  const byId = new Map<string, DiscoveredModel>();
  const add = (entry: unknown): void => {
    if (entry === null || typeof entry !== 'object') return;
    const record = entry as Record<string, unknown>;
    const id = modelIdOf(record);
    if (id === undefined) return;
    const contextWindow = contextOf(record.meta);
    const support = supportOf(record.capabilities);
    const existing = byId.get(id);
    if (existing === undefined) {
      byId.set(id, { id, ...(contextWindow !== undefined ? { contextWindow } : {}), ...support });
    } else {
      // Merge: prefer whatever metadata is present (first wins per field).
      byId.set(id, {
        id,
        contextWindow: existing.contextWindow ?? contextWindow,
        supportsTools: existing.supportsTools ?? support.supportsTools,
        supportsReasoning: existing.supportsReasoning ?? support.supportsReasoning,
      });
    }
  };
  for (const entry of Array.isArray(root.data) ? root.data : []) add(entry);
  for (const entry of Array.isArray(root.models) ? root.models : []) add(entry);
  return [...byId.values()];
}

/** Parse /props for the loaded model alias and slot context window. */
function parseProps(payload: unknown): { loadedModelId?: string; nCtx?: number } {
  if (payload === null || typeof payload !== 'object') return {};
  const root = payload as Record<string, unknown>;
  const alias = root.model_alias ?? root.model;
  const loadedModelId = typeof alias === 'string' && alias.length > 0 ? alias : undefined;
  const nCtx = typeof root.n_ctx === 'number' && Number.isSafeInteger(root.n_ctx) && root.n_ctx > 0 ? root.n_ctx : undefined;
  return { ...(loadedModelId !== undefined ? { loadedModelId } : {}), ...(nCtx !== undefined ? { nCtx } : {}) };
}

/**
 * Per-endpoint discovery manager with a bounded TTL cache, single-flight
 * dedup, and cancellation. `discover()` never throws for probe failures — it
 * returns a degraded result (empty models + `error`), so discovery can never
 * break a configured deployment.
 */
export class EndpointDiscovery {
  private cache: { result: DiscoveryResult; at: number } | undefined;
  private inflight: Promise<DiscoveryResult> | undefined;

  constructor(
    readonly baseURL: string,
    readonly options: DiscoveryOptions = {},
  ) {}

  private timeoutMs(): number {
    return this.options.timeoutMs ?? DEFAULT_DISCOVERY_TIMEOUT_MS;
  }

  private ttlMs(): number {
    return this.options.ttlMs ?? DEFAULT_DISCOVERY_TTL_MS;
  }

  private aborted(signal?: AbortSignal): never {
    throw new LlmError('llama.cpp discovery aborted by caller', 'ABORTED');
  }

  /**
   * Return the current model list, honoring the bounded TTL cache and
   * single-flight concurrency; a fresh probe runs only when the cache is
   * stale and none is in flight. Caller cancellation aborts promptly.
   */
  async discover(signal?: AbortSignal): Promise<DiscoveryResult> {
    const now = Date.now();
    if (this.cache !== undefined && now - this.cache.at < this.ttlMs()) {
      if (signal?.aborted) this.aborted(signal);
      return this.cache.result;
    }
    if (this.inflight !== undefined) {
      if (signal?.aborted) this.aborted(signal);
      const result = await this.inflight;
      this.cache ??= { result, at: Date.now() };
      return result;
    }
    this.inflight = this.probe(signal).then((result) => {
      this.cache = { result, at: Date.now() };
      return result;
    }).finally(() => {
      this.inflight = undefined;
    });
    return this.inflight;
  }

  /** Force a fresh probe, bypassing the cache (used by tests and diagnostics). */
  async refresh(signal?: AbortSignal): Promise<DiscoveryResult> {
    if (signal?.aborted) this.aborted(signal);
    const result = await this.probe(signal);
    this.cache = { result, at: Date.now() };
    return result;
  }

  /** Invalidate the cache so the next discover() re-probes. */
  clear(): void {
    this.cache = undefined;
  }

  /**
   * Non-blocking cache read for latency-critical paths (e.g. per-request
   * routing): returns the fresh cached result, or `undefined` when the cache
   * is stale or absent. Never triggers a probe.
   */
  discoverCached(): DiscoveryResult | undefined {
    const now = Date.now();
    if (this.cache !== undefined && now - this.cache.at < this.ttlMs()) return this.cache.result;
    return undefined;
  }

  private async probe(signal?: AbortSignal): Promise<DiscoveryResult> {
    const at = Date.now();
    const modelUrl = `${this.baseURL.replace(/\/+$/, '')}/v1/models`;
    const propsUrl = `${this.baseURL.replace(/\/+$/, '')}/props`;
    let models: DiscoveredModel[] = [];
    const errors: string[] = [];
    try {
      // Each probe fails independently: a missing /v1/models can still yield
      // the loaded model from /props, and vice versa. Cancellation is NOT
      // degraded — the catches rethrow it so callers see the abort.
      const modelsPayload = await this.getJson(modelUrl, signal).catch((error) => {
        if (signal?.aborted) throw error;
        errors.push(error instanceof Error ? error.message : String(error));
        return undefined;
      });
      const propsPayload = await this.getJson(propsUrl, signal).catch((error) => {
        if (signal?.aborted) throw error;
        return undefined;
      });
      models = parseModelList(modelsPayload);
      const props = parseProps(propsPayload);
      // /props supplies the slot context when the model list lacks meta, and
      // a single loaded model id when the model list is unavailable.
      if (models.length === 0 && props.loadedModelId !== undefined) {
        models = [{ id: props.loadedModelId, ...(props.nCtx !== undefined ? { contextWindow: props.nCtx } : {}) }];
      } else if (props.nCtx !== undefined && models.length === 1 && models[0]?.contextWindow === undefined) {
        models = [{ ...models[0]!, contextWindow: props.nCtx }];
      }
    } catch (error) {
      // Cancellation must propagate, never degrade into an empty result.
      if (signal?.aborted) throw error;
      errors.push(error instanceof Error ? error.message : String(error));
    }
    if (models.length === 0 && errors.length > 0) {
      return { at, models, error: errors.join('; ') };
    }
    return {
      at,
      models,
      ...(errors.length > 0 ? { error: errors.join('; ') } : {}),
    };
  }

  private async getJson(url: string, signal?: AbortSignal): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs());
    const combined = signal !== undefined ? AbortSignal.any([signal, controller.signal]) : controller.signal;
    try {
      const response = await fetch(url, {
        signal: combined,
        ...(this.options.auth !== undefined ? { headers: { [this.options.auth.name]: this.options.auth.value } } : {}),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} from ${url}`);
      }
      return await response.json();
    } finally {
      clearTimeout(timer);
      if (signal?.aborted) this.aborted(signal);
    }
  }
}

/**
 * Merge configured endpoint capabilities over discovered facts for one model
 * (configured wins per field; discovered fills gaps). `undefined` capabilities
 * mean "unknown", which routing treats as assumed supported.
 */
export function mergeCapabilities(
  configured: { contextWindow?: number; tools?: boolean; reasoning?: boolean } | undefined,
  discovered: DiscoveredModel | undefined,
): { contextWindow?: number; tools?: boolean; reasoning?: boolean } {
  const hasContext = configured?.contextWindow !== undefined || discovered?.contextWindow !== undefined;
  const hasTools = configured?.tools !== undefined || discovered?.supportsTools !== undefined;
  const hasReasoning = configured?.reasoning !== undefined || discovered?.supportsReasoning !== undefined;
  return {
    ...(hasContext ? { contextWindow: configured?.contextWindow ?? discovered?.contextWindow } : {}),
    ...(hasTools ? { tools: configured?.tools ?? discovered?.supportsTools } : {}),
    ...(hasReasoning ? { reasoning: configured?.reasoning ?? discovered?.supportsReasoning } : {}),
  };
}

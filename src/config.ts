/**
 * Plugin configuration schema and the single explicit resolve step from raw
 * config to validated connection facts.
 *
 * The schema (`Config`) is what a configuration surface renders and what an
 * absent settings section resolves through; `resolveAdapterOptions` is where
 * every default and bound is re-judged, because programmatic construction may
 * bypass Schemastery normalization. The adapter reads connection facts through
 * a thunk re-evaluated per operation, so a changed base URL, model, or policy
 * reaches the very next request without restarting anything.
 *
 * @module llm-llamacpp/config
 */
import z from '@deepseek-ai/schemastery';
import {
  RetryPolicySchema,
  resolveRetryPolicy,
  type ResolvedRetryPolicy,
  type RetryPolicyConfig,
} from '@deepseek-ai/dsh-llm';
import {
  validateReasoningConfig,
  type ReasoningExpertOverride,
  type ReasoningLevel,
  type ReasoningPolicyConfig,
  type ReasoningWireMode,
} from './reasoning.ts';
import {
  defaultReasoningWire,
  familyProfileFor,
  type ModelFamilyProfile,
} from './compat.ts';
import type { EndpointCapabilities, EndpointRoutingProfile } from './routing.ts';

/** Cordis plugin short name; also the settings namespace and npm package name. */
export const PLUGIN_NAME = 'llm-llamacpp';
/** The single provider route this plugin owns. */
export const PROVIDER = 'llamacpp-local';
/** Default human-readable provider name for selectors and diagnostics. */
export const DEFAULT_PROVIDER_NAME = 'llama.cpp (Local)';
/** Default local llama.cpp OpenAI-compatible endpoint. */
export const DEFAULT_BASE_URL = 'http://127.0.0.1:8080';
/** Default model id passed to the wire `model` field when no model is configured. */
export const DEFAULT_MODEL = 'qwen3';
/** Default header carrying the API key when one is configured (`Authorization: Bearer <key>`). */
export const DEFAULT_API_KEY_HEADER = 'authorization';
/** Default maximum idle interval while a provider stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000;

/** Expert/advanced reasoning override surface (see reasoning.ts). */
const ReasoningExpertSchema = z.object({
  enabled: z.boolean(),
  effort: z.string(),
  budgetTokens: z.number().step(1).min(1),
  preserveThinking: z.boolean(),
  emitThinking: z.boolean(),
});

/** Adaptive budget bounds and hints (issue #6). */
const AdaptiveSchema = z.object({
  enabled: z.boolean().default(false),
  defaultBudgetTokens: z.number().step(1).min(1),
  minBudgetTokens: z.number().step(1).min(1),
  maxBudgetTokens: z.number().step(1).min(1),
  hints: z.array(z.string()),
});

/** Feedback-informed budget adjustment (issue #11). */
const FeedbackSchema = z.object({
  enabled: z.boolean().default(false),
});

/** Semantic reasoning configuration; wire translation happens in serialize.ts. */
const ReasoningSchema = z.object({
  /**
   * Master thinking switch; `false` advertises and allows only `off`.
   * `.extra('extra', {controls: [...]})` is a generic schema-level hint
   * consumed by the DSH Models-page generic provider editor (upstream issue
   * #14): it disables the named sibling fields while this switch is off, so
   * the UI never offers a preset the adapter would ignore.
   */
  enabled: z.boolean().default(true).extra('extra', { controls: ['preset'] }),
  /** Default semantic level when a request names none. */
  preset: z.union(['off', 'low', 'medium', 'xhigh']).default('medium'),
  /** Explicit expert override surface; never rewrites the preset table. */
  expert: ReasoningExpertSchema.extra('extra', { ui: { label: 'Expert override', collapsed: true } }),
  /** llama.cpp wire translation mode; `none` sends no reasoning wire fields (issue #18). */
  wire: z.union(['chat-template-kwargs', 'reasoning-fields', 'none']).default('none'),
  /** Optional adaptive budget adjustment from request context. */
  adaptive: AdaptiveSchema.extra('extra', { ui: { label: 'Adaptive budget', collapsed: true } }),
  /** Optional feedback-informed budget adjustment (issue #11). */
  feedback: FeedbackSchema.extra('extra', { ui: { label: 'Feedback', collapsed: true } }),
});

/** Structured observability toggle (issue #8). */
const TelemetrySchema = z.object({
  enabled: z.boolean().default(true),
});

/** llama.cpp model/capability discovery (issue #10); disabled by default. */
const DiscoverySchema = z.object({
  enabled: z.boolean().default(false),
  ttlMs: z.number().step(1).min(1),
  timeoutMs: z.number().step(1).min(1),
});

/** Bounded diagnostics store (issue #12); on by default, passive and cheap. */
const DiagnosticsSchema = z.object({
  enabled: z.boolean().default(true),
});

/** Endpoint capability metadata for capability-aware routing (issue #9). */
const EndpointCapabilitiesSchema = z.object({
  models: z.array(z.string()),
  contextWindow: z.number().step(1).min(1),
  tools: z.boolean(),
  reasoning: z.boolean(),
  workload: z.array(z.string()),
});

/** One endpoint entry: a plain URL, or a URL plus capability metadata. */
const EndpointSchema = z.union([
  z.string(),
  z.object({
    url: z.string(),
    capabilities: EndpointCapabilitiesSchema,
  }),
]);

/**
 * Plugin entry config. Every field is optional so the plugin loads with
 * llama.cpp's own defaults; `baseURL` still fails clearly when it is present
 * but not a valid http(s) URL (see {@link resolveAdapterOptions}).
 */
export const Config = z.object({
  /**
   * Base URL of the llama.cpp OpenAI-compatible server, e.g. `http://127.0.0.1:8080`.
   * UI metadata (issue #19): user-facing label + concise help for the generic editor.
   */
  baseURL: z.string().default(DEFAULT_BASE_URL)
    .extra('extra', { ui: { label: 'Base URL', description: 'OpenAI-compatible endpoint of the llama.cpp server.' } }),
  /** Human-readable provider name surfaced by selectors and diagnostics. */
  providerName: z.string().default(DEFAULT_PROVIDER_NAME)
    .extra('extra', { ui: { label: 'Display name', description: 'Name shown in model selectors and diagnostics.' } }),
  /** Default model id sent to the wire `model` field. */
  model: z.string().default(DEFAULT_MODEL)
    .extra('extra', { ui: { label: 'Model', description: 'Model id the server accepts, as listed by /v1/models.' } }),
  /**
   * Explicit model-family selector (issue #18). `'auto'` resolves without a
   * model-name heuristic to the `unknown` compatibility profile — nothing is
   * silently inherited from the Qwen profile; `'qwen'` selects the validated
   * Qwen compatibility profile explicitly. Explicit `reasoning.wire` always
   * beats either profile's default.
   */
  modelFamily: z.union(['auto', 'qwen']).default('auto')
    .extra('extra', { ui: { label: 'Model family', description: 'Enables model-family compatibility behavior; keep Auto when unsure.' } }),
  /**
   * Optional environment variable naming the API key. Local llama.cpp needs
   * none; a reverse proxy in front of it may require one. The value is read
   * per request and never stored in the settings document.
   */
  apiKeyEnv: z.string().role('credential-ref')
    .extra('extra', { ui: { label: 'API key env var', description: 'Environment variable or credential reference for the API key.' } }),
  /** Header that carries the key: `authorization` sends `Bearer <key>`, anything else sends the raw key. */
  apiKeyHeader: z.string().default(DEFAULT_API_KEY_HEADER)
    .extra('extra', { ui: { label: 'API key header', description: 'Header carrying the key; authorization sends Bearer <key>.' } }),
  /** Maximum idle interval (ms) for one outstanding provider stream read. */
  streamIdleTimeoutMs: z.number().min(1).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS)
    .extra('extra', { ui: { label: 'Stream idle timeout (ms)', description: 'How long a streaming response may stay idle before it is treated as timed out.' } }),
  /**
   * Hard per-request-attempt timeout (ms), regardless of activity. Optional;
   * absent means no total deadline (the idle watchdog still applies).
   */
  requestTimeoutMs: z.number().step(1).min(1)
    .extra('extra', { ui: { label: 'Request timeout (ms)', description: 'Hard deadline for one request attempt; absent means no total deadline.' } }),
  /**
   * Ordered list of llama.cpp endpoint base URLs for fallback (issue #7),
   * optionally with capability metadata for capability-aware routing
   * (issue #9). When present it replaces `baseURL` as the candidate set; the
   * first entry is the primary. Omission keeps single-endpoint behavior.
   */
  endpoints: z.array(EndpointSchema)
    .extra('extra', { ui: { label: 'Endpoints', collapsed: true } }),
  /** Provider-owned retry policy (reliability layer, issue #7). */
  retryPolicy: RetryPolicySchema
    .extra('extra', { ui: { label: 'Reliability', collapsed: true } }),
  /** Structured request telemetry (issue #8); disable to stop emission. */
  telemetry: TelemetrySchema
    .extra('extra', { ui: { label: 'Telemetry', collapsed: true } }),
  /** Model/capability discovery (issue #10); off keeps single-server simple. */
  discovery: DiscoverySchema
    .extra('extra', { ui: { label: 'Discovery', collapsed: true } }),
  /** Bounded production diagnostics (issue #12). */
  diagnostics: DiagnosticsSchema
    .extra('extra', { ui: { label: 'Diagnostics', collapsed: true } }),
  /** Semantic reasoning controls (model-family aware; Qwen is the validated family). */
  reasoning: ReasoningSchema
    .extra('extra', { ui: { label: 'Reasoning', collapsed: true } }),
});

export type ConfigType = {
  /** Base URL of the llama.cpp OpenAI-compatible server, e.g. `http://127.0.0.1:8080`. */
  baseURL?: string;
  /** Human-readable provider name surfaced by selectors and diagnostics. */
  providerName?: string;
  /** Default model id sent to the wire `model` field. */
  model?: string;
  /** Explicit model-family selector; `'auto'` resolves to the unknown profile (no heuristic). */
  modelFamily?: 'auto' | 'qwen';
  /** Optional environment variable naming the API key. */
  apiKeyEnv?: string;
  /** Header that carries the key: `authorization` sends `Bearer <key>`, anything else sends the raw key. */
  apiKeyHeader?: string;
  /** Maximum idle interval (ms) for one outstanding provider stream read. */
  streamIdleTimeoutMs?: number;
  /** Hard per-request-attempt timeout (ms), regardless of activity. */
  requestTimeoutMs?: number;
  /** Ordered fallback endpoint list; replaces `baseURL` when present. */
  endpoints?: (string | { url: string; capabilities?: EndpointCapabilities })[];
  /** Provider-owned retry policy (reliability layer, issue #7). */
  retryPolicy?: RetryPolicyConfig;
  /** Structured request telemetry (issue #8). */
  telemetry?: { enabled?: boolean };
  /** Model/capability discovery (issue #10). */
  discovery?: { enabled?: boolean; ttlMs?: number; timeoutMs?: number };
  /** Bounded production diagnostics (issue #12). */
  diagnostics?: { enabled?: boolean };
  /** Semantic reasoning controls (model-family aware). */
  reasoning?: {
    enabled?: boolean;
    preset?: ReasoningLevel;
    expert?: ReasoningExpertOverride;
    wire?: ReasoningWireMode;
    adaptive?: {
      enabled?: boolean;
      defaultBudgetTokens?: number;
      minBudgetTokens?: number;
      maxBudgetTokens?: number;
      hints?: string[];
    };
    /** Feedback-informed budget adjustment (issue #11). */
    feedback?: { enabled?: boolean };
  };
};

/** Validated, detached connection facts the adapter reads per operation. */
export interface ResolvedAdapterOptions {
  readonly providerName: string;
  /** Primary endpoint (first candidate). */
  readonly baseURL: string;
  /** Ordered candidate endpoints (fallback order); `[baseURL]` when unset. */
  readonly endpoints: readonly string[];
  /** Same endpoints with optional capability metadata (issue #9). */
  readonly endpointProfiles: readonly EndpointRoutingProfile[];
  readonly model: string;
  /** Environment variable naming the API key, when one is configured. */
  readonly apiKeyEnv?: string;
  readonly apiKeyHeader: string;
  readonly streamIdleTimeoutMs: number;
  /** Hard per-request-attempt timeout (ms), when configured. */
  readonly requestTimeoutMs?: number;
  /** Provider-owned retry policy captured at registration. */
  readonly retryPolicy: ResolvedRetryPolicy;
  /** Structured request telemetry toggle (issue #8). */
  readonly telemetry: { enabled: boolean };
  /** Model/capability discovery tuning (issue #10). */
  readonly discovery: { enabled: boolean; ttlMs?: number; timeoutMs?: number };
  /** Bounded diagnostics toggle (issue #12). */
  readonly diagnostics: { enabled: boolean };
  /** Model-family compatibility profile (issue #18); never inferred from the model name. */
  readonly family: ModelFamilyProfile;
  readonly reasoning: ReasoningPolicyConfig;
}

/** Reject a base URL that is present but not a usable http(s) endpoint. */
function validateBaseURL(value: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`llm-llamacpp: baseURL "${value}" is not a valid URL`);
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`llm-llamacpp: baseURL "${value}" must use http or https`);
  }
  if (parsed.hostname.length === 0) {
    throw new Error(`llm-llamacpp: baseURL "${value}" has no host`);
  }
  return parsed.toString().replace(/\/+$/, '');
}

/**
 * The one explicit resolve step from raw config to validated connection
 * facts. Throws on invalid values so a bad entry config fails the plugin load
 * clearly, and a bad settings snapshot keeps the last good facts instead.
 * @param config - raw plugin config or resolved settings snapshot.
 * @returns validated, detached connection facts.
 */
export function resolveAdapterOptions(config: ConfigType): ResolvedAdapterOptions {
  const baseURL = validateBaseURL(config.baseURL ?? DEFAULT_BASE_URL);
  const providerName = (config.providerName ?? DEFAULT_PROVIDER_NAME).trim();
  if (providerName.length === 0) throw new Error('llm-llamacpp: providerName must not be empty');
  const model = (config.model ?? DEFAULT_MODEL).trim();
  if (model.length === 0) throw new Error('llm-llamacpp: model must not be empty');
  const apiKeyHeader = (config.apiKeyHeader ?? DEFAULT_API_KEY_HEADER).trim().toLowerCase();
  if (apiKeyHeader.length === 0) throw new Error('llm-llamacpp: apiKeyHeader must not be empty');
  const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS;
  if (!Number.isFinite(streamIdleTimeoutMs) || streamIdleTimeoutMs <= 0) {
    throw new Error('llm-llamacpp: streamIdleTimeoutMs must be a positive finite number');
  }
  const apiKeyEnv = config.apiKeyEnv?.trim();
  const requestTimeoutMs = config.requestTimeoutMs;
  if (requestTimeoutMs !== undefined && (!Number.isSafeInteger(requestTimeoutMs) || requestTimeoutMs <= 0)) {
    throw new Error('llm-llamacpp: requestTimeoutMs must be a positive safe integer');
  }
  // Ordered fallback candidate list; `endpoints` replaces `baseURL` when set.
  // Each entry is a plain URL or a URL + capability profile (issue #9).
  const rawEndpoints: readonly (string | { url: string; capabilities?: EndpointCapabilities })[] =
    config.endpoints !== undefined && config.endpoints.length > 0 ? config.endpoints : [baseURL];
  const seen = new Set<string>();
  const endpointProfiles: EndpointRoutingProfile[] = [];
  for (const raw of rawEndpoints) {
    const url = validateBaseURL(typeof raw === 'string' ? raw : raw.url);
    if (seen.has(url)) continue;
    seen.add(url);
    const capabilities = typeof raw === 'string' ? undefined : raw.capabilities;
    if (capabilities !== undefined) {
      if (capabilities.contextWindow !== undefined && (!Number.isSafeInteger(capabilities.contextWindow) || capabilities.contextWindow <= 0)) {
        throw new Error(`llm-llamacpp: endpoints[${url}].contextWindow must be a positive safe integer`);
      }
      for (const list of [capabilities.models, capabilities.workload] as const) {
        if (list !== undefined && list.some((item) => typeof item !== 'string' || item.length === 0)) {
          throw new Error(`llm-llamacpp: endpoints[${url}] model/workload lists must contain only non-empty strings`);
        }
      }
    }
    endpointProfiles.push({ baseURL: url, ...(capabilities !== undefined ? { capabilities } : {}) });
  }
  const endpoints = endpointProfiles.map((profile) => profile.baseURL);
  const reasoningRaw = config.reasoning ?? {};
  const adaptiveRaw = reasoningRaw.adaptive;
  const feedbackRaw = reasoningRaw.feedback;
  // Model-family profile: explicit selector only — never a model-name guess.
  // Explicit reasoning.wire beats the profile's default below.
  const family = familyProfileFor(config.modelFamily);
  const reasoning: ReasoningPolicyConfig = {
    enabled: reasoningRaw.enabled ?? true,
    preset: reasoningRaw.preset ?? 'medium',
    ...(reasoningRaw.expert !== undefined ? { expert: reasoningRaw.expert } : {}),
    wire: reasoningRaw.wire ?? defaultReasoningWire(family),
    ...(adaptiveRaw !== undefined
      ? {
          adaptive: {
            enabled: adaptiveRaw.enabled ?? false,
            ...(adaptiveRaw.defaultBudgetTokens !== undefined ? { defaultBudgetTokens: adaptiveRaw.defaultBudgetTokens } : {}),
            ...(adaptiveRaw.minBudgetTokens !== undefined ? { minBudgetTokens: adaptiveRaw.minBudgetTokens } : {}),
            ...(adaptiveRaw.maxBudgetTokens !== undefined ? { maxBudgetTokens: adaptiveRaw.maxBudgetTokens } : {}),
            ...(adaptiveRaw.hints !== undefined && adaptiveRaw.hints.length > 0 ? { hints: adaptiveRaw.hints } : {}),
          },
        }
      : {}),
    ...(feedbackRaw !== undefined ? { feedback: { enabled: feedbackRaw.enabled ?? false } } : {}),
  };
  validateReasoningConfig(reasoning, 'llm-llamacpp: reasoning');
  return {
    providerName,
    baseURL: endpoints[0] ?? baseURL,
    endpoints,
    endpointProfiles,
    model,
    ...apiKeyEnv !== undefined && apiKeyEnv.length > 0 ? { apiKeyEnv } : {},
    apiKeyHeader,
    streamIdleTimeoutMs,
    ...(requestTimeoutMs !== undefined ? { requestTimeoutMs } : {}),
    retryPolicy: resolveRetryPolicy(config.retryPolicy, 'llm-llamacpp: retryPolicy'),
    telemetry: { enabled: config.telemetry?.enabled ?? true },
    discovery: {
      enabled: config.discovery?.enabled ?? false,
      ...(config.discovery?.ttlMs !== undefined ? { ttlMs: config.discovery.ttlMs } : {}),
      ...(config.discovery?.timeoutMs !== undefined ? { timeoutMs: config.discovery.timeoutMs } : {}),
    },
    diagnostics: { enabled: config.diagnostics?.enabled ?? true },
    family,
    reasoning,
  };
}

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

/**
 * Plugin entry config. Every field is optional so the plugin loads with
 * llama.cpp's own defaults; `baseURL` still fails clearly when it is present
 * but not a valid http(s) URL (see {@link resolveAdapterOptions}).
 */
export const Config = z.object({
  /** Base URL of the llama.cpp OpenAI-compatible server, e.g. `http://127.0.0.1:8080`. */
  baseURL: z.string().default(DEFAULT_BASE_URL),
  /** Human-readable provider name surfaced by selectors and diagnostics. */
  providerName: z.string().default(DEFAULT_PROVIDER_NAME),
  /** Default model id sent to the wire `model` field. */
  model: z.string().default(DEFAULT_MODEL),
  /**
   * Optional environment variable naming the API key. Local llama.cpp needs
   * none; a reverse proxy in front of it may require one. The value is read
   * per request and never stored in the settings document.
   */
  apiKeyEnv: z.string(),
  /** Header that carries the key: `authorization` sends `Bearer <key>`, anything else sends the raw key. */
  apiKeyHeader: z.string().default(DEFAULT_API_KEY_HEADER),
  /** Maximum idle interval (ms) for one outstanding provider stream read. */
  streamIdleTimeoutMs: z.number().min(1).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  /** Provider-owned request retry policy captured at registration. */
  retryPolicy: RetryPolicySchema,
});

export type ConfigType = {
  /** Base URL of the llama.cpp OpenAI-compatible server, e.g. `http://127.0.0.1:8080`. */
  baseURL?: string;
  /** Human-readable provider name surfaced by selectors and diagnostics. */
  providerName?: string;
  /** Default model id sent to the wire `model` field. */
  model?: string;
  /** Optional environment variable naming the API key. */
  apiKeyEnv?: string;
  /** Header that carries the key: `authorization` sends `Bearer <key>`, anything else sends the raw key. */
  apiKeyHeader?: string;
  /** Maximum idle interval (ms) for one outstanding provider stream read. */
  streamIdleTimeoutMs?: number;
  /** Provider-owned request retry policy captured at registration. */
  retryPolicy?: RetryPolicyConfig;
};

/** Validated, detached connection facts the adapter reads per operation. */
export interface ResolvedAdapterOptions {
  readonly providerName: string;
  readonly baseURL: string;
  readonly model: string;
  /** Environment variable naming the API key, when one is configured. */
  readonly apiKeyEnv?: string;
  readonly apiKeyHeader: string;
  readonly streamIdleTimeoutMs: number;
  readonly retryPolicy: ResolvedRetryPolicy;
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
  return {
    providerName,
    baseURL,
    model,
    ...apiKeyEnv !== undefined && apiKeyEnv.length > 0 ? { apiKeyEnv } : {},
    apiKeyHeader,
    streamIdleTimeoutMs,
    retryPolicy: resolveRetryPolicy(config.retryPolicy as ConfigType['retryPolicy'], 'llm-llamacpp: retryPolicy'),
  };
}

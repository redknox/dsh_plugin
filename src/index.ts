/**
 * Cordis plugin entrypoint for the `llm-llamacpp` DeepSeek Harness LLM
 * provider.
 *
 * Loads the package as a plugin (`name: 'llm-llamacpp'` in the harness
 * composition), validates the entry config, registers one `LlamacppAdapter`
 * for the `llamacpp-local` provider route on the public `ctx.llm` seam, and
 * declares the route in the configurable-provider directory. Connection facts
 * are resolved per operation through a thunk instead of frozen at load, so the
 * optional `llm-llamacpp` user-settings section (`ctx.settings`) can override
 * any field without a restart. The adapter registration and directory entry
 * ride the Cordis fiber and are released with the plugin's disposal — no
 * agent-loop internals are touched.
 *
 * @module llm-llamacpp
 */
import type { Context } from '@deepseek-ai/cordis';
import { credentialRef } from '@deepseek-ai/dsh-credentials';
import { LlmError, assertUsableApiKey } from '@deepseek-ai/dsh-llm';
import { deepEqualJson, installSettingsSection, settingsNamespace } from '@deepseek-ai/dsh-settings';
import { DiagnosticsStore } from './diagnostics.ts';
import { LlamacppAdapter } from './adapter.ts';
import {
  Config,
  DEFAULT_PROVIDER_NAME,
  PLUGIN_NAME,
  PROVIDER,
  resolveAdapterOptions,
  type ConfigType,
  type ResolvedAdapterOptions,
} from './config.ts';
import { compositeTelemetry, logTelemetry, type TelemetrySink } from './telemetry.ts';

export { Config, DEFAULT_PROVIDER_NAME, PLUGIN_NAME, PROVIDER, resolveAdapterOptions };
export type { ConfigType, ResolvedAdapterOptions };

/** Cordis plugin short name; also the settings namespace. */
export const name = PLUGIN_NAME;
/** Require the `llm` service before `apply` runs. */
export const inject = ['llm'] as const;
/** Settings namespace owning this plugin's configurable-provider profile. */
export const NS = settingsNamespace(PLUGIN_NAME);

/**
 * Apply the plugin on one context.
 * @param ctx - the mounting context; `ctx.llm` is guaranteed by `inject`.
 * @param config - composition entry config (schema-normalized by the loader).
 */
export function apply(ctx: Context, config: ConfigType): void {
  let current: () => ConfigType = () => config;
  let lastRaw: ConfigType | undefined;
  let lastGood: ResolvedAdapterOptions | undefined;

  /** Re-read validated connection facts, keeping the last good on bad settings. */
  const options = (): ResolvedAdapterOptions => {
    const raw = current();
    if (raw === lastRaw && lastGood !== undefined) return lastGood;
    try {
      const next = resolveAdapterOptions(raw);
      lastRaw = raw;
      lastGood = next;
      return next;
    } catch (error) {
      if (lastGood === undefined) throw error;
      lastRaw = raw;
      ctx.logger.error('llm-llamacpp: keeping the last good configuration after an invalid settings section');
      ctx.logger.error(error);
      return lastGood;
    }
  };
  options(); // fail loud on an invalid entry config

  const resolveApiKey = async (): Promise<string | undefined> => {
    const opts = options();
    if (opts.apiKeyEnv === undefined) return undefined;
    // Resolve through the DSH credentials seam when mounted (its `env` layer
    // covers the launching environment and the GUI Models page can store the
    // key), falling back to a direct process.env read otherwise. A configured
    // reference that resolves nowhere fails clearly instead of silently
    // sending an unauthenticated request.
    const credentials = ctx.get('credentials');
    if (credentials !== undefined) {
      const hit = await credentials.resolve(credentialRef(opts.apiKeyEnv));
      if (hit !== undefined) return assertUsableApiKey(hit.value, PLUGIN_NAME, opts.apiKeyEnv);
    } else {
      const ambient = process.env[opts.apiKeyEnv];
      if (ambient !== undefined && ambient.length > 0) {
        return assertUsableApiKey(ambient, PLUGIN_NAME, opts.apiKeyEnv);
      }
    }
    throw new LlmError(
      `llm-llamacpp: no API key for "${opts.apiKeyEnv}"; store it through the credentials service ` +
        `(the web Models page writes it) or export it in the launching environment`,
      'MISSING_CREDENTIAL',
    );
  };

  // Bounded diagnostics store (issue #12): a passive telemetry consumer that
  // never retains content; its machine-readable snapshot is provided as a
  // context service for operators/tooling.
  const diagnosticsStore = new DiagnosticsStore();

  // Structured telemetry (issue #8): enabled by default as structured debug
  // log lines; `telemetry.enabled: false` disables emission without changing
  // provider behavior. The diagnostics store keeps consuming events whenever
  // diagnostics is enabled. Re-read per operation so toggles apply live.
  const telemetry = (): TelemetrySink => compositeTelemetry([
    ...(options().telemetry.enabled ? [logTelemetry(ctx.logger)] : []),
    ...(options().diagnostics.enabled ? [diagnosticsStore] : []),
  ]);
  const adapter = new LlamacppAdapter({ options, resolveApiKey, logger: ctx.logger, telemetry });

  const modelList = (): readonly import('./diagnostics.ts').DiagnosticsModel[] => adapter.diagnosticModels();
  const endpointUrls = (): string[] => options().endpointProfiles.map((profile) => profile.baseURL);
  ctx.provide('llm-llamacpp/diagnostics', {
    /** Machine-readable snapshot (issue #12). */
    snapshot: () => diagnosticsStore.snapshot(adapter.pool, endpointUrls(), modelList()),
    /** Human-readable rendering for local operations. */
    render: () => diagnosticsStore.render(adapter.pool, endpointUrls(), modelList()),
  });

  ctx.llm.registerConfigurableProviders([
    {
      provider: PROVIDER,
      displayName: options().providerName,
      settingsNs: NS,
      settingsPath: [],
    },
  ]);

  // Registration is disposed with this fiber (Cordis effect semantics), so
  // plugin unload unregisters the route and the directory entry automatically.
  const registration = ctx.llm.registerAdapter([PROVIDER], adapter);

  // The one registration-captured fact is the retry policy (issue #7):
  // re-register the route in place when it changes so providerRetryPolicy()
  // stays current.
  let registeredPolicy = options().retryPolicy;
  const ensureRegistrationFacts = (): void => {
    const policy = options().retryPolicy;
    if (deepEqualJson(policy, registeredPolicy)) return;
    registration.replace([PROVIDER]);
    registeredPolicy = policy;
  };

  installSettingsSection(ctx, NS, Config, config, {
    setSource: (source) => {
      // The settings scope hands back the schema-normalized shape, whose
      // optional fields the schemastery mapped type types as `| null`; our
      // manual ConfigType mirror omits the null branch, so assert it.
      current = source as unknown as () => ConfigType;
    },
    onChange: ensureRegistrationFacts,
  });
}

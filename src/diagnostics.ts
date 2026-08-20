/**
 * Production diagnostics (issue #12): a bounded, machine-readable snapshot
 * summarizing provider health, routing/reliability behavior, latency,
 * reasoning usage, and recent failures — sourced from the #8 telemetry events
 * and the #7 endpoint health state.
 *
 * {@link DiagnosticsStore} is a {@link TelemetrySink}: it consumes the same
 * structured events the adapter emits and maintains bounded counters and
 * rolling windows (no raw prompts, completions, tool arguments, or secrets —
 * events are content-free by construction and the store retains only field
 * names and counts). {@link renderDiagnostics} renders a human-readable
 * block for local operations/debugging. The core is framework-independent:
 * `snapshot()` is the machine-readable API.
 *
 * @module llm-llamacpp/diagnostics
 */
import type { EndpointPool } from './reliability.ts';
import type { TelemetryEvent, TelemetrySink } from './telemetry.ts';

/** Bounded rolling-window sizes (retention limits). */
export const MAX_LATENCY_WINDOW = 200;
export const MAX_RECENT_ROUTING = 20;
export const MAX_RECENT_FAILURES = 20;
export const MAX_RATE_SAMPLES = 300;
export const MAX_REASONING_BUDGET_WINDOW = 100;
/** Request-rate observation window. */
export const RATE_WINDOW_MS = 60_000;

/** Latency aggregate over a bounded rolling window. */
export interface LatencySummary {
  readonly count: number;
  readonly avgMs: number;
  readonly minMs?: number;
  readonly maxMs?: number;
}

/** One endpoint row: health facts plus request volume. */
export interface DiagnosticsEndpoint {
  readonly baseURL: string;
  readonly healthy: boolean;
  readonly inBackoff: boolean;
  readonly backoffUntilMs?: number;
  readonly consecutiveFailures?: number;
  readonly requests: number;
}

/** Structured model/capability facts (configured wins over discovered). */
export interface DiagnosticsModel {
  readonly id: string;
  readonly contextWindow?: number;
  readonly supportsTools?: boolean;
  readonly supportsReasoning?: boolean;
  /** `configured` (authoritative overrides) or `discovered` (cached #10 facts). */
  readonly source: 'configured' | 'discovered';
}

/** Request counters over the store's lifetime (bounded aggregates). */
export interface RequestSummary {
  readonly total: number;
  readonly success: number;
  readonly failure: number;
  readonly timeout: number;
  readonly aborted: number;
  readonly retries: number;
  readonly fallbacks: number;
  readonly toolCalls: number;
  readonly requestsWithTools: number;
  readonly requestsWithReasoning: number;
  readonly reasoningTokensTotal: number;
  /** Requests per minute over the bounded rate window (extrapolated). */
  readonly requestsPerMinute: number;
  readonly reasoningByEffort: Record<string, number>;
  /** Rolling reasoning-budget summary (bounded window). */
  readonly reasoningBudgetTokens: LatencySummary;
  readonly byFailureCode: Record<string, number>;
  readonly byFinishReason: Record<string, number>;
  readonly byEndpoint: Record<string, number>;
}

/** A recent routing decision (issue #9), bounded. */
export interface RecentRouting {
  readonly at: number;
  readonly candidates: readonly string[];
  readonly rationale: readonly string[];
}

/** A recent terminal failure, bounded. */
export interface RecentFailure {
  readonly at: number;
  readonly endpoint: string;
  readonly failureCode: string;
}

/** The machine-readable diagnostics snapshot. */
export interface DiagnosticsSnapshot {
  readonly at: number;
  readonly uptimeMs: number;
  readonly endpoints: readonly DiagnosticsEndpoint[];
  /** Structured model/capability facts (configured authoritative, #10 cached). */
  readonly models: readonly DiagnosticsModel[];
  readonly requests: RequestSummary;
  readonly latency: {
    readonly ttftMs: LatencySummary;
    readonly totalMs: LatencySummary;
  };
  readonly recentRouting: readonly RecentRouting[];
  readonly recentFailures: readonly RecentFailure[];
}

function latencySummary(samples: readonly number[]): LatencySummary {
  if (samples.length === 0) return { count: 0, avgMs: 0 };
  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = 0;
  for (const sample of samples) {
    sum += sample;
    if (sample < min) min = sample;
    if (sample > max) max = sample;
  }
  return { count: samples.length, avgMs: Math.round(sum / samples.length), minMs: min, maxMs: max };
}

/** Push into a bounded rolling window (drop oldest beyond the cap). */
function pushBounded(window: number[], value: number, cap: number): void {
  window.push(value);
  if (window.length > cap) window.shift();
}

/** Requests per minute over the last {@link RATE_WINDOW_MS}, extrapolated. */
export function requestsPerMinute(startTimes: readonly number[], now: number): number {
  const cutoff = now - RATE_WINDOW_MS;
  const recent = startTimes.filter((timestamp) => timestamp >= cutoff);
  if (recent.length === 0) return 0;
  const spanMs = Math.max(now - Math.min(...recent), 1);
  const minutes = spanMs / RATE_WINDOW_MS;
  return Math.round((recent.length / minutes) * 10) / 10;
}

/**
 * Bounded diagnostics store: a telemetry sink that aggregates request
 * outcomes without retaining any content. Also accepts the endpoint pool and
 * model facts at snapshot time so health/backoff state is reflected live.
 */
export class DiagnosticsStore implements TelemetrySink {
  private readonly requests = {
    total: 0,
    success: 0,
    failure: 0,
    timeout: 0,
    aborted: 0,
    retries: 0,
    fallbacks: 0,
    toolCalls: 0,
    requestsWithTools: 0,
    requestsWithReasoning: 0,
    reasoningTokensTotal: 0,
    reasoningByEffort: {} as Record<string, number>,
    byFailureCode: {} as Record<string, number>,
    byFinishReason: {} as Record<string, number>,
    byEndpoint: {} as Record<string, number>,
  };
  private readonly ttftWindow: number[] = [];
  private readonly totalWindow: number[] = [];
  private readonly startTimes: number[] = [];
  private readonly budgetWindow: number[] = [];
  private readonly routingWindow: RecentRouting[] = [];
  private readonly failureWindow: RecentFailure[] = [];
  private readonly startedAt = Date.now();

  emit(event: TelemetryEvent): void {
    switch (event.type) {
      case 'started':
        this.requests.total += 1;
        if (event.context.toolsAvailable) this.requests.requestsWithTools += 1;
        pushBounded(this.startTimes, event.at, MAX_RATE_SAMPLES);
        break;
      case 'reasoning':
        if (event.decision.enabled) this.requests.requestsWithReasoning += 1;
        if (event.decision.enabled && event.decision.effort !== undefined) {
          this.requests.reasoningByEffort[event.decision.effort] = (this.requests.reasoningByEffort[event.decision.effort] ?? 0) + 1;
        }
        if (event.decision.budgetTokens !== undefined) {
          pushBounded(this.budgetWindow, event.decision.budgetTokens, MAX_REASONING_BUDGET_WINDOW);
        }
        break;
      case 'attempt':
        if (event.attempt.outcome === 'retry') this.requests.retries += 1;
        else if (event.attempt.outcome === 'fallback') this.requests.fallbacks += 1;
        break;
      case 'routing':
        this.routingWindow.push({ at: event.at, candidates: [...event.decision.candidates], rationale: [...event.decision.rationale] });
        if (this.routingWindow.length > MAX_RECENT_ROUTING) this.routingWindow.shift();
        break;
      case 'finished':
        this.finish(event);
        break;
    }
  }

  private finish(event: Extract<TelemetryEvent, { type: 'finished' }>): void {
    const { outcome } = event;
    const endpoint = outcome.endpoint;
    this.requests.byEndpoint[endpoint] = (this.requests.byEndpoint[endpoint] ?? 0) + 1;
    const reasonKind = outcome.finishReason?.kind;
    if (reasonKind !== undefined) {
      this.requests.byFinishReason[reasonKind] = (this.requests.byFinishReason[reasonKind] ?? 0) + 1;
    }
    if (outcome.failureCode !== undefined) {
      this.requests.byFailureCode[outcome.failureCode] = (this.requests.byFailureCode[outcome.failureCode] ?? 0) + 1;
      if (outcome.failureCode === 'TIMEOUT') this.requests.timeout += 1;
      else if (outcome.failureCode === 'ABORTED') this.requests.aborted += 1;
      else this.requests.failure += 1;
      this.failureWindow.push({ at: event.at, endpoint, failureCode: outcome.failureCode });
      if (this.failureWindow.length > MAX_RECENT_FAILURES) this.failureWindow.shift();
    } else {
      this.requests.success += 1;
    }
    this.requests.toolCalls += outcome.toolCallCount ?? 0;
    const reasoningTokens = outcome.usage?.reasoningTokens;
    if (reasoningTokens !== undefined) this.requests.reasoningTokensTotal += reasoningTokens;
    if (outcome.ttftMs !== undefined) pushBounded(this.ttftWindow, outcome.ttftMs, MAX_LATENCY_WINDOW);
    pushBounded(this.totalWindow, outcome.totalMs, MAX_LATENCY_WINDOW);
  }

  /** Machine-readable snapshot, reflecting the current pool/model state. */
  snapshot(
    pool?: EndpointPool,
    endpointUrls: readonly string[] = [],
    models: readonly DiagnosticsModel[] = [],
    now: number = Date.now(),
  ): DiagnosticsSnapshot {
    const endpoints: DiagnosticsEndpoint[] = endpointUrls.map((baseURL) => {
      const health = pool?.healthOf(baseURL) ?? { healthy: true, inBackoff: false };
      return {
        baseURL,
        ...health,
        requests: this.requests.byEndpoint[baseURL] ?? 0,
      };
    });
    return {
      at: now,
      uptimeMs: now - this.startedAt,
      endpoints,
      models: models.map((entry) => ({ ...entry })),
      requests: {
        ...this.requests,
        requestsPerMinute: requestsPerMinute(this.startTimes, now),
        reasoningByEffort: { ...this.requests.reasoningByEffort },
        reasoningBudgetTokens: latencySummary(this.budgetWindow),
        byFailureCode: { ...this.requests.byFailureCode },
        byFinishReason: { ...this.requests.byFinishReason },
        byEndpoint: { ...this.requests.byEndpoint },
      },
      latency: {
        ttftMs: latencySummary(this.ttftWindow),
        totalMs: latencySummary(this.totalWindow),
      },
      recentRouting: [...this.routingWindow],
      recentFailures: [...this.failureWindow],
    };
  }

  /** Human-readable rendering for local operations/debugging. */
  render(pool?: EndpointPool, endpointUrls: readonly string[] = [], models: readonly DiagnosticsModel[] = []): string {
    const snapshot = this.snapshot(pool, endpointUrls, models);
    return renderDiagnostics(snapshot);
  }
}

/** Render a snapshot as a compact multi-line text block. */
export function renderDiagnostics(snapshot: DiagnosticsSnapshot): string {
  const lines: string[] = [];
  const { requests, latency } = snapshot;
  lines.push(`llm-llamacpp diagnostics (uptime ${Math.round(snapshot.uptimeMs / 1000)}s)`);
  if (snapshot.endpoints.length > 0) {
    lines.push('endpoints:');
    for (const endpoint of snapshot.endpoints) {
      const health = endpoint.inBackoff
        ? `BACKOFF (${endpoint.consecutiveFailures} consecutive, until ${new Date(endpoint.backoffUntilMs ?? 0).toISOString()})`
        : endpoint.healthy ? 'healthy' : 'down';
      lines.push(`  ${endpoint.baseURL} — ${health} — ${endpoint.requests} requests`);
    }
  }
  if (snapshot.models.length > 0) {
    lines.push('models:');
    for (const model of snapshot.models) {
      const facts = [
        model.contextWindow !== undefined ? `ctx ${model.contextWindow}` : undefined,
        model.supportsTools !== undefined ? `tools ${model.supportsTools}` : undefined,
        model.supportsReasoning !== undefined ? `reasoning ${model.supportsReasoning}` : undefined,
      ].filter((f): f is string => f !== undefined);
      lines.push(`  ${model.id} [${model.source}]${facts.length > 0 ? ` (${facts.join(', ')})` : ''}`);
    }
  }
  lines.push(
    `requests: ${requests.total} total (${requests.success} ok, ${requests.failure} failed, ${requests.timeout} timeout, ${requests.aborted} aborted) @ ${requests.requestsPerMinute}/min`,
  );
  lines.push(`retries: ${requests.retries}, fallbacks: ${requests.fallbacks}`);
  lines.push(
    `latency: ttft ${latency.ttftMs.count > 0 ? `avg ${latency.ttftMs.avgMs}ms (min ${latency.ttftMs.minMs}, max ${latency.ttftMs.maxMs})` : 'n/a'} | total avg ${latency.totalMs.avgMs}ms`,
  );
  if (requests.requestsWithReasoning > 0) {
    const avgReasoning = requests.requestsWithReasoning > 0
      ? Math.round(requests.reasoningTokensTotal / requests.requestsWithReasoning)
      : 0;
    const efforts = Object.entries(requests.reasoningByEffort);
    const budget = requests.reasoningBudgetTokens;
    lines.push(
      `reasoning: ${requests.requestsWithReasoning} requests, ${requests.reasoningTokensTotal} tokens total (avg ${avgReasoning}/req)` +
        `${efforts.length > 0 ? `; efforts ${efforts.map(([e, n]) => `${e}=${n}`).join(', ')}` : ''}` +
        (budget.count > 0 ? `; budget avg ${budget.avgMs} (min ${budget.minMs}, max ${budget.maxMs})` : ''),
    );
  }
  if (requests.toolCalls > 0) lines.push(`tools: ${requests.toolCalls} tool calls across ${requests.requestsWithTools} requests`);
  const failureCodes = Object.entries(requests.byFailureCode);
  if (failureCodes.length > 0) {
    lines.push(`failure codes: ${failureCodes.map(([code, n]) => `${code}=${n}`).join(', ')}`);
  }
  if (snapshot.recentFailures.length > 0) {
    lines.push(`recent failures (last ${snapshot.recentFailures.length}):`);
    for (const failure of snapshot.recentFailures.slice(-5)) {
      lines.push(`  ${new Date(failure.at).toISOString()} ${failure.endpoint} — ${failure.failureCode}`);
    }
  }
  return lines.join('\n');
}

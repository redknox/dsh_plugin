/**
 * Capability-aware endpoint routing (issue #9): choose eligible llama.cpp
 * endpoints from request/model capabilities BEFORE reliability fallback
 * begins.
 *
 * A plain endpoint list with no capability metadata routes exactly as #7 does
 * today (all endpoints eligible, configuration order). With metadata, requests
 * are never sent to an endpoint known to be incompatible with the requested
 * model/capabilities, ordering among equally eligible candidates stays
 * deterministic, and the decision + rationale is exposed through the #8
 * observability seam. Reliability still owns transient failures after routing
 * selects the eligible candidates.
 *
 * @module llm-llamacpp/routing
 */
import { LlmError } from '@deepseek-ai/dsh-llm';

/** Endpoint capability facts, independent of HTTP transport. */
export interface EndpointCapabilities {
  /** Exact model ids this endpoint serves; absent/empty = any model. */
  readonly models?: string[];
  /** Maximum context window in tokens. */
  readonly contextWindow?: number;
  /** Whether tool calling is supported; absent = unknown (assume yes). */
  readonly tools?: boolean;
  /** Whether reasoning/thinking is supported; absent = unknown (assume yes). */
  readonly reasoning?: boolean;
  /** Preferred workload classes, e.g. 'chat' | 'code' | 'reasoning' | 'batch' | 'title' | 'compaction'. */
  readonly workload?: string[];
}

/** One configured endpoint with optional capability metadata. */
export interface EndpointRoutingProfile {
  readonly baseURL: string;
  readonly capabilities?: EndpointCapabilities;
}

/** Request constraints the router derives from one model request. */
export interface RoutingRequest {
  readonly model: string;
  readonly toolsAvailable: boolean;
  /** Whether the resolved policy enables thinking (non-off effort). */
  readonly reasoningEnabled: boolean;
  /** Derived workload class for preference ordering, when one applies. */
  readonly workload?: string;
  /** Estimated prompt size in tokens (context-window constraint). */
  readonly estimatedPromptTokens?: number;
}

/** The routing outcome: eligible candidates in preference order + rationale. */
export interface RoutingDecision {
  /** Eligible base URLs in deterministic order (never empty). */
  readonly candidates: readonly string[];
  /** Human-readable rationale: exclusions and the final candidate order. */
  readonly rationale: readonly string[];
}

/** Derive the request workload class from purpose and reasoning state. */
export function deriveWorkload(
  purpose: 'compaction' | 'session-title' | undefined,
  reasoningEnabled: boolean,
): string | undefined {
  if (purpose === 'session-title') return 'title';
  if (purpose === 'compaction') return 'compaction';
  return reasoningEnabled ? 'reasoning' : 'chat';
}

/** Why one endpoint cannot serve the request; empty = eligible. */
function exclusions(request: RoutingRequest, capabilities: EndpointCapabilities | undefined): string[] {
  if (capabilities === undefined) return [];
  const out: string[] = [];
  if (capabilities.models !== undefined && capabilities.models.length > 0 && !capabilities.models.includes(request.model)) {
    out.push(`model "${request.model}" not served`);
  }
  if (
    request.estimatedPromptTokens !== undefined &&
    capabilities.contextWindow !== undefined &&
    request.estimatedPromptTokens > capabilities.contextWindow
  ) {
    out.push(`prompt ≈${request.estimatedPromptTokens}t exceeds context window ${capabilities.contextWindow}`);
  }
  if (request.toolsAvailable && capabilities.tools === false) {
    out.push('tools requested but unsupported');
  }
  if (request.reasoningEnabled && capabilities.reasoning === false) {
    out.push('reasoning requested but unsupported');
  }
  return out;
}

/** Workload-preference rank for stable ordering (higher sorts first). */
function workloadRank(profile: EndpointRoutingProfile, request: RoutingRequest): number {
  if (request.workload === undefined) return 0;
  if (profile.capabilities?.workload === undefined) return 0;
  return profile.capabilities.workload.includes(request.workload) ? 1 : 0;
}

/**
 * Route one request over the configured endpoint profiles. Deterministic for
 * the same inputs: eligibility filters first (exact model compatibility,
 * context window, tool/reasoning requirements), then candidates are ordered
 * by workload preference with stable configuration order among equals. Throws
 * `NO_ELIGIBLE_ENDPOINT` when no configured endpoint can satisfy mandatory
 * capabilities.
 */
export function routeEndpoints(
  request: RoutingRequest,
  profiles: readonly EndpointRoutingProfile[],
): RoutingDecision {
  const rationale: string[] = [];
  const eligible: EndpointRoutingProfile[] = [];
  for (const profile of profiles) {
    const problems = exclusions(request, profile.capabilities);
    if (problems.length === 0) {
      eligible.push(profile);
    } else {
      rationale.push(`${profile.baseURL}: excluded (${problems.join('; ')})`);
    }
  }
  if (eligible.length === 0) {
    throw new LlmError(
      `llm-llamacpp: no endpoint satisfies mandatory capabilities for model "${request.model}" ` +
        `(tools=${request.toolsAvailable}, reasoning=${request.reasoningEnabled})`,
      'NO_ELIGIBLE_ENDPOINT',
    );
  }
  // Stable sort: workload-preferring endpoints first, configuration order
  // preserved among equals (deterministic).
  const ordered = [...eligible].sort((a, b) => workloadRank(b, request) - workloadRank(a, request));
  const candidates = ordered.map((profile) => profile.baseURL);
  rationale.push(`candidates: ${candidates.join(' -> ')}`);
  return { candidates, rationale };
}

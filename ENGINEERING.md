# Engineering Principles

This document is normative for both human and AI contributors to `llm-llamacpp`.

When implementation convenience conflicts with these principles, prefer the principles unless an issue or design note explicitly documents why an exception is necessary.

The goal is not architectural purity. The goal is to preserve clear boundaries, evidence-backed behavior, maintainability, and a predictable user experience as the project evolves.

## 1. llama.cpp is the provider boundary

`llm-llamacpp` is a DeepSeek Harness provider for llama.cpp.

Qwen is the first-class / best-validated model family, but it is not the provider identity. Generic transport, streaming, tools, discovery, routing, reliability, observability, and diagnostics should remain llama.cpp-oriented.

Model-family-specific behavior belongs behind explicit compatibility seams.

Do not silently turn a Qwen workaround into a generic provider contract.

## 2. Keep a stable domain model in front of volatile wire APIs

Treat llama.cpp request/response shapes as an external protocol that may evolve.

Prefer this flow:

```text
Harness contracts
    ↓
provider domain semantics
    ↓
protocol / serialization / translation / compatibility
    ↓
llama.cpp wire API
```

New or renamed llama.cpp fields should normally be isolated at the protocol edge rather than propagated through the entire runtime.

Examples of wire-level concerns include:

- request/response DTO changes;
- SSE chunk shapes;
- reasoning field names;
- chat-template kwargs;
- discovery metadata;
- usage/finish details;
- server-specific error shapes.

Only promote a wire concept into the provider domain when it represents a stable user- or Harness-visible capability.

## 3. Explicit beats heuristic; unknown != false

Use this precedence whenever possible:

```text
explicit configuration
→ server/provider capability metadata
→ compatibility profile
→ model-name heuristic as a last resort
```

Do not infer capability merely because a model name resembles a known family.

Do not collapse `unknown`, `unsupported`, and `false` into the same value when the distinction is observable or affects behavior.

If support cannot be determined safely, preserve the unknown state or require explicit configuration.

## 4. Prefer capability-driven behavior over model-family branching

If llama.cpp can report a capability reliably, prefer that signal over maintaining a growing model-family knowledge base.

Compatibility profiles should be small, explicit, and limited to behavior the provider/server contract cannot express reliably.

Avoid patterns such as:

```text
if Qwen ...
if Llama ...
if Gemma ...
if DeepSeek ...
```

unless there is verified family-specific behavior that cannot be represented through generic capability metadata.

## 5. Use Harness public contracts only

Integrate through public DeepSeek Harness and Cordis services, events, adapters, configuration seams, and lifecycle APIs.

Do not depend on Agent Loop internals, private orchestration classes, internal inboxes, or other implementation details that are not part of the public extension contract.

Prefer:

- LLM adapters;
- tools;
- public services;
- typed events/hooks;
- Cordis Context/Scope lifecycle;
- settings and credentials services.

## 6. Policies are replaceable seams

Reasoning, routing, retry/fallback, feedback, and similar decisions should remain policy seams rather than becoming hard-coded branches inside the adapter.

The adapter should coordinate behavior; policies should own decisions that may change independently.

Prefer narrow interfaces and explicit inputs/outputs over global mutable state.

Do not introduce a new abstraction merely because one may be useful someday. Add a seam when there is a concrete behavior or replacement need.

## 7. Separate transport, domain, and configuration concerns

Keep external DTOs, provider semantics, and user configuration conceptually separate.

A useful mental model is:

```text
external API DTO
→ anti-corruption / translation layer
→ provider domain model
→ policy / service layer
```

Configuration should resolve through one explicit validation/defaulting step before runtime use.

Avoid allowing raw settings objects or wire DTOs to leak throughout the codebase.

## 8. Configuration UX is part of the product contract

User-facing configuration should optimize for comprehension, not expose internal implementation detail.

### 8.1 Basic settings first

Common settings should be easy to find immediately.

Advanced settings should be grouped by user intent and may default to collapsed. Prefer several semantic groups over one large catch-all `Advanced` section.

Examples of useful groups include:

- Basic;
- Reasoning;
- Reliability;
- Endpoints;
- Discovery & Diagnostics.

### 8.2 Every user-facing field should explain its meaning

Every configuration field exposed in the UI should provide a concise user-facing description where the schema/UI system supports it.

Descriptions should explain the effect a user will observe, not how the implementation works internally.

Prefer:

> How long a streaming response may stay idle before it is treated as timed out.

Avoid:

> Controls the AbortController watchdog used by Promise.race().

Inline descriptions should normally fit in one short line. Longer explanations may use a tooltip, info affordance, or linked documentation.

### 8.3 Keep the generic editor generic

Presentation hints should be expressed through generic schema metadata where possible.

Do not add provider-specific UI branches such as:

```ts
if (provider === 'llm-llamacpp') {
  // special editor behavior
}
```

A provider with no optional UI metadata must continue to render correctly using the generic fallback behavior.

### 8.4 Preserve configuration ownership

The provider schema/settings layer owns configuration semantics.

The UI should not mutate adapter internals directly or create a second configuration system.

Credentials must remain outside ordinary settings storage. Never persist raw secrets in `settings.yaml`.

Unsupported schema shapes must be surfaced explicitly; do not silently discard them.

## 9. Compatibility claims require evidence

Distinguish clearly between:

- verified compatibility;
- designed/expected generic behavior;
- unknown or unsupported behavior.

Do not claim support for a model family solely because llama.cpp can load it.

Documentation, package metadata, release notes, and diagnostics should use evidence-backed wording.

Record exact versions and environments when compatibility depends on them.

## 10. Reliability must not erase semantics

Retries, fallback, routing, timeout handling, and cancellation should preserve the semantic meaning of the request.

Do not silently drop unsupported request fields to make a request succeed.

Unsupported capabilities should fail explicitly with stable errors where appropriate.

Cancellation signals must be forwarded through network operations and long-running work.

Reliability and routing should remain separate concerns: routing decides where a request should go; reliability decides what to do when an attempt fails.

## 11. Observability should be useful, bounded, and privacy-aware

Telemetry and diagnostics should explain what the runtime decided without requiring prompt/completion logging.

Prefer structured, bounded signals such as:

- request/attempt counts;
- routing rationale;
- endpoint health/backoff;
- TTFT and total latency;
- reasoning effort/budget/token use;
- tool-call activity;
- capability/discovery source;
- recent failures.

Do not log secrets, prompt content, completion content, or tool arguments by default.

Unknown facts should remain unknown rather than being reported as false.

## 12. Tests should protect architectural boundaries, not just happy paths

Tests should cover behavior that prevents accidental coupling or silent degradation.

Important regression categories include:

- unknown/non-Qwen paths do not inherit Qwen-only parameters;
- explicit configuration overrides defaults/profiles;
- unsupported content is rejected rather than dropped;
- streaming block lifecycle and tool-call deltas remain valid;
- cancellation and timeout behavior;
- configuration fallback behavior;
- credential isolation;
- generic editor behavior with a non-llama.cpp fixture;
- packaging preserves a single Harness/Cordis runtime identity.

When a test depends on a validated model-family behavior, make that profile explicit instead of relying on an implicit default.

## 13. Review discipline

Review against observable requirements and project boundaries, not abstract architectural perfection.

Do not request a refactor solely to make the design more theoretically pure when the existing seam is clear, bounded, tested, and maintainable.

At the same time, do not accept convenience changes that silently weaken a documented boundary.

For review conclusions:

- distinguish static inspection from tests actually run;
- treat repository-recorded E2E results as recorded evidence, not as locally reproduced evidence;
- call out blockers precisely and keep non-blocking observations separate.

## 14. Release discipline

Published artifacts should preserve the same architecture as source installs.

Requirements include:

- llama.cpp remains the public provider identity;
- Qwen remains a validated compatibility family rather than package identity;
- peer framework dependencies must not create a duplicate Harness/Cordis runtime identity;
- public compatibility claims must be evidence-backed;
- prebuilt artifacts should not require an install-time toolchain when avoidable;
- external publication actions such as `npm publish` require explicit authorization.

## 15. Prefer incremental evolution

Do not turn this provider into a distributed inference scheduler, model loader, or Agent Loop fork unless a concrete requirement justifies that scope.

Prefer small, composable improvements that extend public seams:

- richer llama.cpp capability discovery;
- compatibility tracking;
- better configuration UX;
- additional observability/export surfaces;
- capability-aware routing improvements;
- real provider contract tests.

The default question for a new feature should be:

> Can this be added at the provider boundary without widening unrelated parts of the system?

If yes, keep it there.

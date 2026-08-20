# llm-llamacpp

[![npm version](https://img.shields.io/npm/v/llm-llamacpp)](https://www.npmjs.com/package/llm-llamacpp)
[![license](https://img.shields.io/npm/l/llm-llamacpp)](https://github.com/redknox/dsh_plugin/blob/main/LICENSE)

> **A production-oriented llama.cpp LLM provider for DeepSeek Harness, with
> first-class Qwen reasoning support**, streaming tool calls, adaptive
> inference policies, multi-endpoint reliability, capability-aware routing,
> model/capability discovery, observability, and diagnostics.

`llm-llamacpp` targets **[llama.cpp](https://github.com/ggml-org/llama.cpp)** as
the provider/backend, through its OpenAI-compatible
`/v1/chat/completions` endpoint — it is **not** a Qwen-only plugin. The
provider is *designed* to work with the model families a llama.cpp build
serves; **Qwen is the currently validated family**, and other families follow
the generic llama.cpp path but are not yet claimed as verified. The runtime
is llama.cpp-generic (streaming, tool calls, reliability, routing,
discovery, observability, diagnostics), while model-family behavior — above
all **Qwen's chat-template reasoning/thinking semantics** — is isolated
behind explicit compatibility profiles (`modelFamily`, see
[Reasoning](#reasoning)). Qwen is the best-validated, first-class model
family (tested end-to-end against a real Qwen3.8 server).

The plugin owns the single provider route `llamacpp-local`. It is loaded by
DeepSeek Harness as a Cordis plugin and registers itself through the public
`ctx.llm` service contract only — no agent-loop internals are touched.

> **Status.** Issues #1-#19 implemented, approved, and validated end-to-end
> against a real llama.cpp server running the Qwen3.8 family: text streaming
> with reasoning, parallel tool calls, reasoning off/on and both wire modes,
> multi-endpoint fallback on real network failures, model/capability
> discovery, diagnostics, model-family compatibility profiles, the DSH
> schema-driven generic settings editor, and the full install path —
> **published on npm as `llm-llamacpp@0.1.0`**.

## Requirements

- DeepSeek Harness with the web profile (`dsh web`); the host provides
  `@deepseek-ai/cordis`, `@deepseek-ai/dsh-llm`, `@deepseek-ai/dsh-settings`,
  and `@deepseek-ai/dsh-credentials` as peer dependencies.
- A llama.cpp server with an OpenAI-compatible endpoint
  (e.g. `llama-server -m path/to/qwen3.gguf --port 8080` — the Qwen family is
  the validated one; other families are not yet claimed as verified),
  optionally started with `--api-key <token>`.

## Installation

The package is a DSH-native installable bundle (declares `dsh.bundle`), so it
installs through the official `dsh plugin` mechanism — no manual patch edits,
no absolute paths, no `node_modules` symlinks. Pick one:

**Option 1 — npm registry (recommended; `llm-llamacpp` is published):**

```bash
dsh plugin --profile web add llm-llamacpp
```

**Option 2 — exact Git commit** (source build; pnpm ≥ 10 needs a one-time
`allowBuilds` step — see `docs/install.md`):

```bash
dsh plugin --profile web add github:redknox/dsh_plugin#<commit-sha>
```

**Option 3 — prebuilt tarball** (no build authorization):

```bash
dsh plugin --profile web add ./llm-llamacpp-0.1.0.tgz
```

All three register the bundle automatically (`dsh.profile.bundles` gains
`llm-llamacpp`), mount the plugin by package name, and share the host's
Harness/Cordis runtime identity. Verify it is registered:

```bash
curl -s -X POST http://127.0.0.1:3080/api/llm.providers \
  -H 'Content-Type: application/json' \
  -d '{"type":"client-request","rpcId":"probe","method":"llm.providers","payload":{}}'
# llamacpp-local should appear with "active": true
```

Then select **the provider/model in the web GUI's model selector** and chat.
The `model` value is passed to the wire verbatim; the plugin registers the
`llamacpp-local` provider route and the configurable-provider directory entry
`llamacpp-local@llm-llamacpp`. Edit connection details later in the web GUI's
**Settings → Models** page (or via `~/.dsh/profiles/<name>/cordis.patch.yml`).

> **Developing from source** — clone the repo, `npm install && npm run build`,
> and mount `dist/index.js` by path in the profile patch; keep a single copy
> of the shared `@deepseek-ai/*` packages so the plugin shares identity with
> the host (see `docs/install.md`).

## Configuration

### `baseURL` (server address)

The llama.cpp server's OpenAI-compatible **base** URL — scheme + host + port,
**without** `/v1` (the client appends `/v1/chat/completions`) and without a
trailing slash. Local default is `http://127.0.0.1:8080`.

### `model` (model id)

The exact model id the server accepts, listed by `GET /v1/models`. It is
passed to the wire `model` field verbatim, so changing models needs no plugin
reload.

### `apiKeyEnv` (token — never put the key in config)

`apiKeyEnv` names an **environment variable / credential reference**
(e.g. `LLAMA_API_TOKEN`), not the key itself. The plugin resolves the value
per request through the DSH credentials seam (`ctx.credentials`, whose `env`
layer covers the launching environment and whose store the web Models page
can write), falling back to a direct `process.env` read when no credentials
service is mounted. Provide the token either way:

```bash
export LLAMA_API_TOKEN=<your-token>   # in the shell that launches dsh web
```

or store it via the web GUI's Models page. A configured reference that
resolves nowhere fails clearly with `MISSING_CREDENTIAL`.

Optional keys (all documented above in the example):

| key | default | meaning |
|---|---|---|
| `apiKeyHeader` | `authorization` | header carrying the key; `authorization` sends `Bearer <key>`, anything else sends the raw key |
| `streamIdleTimeoutMs` | `300000` | idle timeout per outstanding provider read |
| `requestTimeoutMs` | — | hard per-attempt deadline, regardless of activity |
| `endpoints` | `[baseURL]` | ordered fallback list (replaces `baseURL`); first entry is primary |
| `retryPolicy` | bounded normal | retryable codes, max retries, exponential backoff with jitter |
| `reasoning` | medium preset | semantic presets, expert overrides, wire mode, adaptive budget |

Configuration is validated at load: an invalid `baseURL`, a non-http(s)
endpoint, or an empty `model`/`providerName` fails clearly. When the harness
settings service is mounted, the same schema drives an `llm-llamacpp` settings
section that can override any field without a restart.

### Settings-page presentation (issue #19)

The DSH Models page renders this plugin through the generic schema-driven
provider editor (upstream `GenericSchemaEditor`). The plugin supplies
**advisory presentation metadata** through its schema — semantic groups that
start collapsed, human-readable labels, and one-line field help — so basic
settings stay immediately visible and advanced sections (Reasoning,
Reliability, Endpoints, Discovery, Diagnostics, Telemetry) are collapsible.
The mechanism is provider-generic; any configurable provider can declare the
same hints and no provider family is hard-coded in the editor.

## Capability-aware endpoint routing (issue #9)

Evolve ordered fallback into capability-aware routing: an eligible endpoint
set is chosen from request/model capabilities **before** reliability fallback
begins. A plain endpoint list with no capability metadata routes exactly as
#7 does today (all eligible, configuration order).

`endpoints` accepts either plain URLs or URL + capability objects:

```yaml
endpoints:
  - http://10.0.0.1:8080
  - url: http://10.0.0.2:8080
    capabilities:
      models: [qwen3]            # exact model ids served; absent = any
      contextWindow: 32768       # max context window in tokens
      tools: true                # false = no tool calling
      reasoning: true            # false = no thinking support
      workload: [chat, code]     # preferred workload classes
```

Eligibility rules (deterministic): exact model compatibility first, then
context-window fit (estimated prompt size), then tool/reasoning requirements
(absent capability = unknown = assumed supported). Equally eligible candidates
keep configuration order, with matching `workload` classes sorting first
(stable). When **no** configured endpoint satisfies mandatory capabilities the
request fails explicitly with `NO_ELIGIBLE_ENDPOINT`. Reliability (health,
backoff, transient retry/fallback from #7) still owns failures after routing
selects the eligible candidates.

The routing decision (candidates + rationale) is emitted through the #8
telemetry seam as a `routing` event:

| event | fields | units / cardinality |
|---|---|---|
| `routing` | `decision: { candidates, rationale }` | once per request, after capability selection |

## Model and capability discovery (issue #10)

Optional: discover llama.cpp model and server capabilities instead of
requiring every capability to be configured manually. Off by default — a
plain single-server deployment is unchanged; discovery failure never breaks a
valid configured deployment.

```yaml
discovery:
  enabled: true
  ttlMs: 300000    # optional bounded cache TTL (default 300s)
  timeoutMs: 5000  # optional per-probe timeout (default 5s)
```

What is probed and how:

- `/health` — connectivity signal (`healthy: true` on 2xx; a missing or
  unreachable `/health` reports `healthy: false` without affecting model
  discovery or a configured deployment).
- `/v1/models` — exact model ids (`data[].id`, with the llama.cpp/ollama-style
  `models[]` list also accepted) and context-window metadata from
  `data[].meta.n_ctx` (falling back to `/props.n_ctx` for a single-model
  slot).
- `/props` — loaded model alias (used when the model list is unavailable) and
  the slot context window.
- Tool/reasoning support is only set when the server **explicitly** states it
  (markers in a capabilities list); absence is "unknown", which routing treats
  as assumed supported — discovery never causes a false negative that routes a
  request away from a capable endpoint.

Precedence: **user-configured endpoint capabilities (#9) win per field**;
discovered values fill the gaps. `listModels()` advertises the discovered
model ids when discovery is enabled (falling back to the configured model when
nothing is discovered); `resolveModel()` surfaces the merged context window.
Per-request routing uses **freshly cached** discovered facts (non-blocking —
routing never stalls on a probe); the cache refreshes on metadata queries
(`listModels`/`resolveModel`) with a bounded TTL and honors cancellation.

### Adaptive reasoning feedback loop (issue #11)

Optional: layer recent provider outcomes onto the budget decision. Enable
under `reasoning.feedback`:

```yaml
reasoning:
  preset: medium
  adaptive:
    enabled: true      # optional; feedback also layers over a static base
  feedback:
    enabled: true
```

Each completed request records a bounded provider-observable outcome (outcome
class, failure code, retry/fallback use, reasoning token consumption, latency,
finish reason — derived from the #8 telemetry outcome; tool-call retries
execute outside the provider and are not observed). The history keeps a fixed
window (default 20; oldest entries drop out, so stale failures decay and
cannot permanently bias) and can be reset.

Adjustment rules are deterministic and bounded:

- heavy timeouts/aborts/failures → **reduce** the budget (thinking overruns);
- reasoning consumption near the budget cap → **increase** it;
- empty history → the request behaves exactly like the base static/adaptive
  policy;
- explicit per-request effort and expert `budgetTokens` always win (no
  feedback adjustment);
- the adjusted budget is clamped to the hard adaptive min/max safety bounds
  from #6.

The chosen budget and the feedback rationale ride the `reason` field of the
`reasoning` telemetry event, so decisions stay inspectable through #8
observability.

## Production diagnostics (issue #12)

A bounded, machine-readable diagnostics surface sourced from the #8 telemetry
events and the #7 endpoint health state. `diagnostics.enabled` defaults to
`true`; it is a passive, content-free consumer — a plain single-endpoint
deployment neither requires it nor changes behavior.

The plugin provides the context service `llm-llamacpp/diagnostics` with:

- `snapshot()` — machine-readable snapshot (framework-independent API):
  - `endpoints` — configured endpoints with health/backoff state (from the
    reliability pool) and request volume;
  - `models` — **structured model/capability facts** (id, context window,
    tool/reasoning support, `source: configured | discovered`) with
    configured overrides authoritative over #10 cached discovery;
  - `requests` — totals by outcome (success/failure/timeout/aborted),
    retries and fallbacks, tool-call activity, **request rate (requests/min
    over a bounded 60s window, extrapolated)**, **reasoning aggregates**
    (requests with reasoning, effort distribution, rolling budget window,
    token totals), and breakdowns by endpoint / failure code / finish reason;
  - `latency` — bounded rolling windows (last 200 samples) for TTFT and total
    latency (avg/min/max);
  - `recentRouting` / `recentFailures` — bounded (20 each) recent decisions
    and failures.
- `render()` — human-readable block for local operations/debugging.

```bash
npm run build
LLAMACPP_BASE_URL=http://10.60.84.212:8040 \
LLAMACPP_MODEL=/models/Qwen3.8-27B-Q8_0.gguf \
LLAMA_API_TOKEN=<token> node examples/diagnostics.mjs
```

Troubleshooting flow: check endpoint rows for `BACKOFF` (consecutive failures
and retry-until timestamps), then request counters and failure-code breakdown,
then latency windows and the recent failure tail — without reading raw
application logs. Retention is bounded (rolling windows and capped lists) and
no raw prompt/completion/tool-argument content is ever retained.

## Observability (issue #8)

Optional structured request telemetry, emitted through a narrow sink (no
Harness agent-loop coupling). `telemetry.enabled` defaults to `true` and emits
one JSON line per event at `debug` level; set `false` to disable emission
without changing provider behavior.

One request is traced from **adapter entry** (the lifecycle starts before any
work, so reasoning/serialization/credential failures converge into the same
trace with true end-to-end latency) through endpoint selection to the terminal
result via these event kinds (each carries the stable trace `requestId`):

| event | fields | units / cardinality |
|---|---|---|
| `started` | `context: { model, purpose?, toolsAvailable }` | once per request, at adapter entry |
| `reasoning` | `decision: { enabled, effort?, budgetTokens?, reason? }` | once, after policy resolution |
| `attempt` | `attempt: { attempt, baseURL, outcome: selected\|retry\|fallback, failureCode? }` | once per reliability attempt |
| `finished` | `outcome: { endpoint, retryCount, fallbackCount, ttftMs?, totalMs, completionMs?, streamChunkCount, finishReason?, usage?, toolCallCount?, failureCode? }` | once per request (success or failure) |

Metric semantics:

- `ttftMs` — time to first user-visible token (text/reasoning/tool delta) from
  adapter entry; `totalMs` — end-to-end latency; `completionMs` — derived
  (`totalMs - ttftMs`).
- `streamChunkCount` — Harness `StreamChunk` count; `toolCallCount` — streamed
  tool-call blocks (tools are never executed by the provider).
- `usage` — provider token accounting (input/output/reasoning/cache-read) when
  llama.cpp exposes it; `finishReason` — terminal finish kind.
- `retryCount` / `fallbackCount` — reliability layer outcomes; `failureCode` —
  terminal failure code (`ABORTED`, `TIMEOUT`, `TRANSPORT`, …).

Privacy rules (enforced structurally and tested): events carry field names and
counts only — never prompt content, tool arguments, completions, API keys, or
any request payload. Cardinality is bounded per request (≤ 1 + attempts + 1
events); nothing is retained by the plugin itself (issue #12 adds a bounded
in-memory diagnostic snapshot consuming this same surface).

## Reasoning

Optional semantic reasoning/thinking controls. The provider exposes stable
semantic levels (`off`, `low`, `medium`, `xhigh`) through
`resolveModel`/`ctx.llm.resolveModelInfo`, selectable as
`GenerateOptions.reasoningEffort`. Effort (semantic) and `budgetTokens`
(runtime thinking budget) are separate concepts; the built-in preset table
maps each level, and the `reasoning.expert` config overrides individual fields
without rewriting the table. Per-request effort wins over the configured
preset; `session-title` calls always disable thinking.

**Model-family awareness (issue #18).** Whether thinking wire fields are sent
at all is gated by the model-family compatibility profile
(`modelFamily: 'auto' | 'qwen'`, default `'auto'` → `unknown`). A family whose
template-kwargs support is unknown (the default) gets **no** reasoning wire
fields (`wire: 'none'`) — nothing Qwen-oriented is ever sent silently; only
explicit configuration, capability metadata, or an explicit family profile
(Qwen) opts into them. Qwen is the first-class, best-validated reasoning
family: select `modelFamily: 'qwen'` (and/or an explicit `reasoning.wire`) for
Qwen chat-template semantics.

The resolved policy is translated to llama.cpp request fields only in the
request builder, and the fields are version-dependent:

- `wire: chat-template-kwargs`: `chat_template_kwargs = { enable_thinking,
  preserve_thinking? }` — **Qwen chat-template kwargs** honored by llama.cpp
  builds with the per-request template-kwargs hook (llama.cpp PR #13196);
  this is the default for the Qwen profile and any explicit opt-in. The
  runtime thinking budget is a separate inference control and is sent as the
  top-level `thinking_budget_tokens` per-request field.
- `wire: reasoning-fields`: top-level `reasoning_effort` (including
  `"none"`) and `thinking_budget_tokens` — llama.cpp-native fields, usable by
  any family on explicit configuration. Requires newer llama.cpp builds with
  native per-request reasoning support (PRs #22336 / #23116 / #26045).
  `preserve_thinking` is a chat-template kwarg that llama.cpp merges
  independently of the native fields, so it rides alongside them in either
  wire mode.
- `wire: none` (default for unknown families): no reasoning wire fields are
  sent at all.

Semantics of the expert knobs:

- `preserveThinking` is a **request/template behavior** (Qwen
  `chat_template_kwargs.preserve_thinking` — keep historical thinking in the
  prompt), not an output switch.
- `emitThinking` (default true) is the **output-visibility** knob: `false`
  consumes thinking deltas without emitting `reasoning` blocks to the Harness
  stream.

### Adaptive reasoning budget (issue #6)

Optional: choose the reasoning budget from request context instead of using
only static presets. The policy layer is a small provider-domain seam
(`ReasoningPolicy`) with static preset resolution as the default implementation
and an optional adaptive decorator; the adapter depends only on that seam.
Enable adaptive mode under `reasoning.adaptive`:

```yaml
reasoning:
  preset: medium
  adaptive:
    enabled: true
    defaultBudgetTokens: 4096  # optional; overrides the preset base before adjustment
    minBudgetTokens: 512       # optional hard lower bound
    maxBudgetTokens: 65536     # optional hard upper bound
    hints: [deep]              # optional task/profile hints: short | deep | precise
```

Policy inputs: message count and estimated prompt size (adapter-side ~4
chars/token approximation), whether tools are offered, whether the turn
follows a tool result, and the configured hints. The adjustment is a pure,
deterministic function of these inputs (tested), clamped to the hard bounds.

**Precedence (explicit and tested):**

```text
explicit per-request effort
        ↓
selected preset (with expert overrides)
        ↓
adaptive configured default/base budget (only when set, and only when the
  budget was not fixed by an explicit expert budgetTokens)
        ↓
adaptive context adjustment
        ↓
provider defaults / safety bounds (min/max clamp)
```

`defaultBudgetTokens` must lie within the configured (or default) min/max
bounds and is validated at load. The selected effort and budget are emitted as
a `debug` log line per request (`llm-llamacpp reasoning decision: …`), so the
choice is always inspectable. Adaptive mode can be enabled/disabled without
touching adapter code; with it off, static preset resolution from #4 behaves
exactly as before.

## Tools

Harness tool schemas (`GenerateOptions.tools`) are sent to llama.cpp as
OpenAI-compatible `tools`, and streamed `tool_calls` deltas (fragmented ids,
function names, and JSON argument fragments, multiple calls per response,
mixed text+tool output) are reconstructed into Harness tool-call blocks.
Malformed or empty argument JSON fails the stream with
`INVALID_TOOL_ARGUMENTS`; a completed tool call that never received a final
non-empty id or function name fails with `INCOMPLETE_TOOL_CALL` instead of
emitting an unusable empty-branded call. Tool execution stays with Harness
`ctx.tools`; this provider only translates the protocol.

### End-to-end example

With a running llama.cpp server and a tool-capable model (Qwen validated):

```bash
npm run build
node examples/tool-call.mjs                       # defaults to http://127.0.0.1:8080, model qwen3
LLAMACPP_BASE_URL=http://127.0.0.1:8081 node examples/tool-call.mjs
```

The script streams a turn with `get_time` / `echo` tool schemas, executes the
model's tool calls locally (standing in for `ctx.tools`), feeds the results
back as `role: tool` messages, and streams a second turn.

## Reliability (issue #7)

Optional reliability layer for production/self-hosted deployments with
multiple llama.cpp servers or transient failures. Kept separate from the
adapter's translation logic; a single-server local deployment is unchanged.

- **Ordered fallback**: `endpoints` lists candidate servers in fallback order
  (first is primary). A failed primary falls back to the next candidate before
  response streaming has begun; after that, candidates cycle.
- **Retry policy**: only configured retryable codes are retried (default
  `RATE_LIMIT`, `SERVER`, `TIMEOUT`, `TRANSPORT`, `EMPTY_RESPONSE`); `always`
  mode retries every failure. Backoff is bounded exponential with symmetric
  jitter.
- **Cancellation**: an explicitly aborted request never retries or falls back.
- **Never after output**: once user-visible streamed output has begun, any
  later failure is fatal (no retry/fallback), unless behavior is explicitly
  safe and documented.
- **Health state/backoff**: repeatedly failing endpoints accrue exponential
  backoff and are skipped as candidates until they recover; any success resets
  them. State persists across requests per adapter instance.
- **Timeouts**: `streamIdleTimeoutMs` (per-read idle, re-armed on activity)
  and `requestTimeoutMs` (hard per-attempt deadline, regardless of activity).
- **Structured logs**: failures log endpoint/model/attempt/code/backoff at
  `warn` (`llm-llamacpp: endpoint … failed for model …`).
- The provider-owned retry policy is registered with the harness
  (`ctx.llm.providerRetryPolicy`) and re-registered in place when it changes,
  so the built-in `dsh-llm-retry` step-level recovery uses the same policy.
- **Composition with step-level retry**: `dsh-llm-retry` executes the policy at
  the agent-turn boundary (after a step finishes with a retryable error), while
  this plugin retries/falls back internally before streaming begins. The two
  layers compose intentionally at different boundaries, but finite budgets
  stack: with both engaged, the effective wire-request count per step is up to
  `(client maxRetries + 1) × (harness maxRetries + 1)`.

## Development

```bash
npm install        # installs dependencies (use a writable cache if ~/.npm is root-owned)
npm test           # vitest unit/integration tests
npm run typecheck  # tsc --noEmit
npm run build      # esbuild bundle -> dist/index.js, tsc declarations -> dist/types
```

## Layout

```text
src/
├── index.ts        # Cordis plugin entrypoint (registration lifecycle)
├── adapter.ts      # Harness LlmAdapter implementation
├── client.ts       # llama.cpp HTTP/SSE transport client
├── serialize.ts    # GenerateOptions -> llama.cpp wire request
├── translate.ts    # llama.cpp wire chunks -> Harness StreamChunks
├── reasoning.ts    # semantic reasoning policy/presets (model-family aware)
├── compat.ts       # model-family compatibility profiles (Qwen vs unknown)
├── protocol.ts     # llama.cpp request/response wire types
└── config.ts       # plugin config schema and validation
tests/              # vitest suites (mock-based; no Harness core required)
```

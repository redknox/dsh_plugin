# llm-llamacpp

DeepSeek Harness LLM provider plugin targeting a local
[llama.cpp](https://github.com/ggml-org/llama.cpp) server through its
OpenAI-compatible `/v1/chat/completions` endpoint, designed for locally hosted
Qwen models.

The plugin owns the single provider route `llamacpp-local`. It is loaded by
DeepSeek Harness as a Cordis plugin and registers itself through the public
`ctx.llm` service contract only — no agent-loop internals are touched.

> **Status.** Issues #1-#5 are implemented: scaffold + registration, the
> llama.cpp streaming client, `LlmAdapter` message/stream translation, Qwen
> reasoning presets (`off`/`low`/`medium`/`xhigh`) with expert overrides, and
> tool-call streaming/schema support. Adaptive budgets (#6) and reliability
> (#7, including the provider-owned retry policy) land next.

## Requirements

- Node.js >= 20 (the harness host provides `@deepseek-ai/cordis`, `@deepseek-ai/dsh-llm`,
  and `@deepseek-ai/dsh-settings` as peer dependencies)
- A local llama.cpp server (e.g. `llama-server -m path/to/qwen3.gguf --port 8080`)

## Mounting in the harness config

Add an entry to the harness plugin list (e.g. `cordis.yml`):

```yaml
- id: llm-llamacpp
  name: llm-llamacpp            # or a path to this package's dist/index.js
  config:
    baseURL: http://127.0.0.1:8080  # llama.cpp OpenAI-compatible endpoint
    model: qwen3                    # default model id sent to the wire
    providerName: llama.cpp (Local) # name shown in model selectors
    # apiKeyEnv: LLAMACPP_API_KEY   # optional; set when a reverse proxy requires a key
    #   resolved per request through ctx.credentials (the web Models page can
    #   store it), falling back to the launching environment
    # apiKeyHeader: authorization   # 'authorization' sends 'Bearer <key>', anything else sends the raw key
    # streamIdleTimeoutMs: 300000
    # requestTimeoutMs: 60000  # optional hard per-attempt timeout, regardless of activity
    # endpoints:               # optional ordered fallback list; replaces baseURL
    #   - http://127.0.0.1:8080
    #   - http://10.0.0.2:8080
    # retryPolicy:             # optional; omission uses bounded normal defaults
    #   mode: normal           # normal | always
    #   maxRetries: 2
    #   retryableCodes: [RATE_LIMIT, SERVER, TIMEOUT, TRANSPORT, EMPTY_RESPONSE]
    #   backoff: { initialDelayMs: 500, maxDelayMs: 10000, jitterRatio: 0.1 }
    # reasoning:                    # optional semantic Qwen reasoning controls
    #   enabled: true               # false advertises and allows only 'off'
    #   preset: medium              # off | low | medium | xhigh
    #   wire: chat-template-kwargs  # chat-template-kwargs | reasoning-fields
    #   expert:                     # explicit expert override surface
    #     enabled: true
    #     effort: medium
    #     budgetTokens: 4096
    #     preserveThinking: true
```

The plugin registers provider route `llamacpp-local` and the configurable-provider
directory entry `llamacpp-local@llm-llamacpp`. Select the provider in the harness
by `provider: llamacpp-local`; the `model` value is passed through to the wire
verbatim, so changing Qwen models needs no plugin reload.

Configuration is validated at load: an invalid `baseURL` (not an http(s) URL)
or an empty `model`/`providerName` fails the plugin load clearly. When the
harness settings service is mounted, the same schema drives an `llm-llamacpp`
settings section that can override any field without a restart.

## Reasoning (Qwen thinking)

The provider exposes stable semantic levels (`off`, `low`, `medium`, `xhigh`)
through `resolveModel`/`ctx.llm.resolveModelInfo`, selectable as
`GenerateOptions.reasoningEffort`. Effort (semantic) and `budgetTokens`
(runtime thinking budget) are separate concepts; the built-in preset table
maps each level, and the `reasoning.expert` config overrides individual fields
without rewriting the table. Per-request effort wins over the configured
preset; `session-title` calls always disable thinking.

The resolved policy is translated to llama.cpp request fields only in the
request builder, and the fields are version-dependent:

- `wire: chat-template-kwargs` (default): `chat_template_kwargs = {
  enable_thinking, preserve_thinking? }` — Qwen chat-template kwargs honored
  by llama.cpp builds with the per-request template-kwargs hook (llama.cpp
  PR #13196). The runtime thinking budget is a separate inference control and
  is sent as the top-level `thinking_budget_tokens` per-request field.
- `wire: reasoning-fields`: top-level `reasoning_effort` (including `"none"`)
  and `thinking_budget_tokens`. Requires newer llama.cpp builds with native
  per-request reasoning support (PRs #22336 / #23116 / #26045).
  `preserve_thinking` is a chat-template kwarg that llama.cpp merges
  independently of the native fields, so it rides alongside them in either
  wire mode.

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

With a running llama.cpp server and a tool-capable Qwen model:

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
├── reasoning.ts    # Qwen reasoning policy/presets
├── protocol.ts     # llama.cpp request/response wire types
└── config.ts       # plugin config schema and validation
tests/              # vitest suites (mock-based; no Harness core required)
```

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
    # apiKeyHeader: authorization   # 'authorization' sends 'Bearer <key>', anything else sends the raw key
    # streamIdleTimeoutMs: 300000
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
  enable_thinking, thinking_budget }`. Honored by llama.cpp builds shipping
  the per-request template-kwargs hook for Qwen3 templates (llama.cpp
  PR #13196) across Qwen3-era versions. `off` sends `enable_thinking: false`.
- `wire: reasoning-fields`: top-level `reasoning_effort` (including `"none"`)
  and `reasoning_budget_tokens`. Requires newer llama.cpp builds with native
  per-request reasoning support (PRs #22336 / #23116 / #26045).

`preserveThinking: false` (expert) consumes thinking deltas without emitting
`reasoning` blocks to the harness stream.

## Tools

Harness tool schemas (`GenerateOptions.tools`) are sent to llama.cpp as
OpenAI-compatible `tools`, and streamed `tool_calls` deltas (fragmented ids,
function names, and JSON argument fragments, multiple calls per response,
mixed text+tool output) are reconstructed into Harness tool-call blocks.
Malformed or empty argument JSON fails the stream with `INVALID_TOOL_ARGUMENTS`.
Tool execution stays with Harness `ctx.tools`; this provider only translates
the protocol.

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

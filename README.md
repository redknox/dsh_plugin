# llm-llamacpp

DeepSeek Harness LLM provider plugin targeting a local
[llama.cpp](https://github.com/ggml-org/llama.cpp) server through its
OpenAI-compatible `/v1/chat/completions` endpoint, designed for locally hosted
Qwen models.

The plugin owns the single provider route `llamacpp-local`. It is loaded by
DeepSeek Harness as a Cordis plugin and registers itself through the public
`ctx.llm` service contract only — no agent-loop internals are touched.

> **Status.** Issues #1 (scaffold + registration), #2 (llama.cpp
> OpenAI-compatible streaming client), and #3 (`LlmAdapter` message and stream
> translation: `GenerateOptions` → wire request, wire chunks → `StreamChunk`,
> explicit rejection of tools/reasoning until #4/#5) are implemented. Tool
> calling, reasoning presets, adaptive budgets, and reliability land in later
> issues.

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
    # retryPolicy:                  # optional; omission uses bounded normal defaults
    #   mode: normal
    #   maxRetries: 2
    #   retryableCodes: [RATE_LIMIT, SERVER, TIMEOUT, TRANSPORT]
    #   backoff: { initialDelayMs: 500, maxDelayMs: 10000, jitterRatio: 0.1 }
```

The plugin registers provider route `llamacpp-local` and the configurable-provider
directory entry `llamacpp-local@llm-llamacpp`. Select the provider in the harness
by `provider: llamacpp-local`; the `model` value is passed through to the wire
verbatim, so changing Qwen models needs no plugin reload.

Configuration is validated at load: an invalid `baseURL` (not an http(s) URL)
or an empty `model`/`providerName` fails the plugin load clearly. When the
harness settings service is mounted, the same schema drives an `llm-llamacpp`
settings section that can override any field without a restart.

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

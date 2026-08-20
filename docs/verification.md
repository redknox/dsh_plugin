# Verification guide

How to check the `llm-llamacpp` plugin against a live DeepSeek Harness web
instance and the real llama.cpp server used during development
(`10.60.84.212:8040`, model `/models/Qwen3.8-27B-Q8_0.gguf`, API key from the
`LLAMA_API_TOKEN` environment variable).

## 1. Provider and model registration (API)

```bash
# 1) The provider route is registered and active
curl -s -X POST http://127.0.0.1:3080/api/llm.providers -H 'Content-Type: application/json' \
  -d '{"type":"client-request","rpcId":"x","method":"llm.providers","payload":{}}'
# expect: llamacpp-local  active: true, displayName "llama.cpp (Local Qwen3.8)"

# 2) Model catalog (with discovery enabled, this reflects the server's real model)
curl -s -X POST http://127.0.0.1:3080/api/llm.models -H 'Content-Type: application/json' \
  -d '{"type":"client-request","rpcId":"x","method":"llm.models","payload":{}}'
# expect: llamacpp-local group with /models/Qwen3.8-27B-Q8_0.gguf and reasoning
#         efforts off/low/medium/xhigh (default medium)

# 3) The API key resolves in the running process (credentials seam, env layer)
curl -s -X POST http://127.0.0.1:3080/api/credentials.describe -H 'Content-Type: application/json' \
  -d '{"type":"client-request","rpcId":"x","method":"credentials.describe","payload":{"refs":["LLAMA_API_TOKEN"]}}'
# expect: {"LLAMA_API_TOKEN": {"configured": true, "source": "env"}}
```

> The host's model-catalog view does not surface the context window; the
> plugin's `resolveModelInfo` does return it (`contextWindow: 204800`,
> discovered by issue #10).

## 2. Chat in the GUI

1. Open http://127.0.0.1:3080 and pick **llama.cpp (Local Qwen3.8)** in the
   model selector.
2. Send *"What is 2+2? Answer in one word."* — you should see grey reasoning
   (thinking) deltas then the answer (issues #3 translation + #4 reasoning).
3. Switch reasoning presets (off/low/medium/xhigh) and compare thinking time.
4. Ask something that needs a tool (math, lookup) — the agent should call
   tools through this provider (issue #5).

## 3. Observability (#8, #6, #11)

The plugin always emits structured telemetry events through its telemetry
sink (`started -> reasoning -> routing -> attempt -> finished`, one trace per
request, content-free). The most reliable way to inspect them is the
diagnostics surface (section 4 below); the standalone `diagnostics.mjs`
example shows the same data aggregated.

If the `dsh web` process is launched with a debug-level logger, the raw event
lines appear in its output:

```bash
grep "llm-llamacpp telemetry" <dsh-web-output>       # one JSON line per event
grep "llm-llamacpp reasoning decision" <dsh-web-output> # policy decision + rationale
grep "llm-llamacpp: endpoint .* failed" <dsh-web-output> # reliability retries/fallbacks
```

Events carry only field names and counts — never prompt content, tool
arguments, completions, or secrets (issue #8 privacy).

## 4. Standalone diagnostics (#12)

```bash
cd /path/to/llm-llamacpp
LLAMACPP_BASE_URL=http://10.60.84.212:8040 \
LLAMACPP_MODEL=/models/Qwen3.8-27B-Q8_0.gguf \
LLAMA_API_TOKEN=<token> node examples/diagnostics.mjs
```

Prints the machine-readable snapshot and the human-readable rendering:
endpoint health/backoff, request rate, TTFT/total latency, reasoning
effort/budget/tokens, tool activity, and recent failures.

The running harness also exposes the context service
`llm-llamacpp/diagnostics` (`snapshot()` / `render()`).

## 5. Standalone tool round-trip (#5)

```bash
cd /path/to/llm-llamacpp
LLAMACPP_BASE_URL=http://10.60.84.212:8040 \
LLAMACPP_MODEL=/models/Qwen3.8-27B-Q8_0.gguf \
LLAMA_API_TOKEN=<token> node examples/tool-call.mjs
```

## 6. Settings surface (#13, exploratory)

Issue #13 explored how far llama.cpp provider configuration can go through the
existing DSH Settings UI without a custom front-end. Findings and design
options live in `docs/exploration/settings-ui.md`; the repeatable smoke is:

```bash
cd /path/to/llm-llamacpp
DSH_URL=http://127.0.0.1:3080 \
LLAMACPP_BASE_URL=http://10.60.84.212:8040 \
LLAMA_API_TOKEN=<token> node examples/settings-poc.mjs
```

The script drives the exact RPC surface the Settings GUI uses and restores the
environment afterwards. Verified live against the running instance
(2026-02-14, plugin commit `5f5a042` after a restart so the rebuilt plugin
with `registerModelDiscovery` is loaded):

| Check | Observed |
|---|---|
| `settings.describe` | `llm-llamacpp` registered, `applies: live`, `writable: true`, `secrets: []`, `reasoning.preset: medium` |
| `settings.update` (`reasoning.preset: low`) | revision 0→1, persisted in `~/.dsh/settings.yaml`, re-describes as `low` with no restart |
| `credentials.set` | POC ref → `source: file`; `LLAMA_API_TOKEN` → `source: env`; the secret never appears in `settings.yaml`; unset cleans up |
| `llm.discoverModels` draft path | `{settingsNs: llm-llamacpp, baseURL, apiKey}` → `/models/Qwen3.8-27B-Q8_0.gguf` (`contextWindow: 204800` from `meta.n_ctx`) |
| `llm.discoverModels` provider path | `{settingsNs, provider: llamacpp-local}` → same model, answered from adapter knowledge, no network call |
| restore | `reasoning.preset` back to `medium`, user layer empty, no leftover sections |

## Issue-to-behavior map

| Issue | Observable in the running instance |
|---|---|
| #1 scaffold/registration | provider route + configurable-provider directory entry, settings live-update |
| #2 streaming client | SSE streaming, cancellation/timeout/HTTP diagnostics |
| #3 adapter translation | text/reasoning blocks, usage, finish reasons |
| #4 reasoning presets | off/low/medium/xhigh in the model selector |
| #5 tool calling | the agent calls tools through this provider |
| #6 adaptive budget | `reasoning.adaptive` config + decision logs |
| #7 reliability | multi-endpoint fallback/retry/backoff logs |
| #8 telemetry | `llm-llamacpp telemetry` JSON events |
| #9 routing | `routing` events (candidates + rationale) |
| #10 discovery | model catalog reflects the server; resolveModelInfo context |
| #11 feedback | `reasoning.feedback.enabled` + feedback rationale in reasoning events |
| #12 diagnostics | `examples/diagnostics.mjs` + `llm-llamacpp/diagnostics` ctx service |
| #13 settings exploration | `examples/settings-poc.mjs` + `docs/exploration/settings-ui.md`; discovery now serves `llm.discoverModels` for the `llm-llamacpp` namespace |

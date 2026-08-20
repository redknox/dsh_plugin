# Issue #13 — Exploring richer llama.cpp provider configuration in the DSH Settings UI

> Exploration/design issue. No complete Settings UI was built. This document
> records what the DSH settings surface already supports, what it cannot do
> without a custom front-end, the minimal proof of concept shipped with this
> issue, and the design options for follow-up work.

## 1. What the issue asked

Explore how far llama.cpp provider configuration can go in the DSH Settings UI
**without** writing a custom front-end:

- Which provider fields a settings surface can already edit and persist.
- How credentials flow (and how we know they never land in settings.yaml).
- Whether model-list data (dropdown) is reachable, and through which seam.
- What is impossible from the existing surface and therefore needs a custom
  front-end or upstream changes.

## 2. The surface we explored

The web GUI's **Models page** (`dsh-client-ui-settings-models`) renders one
editor per configurable provider, sourced from `llm.providers` (the
configurable-provider directory) + `settings.describe` (schema/value/revision)
+ `credentials.describe`.

Host RPC surface used (all plain `POST /api/<method>` with the
`{"type":"client-request","rpcId","method","payload"}` envelope):

| Method | Payload | Returns |
| --- | --- | --- |
| `settings.describe` | `{}` | namespaces: schema, value, base, user, applies, secrets, revision; writable; hasDocument |
| `settings.update` | `{ns, patch, expectedRevision?}` | new namespace view (deep-merge patch) |
| `settings.mutate` | `{ns, ops: [{op:'set'\|'unset', path}]}` | new namespace view |
| `settings.replace` | `{ns, section, expectedRevision?}` | new namespace view |
| `credentials.describe` | `{refs}` | configured/source/writable per ref |
| `credentials.set` / `credentials.unset` | `{ref, value?}` | `{}` |
| `llm.providers` | `{}` | configurable provider directory (provider, displayName, settingsNs, settingsPath, active) |
| `llm.models` | `{}` | registered adapters' model lists |
| `llm.discoverModels` | `{settingsNs, provider?, baseURL?, api?, apiKey?}` | candidate models for a draft (never stored) |

## 3. Findings

### 3.1 A generic form is not schema-driven — family layouts are hardcoded

- `layoutOf(ns)` in the Models page maps **only** `llm-deepseek` → `deepseek`
  and `llm-pi-ai` → `pi-ai`; every other namespace (including our
  `llm-llamacpp`) is `unknown`.
- The `unknown` branch renders the credential field plus the hint
  "remaining fields are edited in settings.yaml", and
  `submitDisabled` **includes `layout === "unknown"`** — the submit button is
  disabled, so the Models page cannot save our provider's non-credential
  fields. The one reachable write is the credential field, which goes through
  `credentials.set`, not settings.
- No generic schema-driven settings renderer exists anywhere in the client
  packages (`dsh-client-ui-settings-*`): the General page is slot-driven
  (`settings.general.item`), Plugins is feature-owned, and Models is
  family-owned. schemastery metadata (`role`, `description`, `comment`,
  `collapse`, `hidden`, `disabled`, `badges`, `link`) is rich but only
  consumed by those hand-written editors.

### 3.2 What a settings surface CAN already do for `llm-llamacpp` (proven by POC)

Everything below is exercised by `examples/settings-poc.mjs` against the live
instance and requires **zero front-end code**:

- **See the namespace** — `settings.describe` returns our schema, the resolved
  value (composition base + defaults), the user layer, `applies: 'live'`
  (changes take effect on the next request, no restart), and the redacted
  secret slots (`secrets: []` — our provider stores no secrets in settings).
- **Edit provider fields** — `settings.update`/`mutate` deep-merge a patch
  into the user layer, validate it against the schema, persist it into
  `~/.dsh/settings.yaml`, and re-resolve the plugin's options thunk live.
  Any field in the plugin `Config` is editable this way: `providerName`,
  `baseURL`, `endpoints`, `model`, `apiKeyHeader`, `apiKeyEnv`,
  `reasoning.*` (preset/expert/wire/adaptive/feedback), `discovery.*`,
  `diagnostics.*`, `telemetry.*`, `retryPolicy.*`.
- **Store credentials out of settings** — the credential field the Models page
  renders maps to a credential ref (derived or `apiKeyEnv`); the GUI calls
  `credentials.set`, the host persists into the credential store
  (`~/.dsh/.credentials.yaml`), and the plugin resolves it per request via
  `ctx.credentials.resolve` (env layer first, then the store). The POC proves
  a value written this way never appears in `settings.yaml`.
- **Model dropdown data** — with issue #13's change, `llm.discoverModels`
  works for our namespace:
  - draft path: `{settingsNs, baseURL, apiKey}` probes that endpoint
    (`/health` + `/v1/models` + `/props`) with the one-shot credential and
    returns `[{id, contextWindow}]`;
  - registered-route path: `{settingsNs, provider: 'llamacpp-local'}` answers
    from the adapter's own knowledge (configured model + TTL-cached
    discoveries), no network call;
  - failures degrade to `[]` (except caller cancellation), and the one-shot
    `apiKey` never leaves the interrogation.

### 3.3 What is NOT reachable from the existing surface

- **Saving non-credential fields from the Models page** — the `unknown`
  family disables submit; only the credential field is wired.
- **A model dropdown** in the Models page — the fetch action that populates
  dropdowns exists only inside the `deepseek`/`pi-ai` branches.
- **Connection testing, per-endpoint management, diagnostics panels** — the
  page has no affordance for these; diagnostics data is available to
  tooling/operators through the `llm-llamacpp/diagnostics` context service
  (issue #12) but not rendered by any UI.
- **Any mutation that bypasses settings/credentials seams** — the GUI cannot
  call the adapter directly; the host only exposes the RPC methods above.

## 4. Proof of concept shipped with this issue

1. `src/adapter.ts` — `LlamacppAdapter.discoverDraft(request)`: answers one
   `LlmModelDiscoveryRequest` (draft baseURL probe with one-shot credential,
   or provider-named knowledge answer), honoring cancellation and degrading
   probe failures to `[]`.
2. `src/index.ts` — registers `ctx.llm.registerModelDiscovery(NS, ...)` under
   the same namespace as the configurable-provider directory entry, so the
   host `llm.discoverModels` RPC serves our namespace.
3. `examples/settings-poc.mjs` — repeatable smoke: describe → live update
   `reasoning.preset` → verify persistence in settings.yaml → credential
   set/describe/unset → verify no secret in settings.yaml →
   `llm.discoverModels` (draft + provider paths) → restore.
4. Tests: 7 unit tests (`tests/discovery.test.ts` `discoverDraft` suite) + 1
   integration test (`tests/integration.test.ts`, `ctx.llm.discoverModels`
   through the mounted plugin, including the `NO_DISCOVERY` error for an
   unknown namespace).

## 5. Design options for follow-up (issue candidates)

### Option A — upstream: generalize the Models page (deepseek-harness)

Teach the Models page to render **any** configurable-provider namespace from
its `settings.describe` schema + schemastery metadata: an unknown family gets
a generic schema-driven form (fields from the schema, credential field via the
secrets slots, model dropdown via `llm.discoverModels` when a discovery is
registered, submit enabled). The family branches for deepseek/pi-ai remain as
curated overrides. This benefits every third-party provider, not just ours.

- Pros: the only path that makes the GUI fully usable for llama.cpp; upstream
  value for all providers.
- Cons: lives in the harness repo (out of this plugin's control); needs
  schemastery metadata discipline from providers.

### Option B — plugin-side: extend the Models page with a curated section

A `dsh-client-ui-*`-style plugin owned by `llm-llamacpp` that registers a
Models-section renderer for our namespace (like the deepseek/pi-ai branches):
baseURL/endpoints/model fields, reasoning preset select, connect-test button
(→ `llm.discoverModels`), and a model dropdown. The host already keys
section rendering by `settingsNs`, so a plugin can attach without touching the
core page.

- Pros: we control the UX; no upstream wait.
- Cons: new client-plugin surface to build and maintain; the client plugin
  ecosystem (dsh-client-modules) must support third-party registration — to be
  verified.

### Option C — do nothing in the GUI; keep the API + settings.yaml as the surface

The POC proves the entire configuration plane already works headlessly:
`settings.update`/`mutate` (live), `credentials.set`, `llm.discoverModels`,
and `settings.yaml` for power users. A follow-up could simply document the
RPC recipes.

- Pros: zero UI work; everything is testable and automatable.
- Cons: non-developers still cannot configure llama.cpp from the GUI.

### Recommended sequence

1. **Option A (upstream)** — small, high-leverage: relax `layoutOf` to render
   a generic schema-driven editor for unknown namespaces, enabling submit and
   a discovery-backed dropdown when available. Do it in deepseek-harness.
2. **Option C documentation** — ship the RPC recipes (this doc + the POC
   script) so operators have a repeatable path today.
3. Only if A is rejected upstream, consider **Option B**.

## 6. Explicit non-goals (from issue #13)

No complete Settings UI, no replacement of the DSH Settings app, no direct
UI→adapter mutation API, no plaintext credentials in plugin config, no full
endpoint-management/diagnostics panel UI, no agent-loop changes.

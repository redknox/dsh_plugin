# Upstream UI PoC — generic llamacpp family in the Models settings page

Issue #13's acceptance criteria require a minimal provider-configuration PoC
**demonstrated in the DSH Settings / Models UI**. The shipped surface today
renders an `unknown` family for `llm-llamacpp` (credential field only, submit
disabled), so a UI PoC cannot be produced from the plugin repository alone:
the Models page editor is upstream code in
`@deepseek-ai/dsh-client-ui-settings-models`.

This directory holds a **minimal upstream patch prototype** that makes the
`llm-llamacpp` namespace fully editable in the running Models page. It is a
prototype, not the final design — the follow-up is an upstream PR
(deepseek-harness) that generalizes the unknown-family renderer so **every**
third-party provider benefits (see `docs/exploration/settings-ui.md`, design
option 1).

## What the patch does

Applied to `dsh-client-ui-settings-models/lib/client.js` (the shipped web
bundle, CJS factory format):

1. `layoutOf(ns)` — recognize `llm-llamacpp` as a new `llamacpp` family
   instead of falling into `unknown` (which disables submit).
2. `LlamacppModelPicker` (new component) — "Fetch available models" button
   that calls the host `llm.discoverModels` (the seam registered in
   `src/index.ts` for issue #13) and writes the picked model id into the
   `model` draft field.
3. `llamacppFields()` (inside `ProviderEditor`) — the curated editor for the
   family: API key (existing credential flow through `credentials.set`),
   Display name (`providerName`), Base URL, Model, Reasoning (`enabled`
   on/off), Reasoning preset (`off/low/medium/xhigh`), and the model
   candidate picker. The preset select is locked (`disabled`) while
   Reasoning is `Disabled` — with the master thinking switch off the preset
   is meaningless.
4. Render point — `layout === "llamacpp" ? llamacppFields() : ...`.

Saving uses the **existing** editor write path untouched: draft path ops via
`settings.mutate` (live update, no restart) plus `credentials.set` for the
key — exactly what the RPC PoC (`examples/settings-poc.mjs`) proved headless.

## Applying / reverting

```bash
# target (host installation):
HOST=~/.npm/_npx/<hash>/node_modules/@deepseek-ai/dsh-client-ui-settings-models/lib/client.js

# apply
patch -p1 -d <dirname $HOST> < settings-models-llamacpp-family.patch

# revert
patch -p1 -R -d <dirname $HOST> < settings-models-llamacpp-family.patch
```

The patch is generated against the bundle as shipped with
`dsh-client-ui-settings-models@0.1.0-rc.8`; a reinstall/upgrade of the host
overwrites it (expected for a PoC). The plugin repository itself is untouched
by the patch — it only changes the host's Models page.

## Verified

- `node --check` on the patched bundle: syntax OK.
- Host restart → Models page renders the llamacpp editor; see
  `docs/verification.md` §7 for the manual UI walkthrough and the live
  `settings.describe` / `settings.update` / `credentials.describe` evidence.
- The underlying write path is the same `settings.mutate` + `credentials.set`
  flow exercised by `examples/settings-poc.mjs`.

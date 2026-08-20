# Upstream: schema-driven generic provider editor (issue #14)

Issue #14 turns the #13 Models-UI prototype into a maintainable,
upstreamable configuration path: instead of patching the installed host
bundle, the generic editor now lives **in the DeepSeek Harness source tree**
as a reviewable branch/PR candidate, and `llm-llamacpp` is its first real
validation case.

## The change

Branch `feat/generic-provider-editor` in `deepseek-ai/deepseek-harness`
(commit `d9f6013`), diff captured in `generic-provider-editor.patch`:

- **`packages/client/ui-settings-models/src/client/GenericSchemaEditor.tsx`**
  (new) — renders any configurable provider's own `settings.describe` schema
  as editable fields: strings, booleans, enum/union-of-literal choices,
  numbers, and simple nested objects. Unsupported
  constructs (array/dict/any/heterogeneous union/…) are surfaced explicitly
  with an "edit in settings.yaml" hint and are never rebuilt or dropped by a
  save. A `model` string field gains an optional discovery-backed picker
  (`llm.discoverModels`) that degrades to the plain text input when
  discovery is unavailable or fails. **Master-switch controls**: a boolean
  field declared with Schemastery's typed `.extra('extra', {controls:
  ['sibling', …]})` metadata disables those sibling fields while it is off
  (e.g. `reasoning.enabled` → `['preset']`) — schema-driven, usable by any
  provider, never a hard-coded family rule. **Semantic groups + field help
  (issue #19)**: providers declare advisory presentation hints through
  `.extra('extra', {ui: {label, description, collapsed}})` — top-level object
  fields render as collapsible semantic groups (label from `ui.label`,
  humanized key fallback; collapsed from `ui.collapsed`, default expanded),
  scalar fields stay visible, field labels honor `ui.label`, and
  `ui.description` renders as one short line of help under the control.
  Absent/malformed `ui` metadata degrades to the previous behavior.
- **`ProviderEditor.tsx`** — unknown namespaces now render the shared
  credential field plus the generic editor, and submit is no longer disabled
  for them. Writes stay on the existing `settings.mutate` path-ops +
  `credentials.set` flow, so revision checking, live-apply, and rollback
  semantics are unchanged. Curated `deepseek`/`pi-ai` layouts are untouched.
- **`locales.ts`** — `genericOptional` / `genericRequired` /
  `genericUnsupported` / `genericModelFetch` (en + zh).
- **Tests** — `tests/generic-editor.client.spec.tsx` (12 cases: render of
  every supported primitive, minimal path-op saves, nested-object subtree
  save, master-switch controls, semantic groups + collapsed state + field
  label/description, ui-metadata fallback, credential isolation via
  `credentials.set`, discovery success and failure fallback, and an
  unrelated synthetic provider proving the fallback is not llama.cpp-
  specific); two existing
  cases updated to the new unknown-namespace behavior. Suite: 231 tests
  green.

## How it validates llama.cpp (issue #14 tasks)

| Task | Status |
|---|---|
| Inspect Models-page implementation, find the smallest seam | `ProviderEditor` unknown branch was the seam; no new slot/registry needed |
| Replace hard-coded llamacpp prototype with a generic renderer | `GenericSchemaEditor` renders any namespace; the #13 `llamacpp` family patch was removed from the bundle |
| Render string / boolean / enum / optional / simple nested objects | Covered by the editor + tests |
| Curated layouts stay explicit overrides | `deepseek`/`pi-ai` branches untouched |
| Credentials through `credentials.describe/set/unset`, never in settings | Shared credential field; test asserts no secret in `settings.mutate` ops |
| `llm.discoverModels` when registered, plain text fallback otherwise | Model picker + fallback test (NO_DISCOVERY) |
| Save honors `settings.mutate/update`, validation, revisions, live apply, rollback | Reuses `applyOnce`/`pathOps` unchanged; tests assert expectedRevision + ops |
| llama.cpp basic fields through the generic path | providerName / baseURL / model / credential / reasoning.enabled / reasoning.preset all render + save |
| Unrelated synthetic provider renders generically | `llm-synthetic` fixture test |
| Unsupported schema shapes documented, never silently dropped | Explicit hint per unsupported field; non-object subtree → explicit hint |
| Clean PR-ready diff, not a generated-bundle edit | This branch/diff; the running instance's bundle is only a temporary e2e artifact |
| Plugin docs updated with upstream status + minimum Harness version | `docs/exploration/settings-ui.md` §5, `docs/verification.md` §8 |

## Reproducing the build/tests

```bash
git clone --depth 1 https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
git checkout feat/generic-provider-editor   # or apply generic-provider-editor.patch
corepack pnpm install --frozen-lockfile
# type declarations (client face; unrelated packages may show pre-existing
# typert codegen errors — ui-settings-models itself compiles):
pnpm exec tsc -b packages/client/ui-settings-models/tsconfig.json
# bundle:
pnpm --filter @deepseek-ai/dsh-client-ui-settings-models exec tsdown --config-loader tsx
# tests:
pnpm exec vitest run packages/client/ui-settings-models/tests
```

## Applying the diff to a source checkout

```bash
patch -p1 < generic-provider-editor.patch
```

## Note on the running instance

For live verification the built `lib/client.js` was copied over the host
install (the #13 llamacpp-family patch is superseded by the generic editor).
That copy is a **temporary e2e artifact only** — the deliverable is this
source-level change; the upstream PR is the shipping path.

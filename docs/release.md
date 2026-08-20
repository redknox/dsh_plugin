# Release / distribution (issue #17)

Issue #15 proved **source/Git installation**; this issue proves the
**prebuilt tarball** path and prepares npm distribution. The published
artifact preserves the provider boundary: **llama.cpp is the provider**,
Qwen is a first-class / best-validated model family (see #16 positioning and
`docs/exploration/qwen-audit.md` for the compatibility-profile seams).

## Package identity — decided

| Item | Decision |
|---|---|
| npm name | **`llm-llamacpp`** (unscoped; verified free on the npm registry, 2026-08-20) |
| version | `0.1.0` (first distributable release candidate) |
| license | MIT (`LICENSE` shipped in the tarball) |
| metadata | `description`, `repository`, `homepage`, `bugs`, `keywords`, `engines.node >= 20`, `files: ["dist", "cordis.patch.yml"]`, `exports` incl. `./cordis.patch.yml`, `dsh.bundle.patch`, `peerDependencies` (cordis / dsh-credentials / dsh-llm / dsh-settings) |
| prebuilt `dist/` | **yes** — the published tarball ships `dist/` built from source; no install-time toolchain needed |
| `prepare` script | **kept, idempotent** — builds only when `dist/index.js` is absent (VCS/Git installs), skips on prebuilt tarball installs |

## Prebuilt tarball verification (recorded)

`pnpm pack` → `llm-llamacpp-0.1.0.tgz` contains exactly the runtime
artifacts: `dist/index.js`, `dist/types/*.d.ts`, `cordis.patch.yml`,
`package.json`, `README.md`, `LICENSE` (no `src/`, `tests/`, or
`node_modules`).

Verified on a fresh DSH profile (`fresh17`, dsh 0.1.0-rc.7, pnpm 11.7.0):

| Check | Result |
|---|---|
| `dsh plugin --profile fresh17 add ./llm-llamacpp-0.1.0.tgz` | ✅ one-pass install; **no Git `prepare`/`allowBuilds` recovery** |
| `dsh.profile.bundles` | ✅ `['@deepseek-ai/dsh-base', 'llm-llamacpp']` (auto-reconciled) |
| `--dump-config` | ✅ `# == llm-llamacpp` → `- id: llm-llamacpp, name: llm-llamacpp` (no absolute path) |
| Boot (official `boot()` API) | ✅ `llm` service mounted; `llamacpp-local` provider active from the packaged dependency tree |
| Dependency identity | ✅ peers resolve via `~/.dsh/profiles/node_modules` shared links — single Harness/Cordis runtime identity |
| `remove` + reinstall | ✅ bundle list reconciles to `['@deepseek-ai/dsh-base']` and back; re-boot OK |

## Public install paths

```bash
# prebuilt tarball (no build authorization, no checkout)
dsh plugin --profile <name> add ./llm-llamacpp-0.1.0.tgz

# source / Git commit (needs the allowBuilds step — see docs/install.md)
dsh plugin --profile <name> add github:redknox/dsh_plugin#<commit-sha>

# npm registry (published)
dsh plugin --profile <name> add llm-llamacpp
```

## Published — llm-llamacpp@0.1.0 (2026-08-20)

Published to the npm registry (first release). Registry E2E on a fresh DSH
profile (`npm-e2e`, dsh 0.1.0-rc.7, pnpm 11.7.0):

| Check | Result |
|---|---|
| `npm publish` (granular token, Read-and-write + Bypass 2FA) | ✅ `+ llm-llamacpp@0.1.0`; registry shows `dist-tags.latest = 0.1.0`, tarball present |
| `dsh plugin --profile npm-e2e add llm-llamacpp@0.1.0` | ✅ registry install, **no allowBuilds / no checkout / no npx cache** |
| `dsh.profile.bundles` | ✅ `['@deepseek-ai/dsh-base', 'llm-llamacpp']` (auto-reconciled) |
| `--dump-config` | ✅ `# == llm-llamacpp` → package-name entry |
| Boot (official `boot()` API) | ✅ `llm` service mounted; `llamacpp-local` provider active from the registry dependency tree |
| Dependency identity | ✅ peers via `~/.dsh/profiles/node_modules` shared links — single Harness/Cordis runtime identity |
| `remove` + reinstall | ✅ bundles reconcile `['@deepseek-ai/dsh-base']` ↔ `['@deepseek-ai/dsh-base', 'llm-llamacpp']` |

Publishing note: npm requires **2FA or a granular access token with "Bypass
2FA for publish"** for new unscoped packages; use the bypass token (or an
OTP) for future releases. `0.1.0` is immutable — later fixes must bump the
version (`0.1.1`, …).

## Release checklist (npm publication)

npm publication is a **separate external release action** and must be
explicitly authorized at release time (issue #17 non-goal / note). When
authorized:

1. `npm test && npm run typecheck && npm run build` (all green baseline).
2. `npm pack` / `pnpm pack` — inspect the tarball (contents above).
3. Record the exact package version and DSH version; install the **registry**
   artifact into a fresh profile and record the E2E (mirroring the tarball
   verification table).
4. `npm publish` with the decided `llm-llamacpp` name.

## Compatibility wording (release constraint)

Public metadata and README present `llm-llamacpp` as a generic llama.cpp
provider; Qwen is first-class / best-validated; **compatibility claims are
evidence-backed only** — verified versions/models are separated from
unverified generic paths (no claim that any llama.cpp-loadable family is
supported).

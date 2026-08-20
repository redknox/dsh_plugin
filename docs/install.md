# Installing llm-llamacpp (issue #15 Git, #17 tarball/npm)

The repository is a DSH-native installable bundle: `package.json` declares
`dsh.bundle.patch` and ships `cordis.patch.yml` (a thin layer mounting the
plugin **by package name**), plus an idempotent `prepare` script that builds
`dist/` from source only when absent (VCS/Git installs). Prebuilt tarballs
(and the future npm artifact) ship `dist/` already built. No manual Cordis
patch edits, no
absolute filesystem paths, no `node_modules` symlink hacks.

## Target user flow

```bash
# 1. install from an exact commit (creates the profile on first use)
dsh plugin --profile <name> add github:redknox/dsh_plugin#<commit-sha>

# 2. inspect the composed profile (bundle layer + loader entry)
dsh --profile <name> --dump-config

# 3. boot
dsh --profile <name>
```

After installation the llama.cpp provider is active automatically — the
installed package declares a DSH bundle, so `dsh plugin` appends it to the
profile's ordered bundle list (`dsh.profile.bundles`), and the profile's
`cordis.patch.yml` mounts `llm-llamacpp` by package name.

## pnpm ≥ 10 build authorization (first install only)

Git dependencies build on install through their `prepare` script. pnpm ≥ 10
blocks untrusted install-time build scripts by default — the first `add`
fails with `ERR_PNPM_GIT_DEP_PREPARE_NOT_ALLOWED` and prints the exact key to
allow. Recovery flow:

```bash
# 1. the failing `dsh plugin ... add github:...` prints e.g.:
#    allowBuilds:
#      llm-llamacpp@https://codeload.github.com/redknox/dsh_plugin/tar.gz/<sha>: true

# 2. append that block to the profile's pnpm-workspace.yaml:
#    ~/.dsh/profiles/<name>/pnpm-workspace.yaml

# 3. re-run the exact same `dsh plugin ... add github:...` command — the
#    package's own prepare builds dist/ and installation completes.
```

The `allowBuilds` key is tied to the exact tarball URL (commit); a new commit
needs the corresponding key (or `pnpm approve-builds`).

## What installs where

- The package is installed into `~/.dsh/profiles/<name>/node_modules/llm-llamacpp`
  (source checkout; `dist/` is built by `prepare`, since `dist/` is
  gitignored).
- `dsh.bundle.patch` → `cordis.patch.yml` → `dsh.plugin` reconciliation adds
  `llm-llamacpp` to `dsh.profile.bundles` (alongside the shipped
  `@deepseek-ai/dsh-base`).
- Runtime `@deepseek-ai/*` dependencies (cordis, dsh-llm, dsh-settings,
  dsh-credentials) resolve through the shared
  `~/.dsh/profiles/node_modules` fallback, which symlinks the dsh
  installation's own dependency tree — the profile shares the **same**
  Cordis/Harness runtime identity as the host, no duplicate instances.

## Verified

Tested with `dsh` **0.1.0-rc.7** + pnpm **11.7.0** on a fresh profile
(`fresh15`, default template `["@deepseek-ai/dsh-base"]`), commit
**`313c210e43256c8d85392dfaf6c50642ac19a503`** (2026-08-20):

| Check | Result |
|---|---|
| `dsh plugin --profile fresh15 add github:redknox/dsh_plugin#313c210…` | ✅ prepare blocked once (`ERR_PNPM_GIT_DEP_PREPARE_NOT_ALLOWED`), allowed via `allowBuilds`, install completed |
| `dsh.profile.bundles` | ✅ `['@deepseek-ai/dsh-base', 'llm-llamacpp']` (auto-appended) |
| `--dump-config` | ✅ `# == llm-llamacpp` layer: `- id: llm-llamacpp, name: llm-llamacpp` (package name, no absolute path) |
| Boot (official `boot()` API) | ✅ `llm` service mounted; `llamacpp-local` in `listConfigurableProviders()` (`displayName: llama.cpp (Local)`, `settingsNs: llm-llamacpp`) |
| Dependency identity | ✅ cordis/dsh-llm/dsh-settings/dsh-credentials resolve via `~/.dsh/profiles/node_modules` symlinks to the dsh installation (single runtime identity) |
| `dsh plugin --profile fresh15 remove llm-llamacpp` | ✅ dependency removed and bundle list reconciled to `['@deepseek-ai/dsh-base']` |
| `dsh plugin --profile fresh15 add github:…` (reinstall) | ✅ dependency and bundle list restored; boot re-verified OK |

## Repeatable smoke

```bash
# from this repo: build + tests still pass unchanged
npm run build && npm test

# fresh-profile Git install (exact SHA), with the allowBuilds step above
dsh plugin --profile <fresh> add github:redknox/dsh_plugin#$(git rev-parse HEAD)
dsh --profile <fresh> --dump-config | grep -A 1 '== llm-llamacpp'
```

#!/usr/bin/env node
/**
 * Deploy the generic-provider-editor UI patch (issues #14/#19) into a
 * DeepSeek Harness source checkout.
 *
 * Why this exists: the Harness Models page renders a provider family it does
 * not know (ours) with a credential field only — `providerName`, `baseURL`,
 * `model`, reasoning and the other settings stay invisible. The generic
 * schema-driven editor that fixes this lives in
 * `upstream/generic-provider-editor.patch` (an upstream-PR candidate), **not**
 * inside the plugin package. A Harness upgrade replaces the client bundle, so
 * the patch must be re-applied after upgrading Harness.
 *
 * Usage:
 *   node scripts/deploy-ui-patch.mjs [harness-checkout]
 *   DSH_HARNESS=/path/to/deepseek-harness node scripts/deploy-ui-patch.mjs
 *
 * Steps: apply the patch (3-way merge) → build the
 * `@deepseek-ai/dsh-client-ui-settings-models` client bundle → verify. The web
 * instance must then be restarted so the bundle revision is re-hashed.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, readlinkSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const patchPath = join(repoRoot, 'upstream', 'generic-provider-editor.patch')
const targetRel = 'packages/client/ui-settings-models'
const bundleRel = `${targetRel}/lib/client.js`
const marker = 'GenericSchemaEditor'

/** Resolve the Harness checkout: argv → DSH_HARNESS → the profile's shared link. */
function resolveHarness() {
  const candidates = []
  if (process.argv[2] !== undefined) candidates.push(process.argv[2])
  if (process.env.DSH_HARNESS !== undefined) candidates.push(process.env.DSH_HARNESS)
  // The profile's shared module fallback links straight into the checkout;
  // walk up from the link target to the repository root.
  const link = join(homedir(), '.dsh', 'profiles', 'node_modules', '@deepseek-ai', 'cordis')
  if (existsSync(link)) {
    try {
      const real = readlinkSync(link)
      let dir = real
      for (let depth = 0; depth < 8; depth += 1) {
        const root = repoRootOf(dir)
        if (root !== undefined) {
          candidates.push(root)
          break
        }
        const parent = dirname(dir)
        if (parent === dir) break
        dir = parent
      }
    } catch {
      // not a symlink: fall through to the explicit candidates
    }
  }
  for (const candidate of candidates) {
    if (existsSync(join(candidate, targetRel, 'src', 'client', 'ProviderEditor.tsx'))) return resolve(candidate)
  }
  throw new Error(
    'cannot locate a DeepSeek Harness checkout; pass it as an argument or set DSH_HARNESS',
  )
}

/** The git repository root containing `dir`, or undefined when it is not in one. */
function repoRootOf(dir) {
  try {
    return run('git', ['-C', dir, 'rev-parse', '--show-toplevel'], dir).trim()
  } catch {
    return undefined
  }
}

function run(command, args, cwd) {
  return execFileSync(command, args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
}

const harness = resolveHarness()
console.log(`harness: ${harness}`)
console.log(`patch:   ${patchPath}`)

// 1. Apply the patch (3-way, so a moved context still merges).
try {
  const out = run('git', ['apply', '--3way', patchPath], harness)
  if (out.trim().length > 0) console.log(out.trim())
  console.log('applied: patch merged into the checkout')
} catch (error) {
  const detail = `${error.stdout ?? ''}${error.stderr ?? ''}`.trim()
  if (detail.includes('already applied') || detail.includes('patch does not apply')) {
    console.error(`patch not applied cleanly:\n${detail}`)
    console.error('If it is already applied, skip this step; otherwise resolve the conflict manually and re-run the build.')
    process.exit(1)
  }
  console.error(`git apply failed:\n${detail}`)
  process.exit(1)
}

// 2. Build the client bundle in place (the running instance loads this file).
const tsdown = join(harness, 'node_modules', '.bin', 'tsdown')
if (!existsSync(tsdown)) {
  console.error(`tsdown not found at ${tsdown}; run the Harness install (pnpm install) first`)
  process.exit(1)
}
try {
  run('sh', [tsdown, '--config-loader', 'tsx'], join(harness, targetRel))
  console.log('built: client bundle')
} catch (error) {
  console.error(`bundle build failed:\n${error.stdout ?? ''}${error.stderr ?? ''}`)
  process.exit(1)
}

// 3. Verify the deployed bundle carries the generic editor.
const bundle = join(harness, bundleRel)
const content = readFileSync(bundle, 'utf8')
if (!content.includes(marker)) {
  console.error(`verification failed: ${bundle} does not contain ${marker}`)
  process.exit(1)
}
console.log(`verified: ${bundle} contains ${marker}`)
console.log('\nNow restart the DSH web instance so the bundle revision is re-hashed.')

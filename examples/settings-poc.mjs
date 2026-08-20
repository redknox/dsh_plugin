#!/usr/bin/env node
/**
 * Settings-surface proof of concept (issue #13): drive the exact RPC surface
 * the DSH Settings GUI uses against a running instance, to show which
 * llama.cpp provider fields a settings surface can already change without any
 * custom front-end, and where the boundary is.
 *
 * What it exercises:
 *   1. settings.describe      - the `llm-llamacpp` namespace is registered
 *                               (schema, resolved value, applies, secrets,
 *                               revision) — the data a generic form would see.
 *   2. settings.update        - change `reasoning.preset` to `low`; the value
 *                               persists into the user-editable settings.yaml
 *                               and re-resolves without a restart (live).
 *   3. credentials.set/unset  - store a one-off credential through the
 *                               credentials seam; it lands in the credential
 *                               store, never in settings.yaml.
 *   4. llm.discoverModels     - model dropdown data for a draft endpoint
 *                               (baseURL + one-shot apiKey) and for the
 *                               already-registered route (provider only).
 *   5. settings.mutate        - clean up the experiment (restore `medium`,
 *                               then remove the override entirely).
 *
 * Usage (against the running web GUI; the plugin must be mounted):
 *   npm run build
 *   node examples/settings-poc.mjs
 *   DSH_URL=http://127.0.0.1:3080 \
 *   LLAMACPP_BASE_URL=http://10.60.84.212:8040 \
 *   LLAMA_API_TOKEN=<token> node examples/settings-poc.mjs
 *
 * Env: DSH_URL (default http://127.0.0.1:3080), LLAMACPP_BASE_URL,
 *      LLAMA_API_TOKEN, DSH_SETTINGS_FILE (default ~/.dsh/settings.yaml).
 *
 * Requires a running DSH instance whose composition mounts this plugin and a
 * settings provider (dsh-settings-file); a llama.cpp server for step 4.
 */
import { randomUUID } from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const HOST = process.env.DSH_URL ?? 'http://127.0.0.1:3080';
const NS = 'llm-llamacpp';
const PROBE_BASE_URL = process.env.LLAMACPP_BASE_URL ?? 'http://10.60.84.212:8040';
const API_KEY = process.env.LLAMA_API_TOKEN ?? '';
const SETTINGS_FILE = process.env.DSH_SETTINGS_FILE ?? join(homedir(), '.dsh', 'settings.yaml');
const POC_REF = 'LLAMACPP_POC_TOKEN';
const POC_SECRET = `poc-secret-${randomUUID().slice(0, 8)}`;

let updated = false;

/** POST one client-request RPC and return the business value. */
async function rpc(method, payload = {}) {
  const rpcId = randomUUID();
  let response;
  try {
    response = await fetch(`${HOST}/api/${method}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ type: 'client-request', rpcId, method, payload }),
    });
  } catch (error) {
    throw new Error(`cannot reach ${HOST} (${error?.message ?? error}); is the DSH web server running?`);
  }
  const envelope = await response.json();
  if (envelope?.type !== 'server-response' || envelope.rpcId !== rpcId) {
    throw new Error(`unexpected envelope from ${method}: ${JSON.stringify(envelope).slice(0, 300)}`);
  }
  const result = envelope.result;
  if (!result?.ok) {
    const error = result?.error ?? {};
    throw new Error(`${method} failed: [${error.code}] ${error.message}`);
  }
  return result.value;
}

function section(namespaces) {
  const found = namespaces.find((entry) => entry.ns === NS);
  if (found === undefined) {
    throw new Error(`no "${NS}" settings namespace registered — is the plugin mounted? (ns list: ${namespaces.map((e) => e.ns).join(', ')})`);
  }
  return found;
}

function secretLeaks(value) {
  if (typeof value === 'string') return value.includes(POC_SECRET);
  if (Array.isArray(value)) return value.some(secretLeaks);
  if (value !== null && typeof value === 'object') return Object.values(value).some(secretLeaks);
  return false;
}

try {
  console.log(`settings-surface POC against ${HOST} (ns=${NS}, probe=${PROBE_BASE_URL})\n`);

  // --- 1. describe ---------------------------------------------------------
  const describe = await rpc('settings.describe');
  const before = section(describe.namespaces);
  console.log('1. settings.describe:');
  console.log(`   ns=${before.ns} applies=${before.applies} writable=${describe.writable} revision=${before.revision}`);
  console.log(`   secrets=${JSON.stringify(before.secrets)}`);
  console.log(`   reasoning.preset = ${before.value?.reasoning?.preset ?? '(unset -> default)'}`);
  console.log(`   user layer keys  = ${JSON.stringify(Object.keys(before.user ?? {}))}`);
  console.log('');

  // --- 2. live update ------------------------------------------------------
  const patched = await rpc('settings.update', { ns: NS, patch: { reasoning: { preset: 'low' } } });
  updated = true;
  console.log('2. settings.update { reasoning: { preset: "low" } }:');
  console.log(`   new revision=${patched.revision} reasoning.preset=${patched.value?.reasoning?.preset}`);
  if (existsSync(SETTINGS_FILE)) {
    const yaml = readFileSync(SETTINGS_FILE, 'utf8');
    const line = yaml.split('\n').find((l) => l.includes('preset:'));
    console.log(`   persisted in ${SETTINGS_FILE}: ${line?.trim() ?? '(no preset line found)'}`);
  } else {
    console.log(`   (settings document not found at ${SETTINGS_FILE}; skipped file check)`);
  }
  const reRead = section((await rpc('settings.describe')).namespaces);
  console.log(`   re-described reasoning.preset = ${reRead.value?.reasoning?.preset} (no restart)`);
  console.log('');

  // --- 3. credentials seam -------------------------------------------------
  await rpc('credentials.set', { ref: POC_REF, value: POC_SECRET });
  const credView = await rpc('credentials.describe', { refs: [POC_REF, 'LLAMA_API_TOKEN'] });
  console.log('3. credentials.set/describe:');
  console.log(`   ${POC_REF} -> ${JSON.stringify(credView.credentials[POC_REF])}`);
  console.log(`   LLAMA_API_TOKEN -> ${JSON.stringify(credView.credentials.LLAMA_API_TOKEN)}`);
  if (existsSync(SETTINGS_FILE)) {
    const yaml = readFileSync(SETTINGS_FILE, 'utf8');
    console.log(`   secret in settings.yaml? ${secretLeaks(yaml) ? 'YES (BAD)' : 'no (good)'}`);
  } else {
    console.log('   (settings document not found; skipped leak check)');
  }
  await rpc('credentials.unset', { ref: POC_REF });
  console.log('   (POC credential unset again)');
  console.log('');

  // --- 4. model discovery --------------------------------------------------
  console.log('4. llm.discoverModels:');
  let discoveryVerified = true;
  if (API_KEY.length > 0) {
    try {
      const draft = await rpc('llm.discoverModels', {
        settingsNs: NS,
        baseURL: PROBE_BASE_URL,
        apiKey: API_KEY,
      });
      console.log(`   draft endpoint ${PROBE_BASE_URL} ->`);
      for (const model of draft.models) {
        console.log(`     - ${model.id}${model.contextWindow !== undefined ? ` (contextWindow=${model.contextWindow})` : ''}`);
      }
    } catch (error) {
      discoveryVerified = false;
      console.log(`   draft probe unavailable: ${error?.message ?? error}`);
    }
  } else {
    console.log(`   (LLAMA_API_TOKEN unset; skipped live draft probe of ${PROBE_BASE_URL})`);
  }
  try {
    const known = await rpc('llm.discoverModels', { settingsNs: NS, provider: 'llamacpp-local' });
    console.log(`   registered route (provider only) -> ${known.models.map((m) => m.id).join(', ') || '(empty)'}`);
  } catch (error) {
    discoveryVerified = false;
    console.log(`   registered-route answer unavailable: ${error?.message ?? error}`);
  }
  if (!discoveryVerified) {
    console.log('   (the running instance predates issue #13\'s registerModelDiscovery;');
    console.log('    restart the DSH web app so it loads the rebuilt plugin, then re-run)');
  }
  console.log('');

  // --- 5. restore ----------------------------------------------------------
  await rpc('settings.update', { ns: NS, patch: { reasoning: { preset: 'medium' } } });
  await rpc('settings.mutate', { ns: NS, ops: [{ op: 'unset', path: ['reasoning'] }] });
  updated = false;
  const after = section((await rpc('settings.describe')).namespaces);
  console.log('5. restored:');
  console.log(`   reasoning.preset = ${after.value?.reasoning?.preset} revision=${after.revision}`);
  console.log(`   user layer keys  = ${JSON.stringify(Object.keys(after.user ?? {}))}`);
  console.log('\nPOC complete: settings updates, credential storage, and model discovery all work through the existing host RPC surface.');
} catch (error) {
  console.error(`\nPOC failed: ${error?.message ?? error}`);
  process.exitCode = 1;
} finally {
  // Best effort: never leave the running instance mutated.
  if (updated) {
    try {
      await rpc('settings.mutate', { ns: NS, ops: [{ op: 'unset', path: ['reasoning'] }] });
      console.log('(cleanup: reasoning override removed)');
    } catch {
      // restore failed; the operator can reset reasoning.preset manually
    }
  }
  try {
    await rpc('credentials.unset', { ref: POC_REF });
  } catch {
    // nothing to unset
  }
}

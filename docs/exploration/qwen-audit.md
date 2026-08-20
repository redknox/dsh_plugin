# Issue #18 — Qwen-specific assumption audit

Audit of Qwen-specific assumptions in the adapter and their isolation behind
explicit compatibility/capability seams. End state: `LlamacppAdapter` owns
Harness ↔ llama.cpp semantics; model-family behavior (Qwen today, other
families later) lives behind `ModelFamilyProfile`.

## Resolution precedence (enforced)

```
explicit configuration → server/provider capability metadata → model-family
compatibility profile → model-name heuristic (never used today)
```

**Explicit beats heuristic; unknown != false.** There is deliberately **no
model-name heuristic**: `modelFamily: 'auto'` (default) resolves to the
`unknown` profile, and a model string that looks like Qwen does not flip the
profile.

## Inventory of Qwen references

| Location | Reference | Classification | Leaks into public adapter semantics? |
|---|---|---|---|
| `src/reasoning.ts` (comments L2/11/43/74) | `preserve_thinking` described as a Qwen chat-template kwarg | documentation | no — the behavior is semantic (`preserveThinking` is an explicit preset/expert knob) |
| `src/protocol.ts` (comments L9/74/105) | `enable_thinking` / `preserve_thinking` wire kwargs, reasoning deltas | documentation (wire fields are llama.cpp-generic) | no |
| `src/config.ts` L39 `DEFAULT_MODEL = 'qwen3'` | default model id | default value (validated family; user-overridable) | no — a default, not a behavioral assumption |
| `src/config.ts` (comments L167/196) | "Semantic Qwen reasoning controls" | documentation | no |
| `src/serialize.ts` `applyReasoningToRequest` | `chat_template_kwargs.enable_thinking` / `preserve_thinking` | **llama.cpp-generic template-kwargs mechanism** (llama.cpp forwards kwargs to any template), plus Qwen-validated defaults | only via explicit `reasoning.wire` / `preserveThinking` configuration |
| `src/serialize.ts` `thinking_budget_tokens` / `reasoning_effort` | runtime reasoning controls | llama.cpp-generic | no |
| `src/discovery.ts` `supportOf` | `supportsTools` / `supportsReasoning` only set from explicit capability markers | capability-dependent (absence = unknown) | no — already "unknown != false" |
| `src/routing.ts` | capability filtering/ordering | llama.cpp-generic | no |
| `src/compat.ts` (new, issue #18) | `QWEN_PROFILE` / `UNKNOWN_PROFILE` / `familyProfileFor` | model-family compatibility profile seam | no — explicit selector only |
| tests fixtures (`qwen3` model ids) | fixture naming | test data | no — but see the new non-Qwen generic-path tests |

## What the audit found

1. **No model-name heuristic anywhere** — the adapter never branches on the
   model string.
2. **No silent Qwen inheritance** — capabilities absent from discovery stay
   unknown; reasoning wire translation is driven by explicit configuration
   (`reasoning.wire`, `reasoning.expert`, presets).
3. The Qwen associations that remain are (a) documentation of which wire
   fields Qwen's chat template consumes, and (b) the validated default
   (`chat-template-kwargs` wire, `qwen3` default model) — both explicit and
   user-overridable.
4. The only behavioral surface that *could* be family-sensitive is the wire
   translation (`serialize.applyReasoningToRequest`); it is now explicitly
   tied to the family profile's default and to user configuration.

## The seam (src/compat.ts)

```text
Harness reasoning semantics
        ↓
llama.cpp provider capability/wire layer
        ↓
model-family compatibility profile        ← ModelFamilyProfile
        ├── Qwen        (validated; supportsThinkingKwargs: true → wire default
        │                chat-template-kwargs)
        └── unknown     (default; supportsThinkingKwargs: undefined → wire
                         default 'none' — no Qwen kwargs sent)
```

- `familyProfileFor(id)` — explicit selector (`'auto' | 'qwen'`); `'auto'`
  and anything unrecognized → `UNKNOWN_PROFILE`.
- **Behaviorally effective**: `defaultReasoningWire(profile)` is driven by
  `supportsThinkingKwargs` — only a profile that declares its template
  handles the Qwen kwargs defaults to sending them; unknown (`undefined`) or
  unsupported (`false`) → `'none'`. `ReasoningWireMode` gained `'none'`, and
  `serialize.applyReasoningToRequest` sends no reasoning wire fields for it.
- Explicit `reasoning.wire` configuration always wins (resolved in
  `resolveAdapterOptions`); `reasoning-fields` stays available to unknown
  families on explicit configuration (llama.cpp-native fields, not
  Qwen-oriented).
- A future family (Llama/Gemma/DeepSeek/…) adds a profile (declaring
  `supportsThinkingKwargs` as appropriate) and an explicit selector instead
  of editing the adapter.

## Acceptance mapping

1. ✅ Core adapter does not rely on undocumented Qwen assumptions for generic
   operation — proven by the non-Qwen model-name tests
   (`tests/compat.test.ts`: serialize / listModels / resolveModel / stream
   with `granite-3.3`).
2. ✅ Qwen-specific reasoning/template behavior is identifiable
   (`src/compat.ts` QWEN_PROFILE + `src/serialize.ts` wire layer comments)
   and isolated behind the profile seam.
3. ✅ Unknown/non-Qwen models receive no Qwen-only wire parameters by default
   — unknown family resolves to `reasoning.wire: 'none'`; `enable_thinking`
   / `preserve_thinking` are absent unless (a) explicitly configured
   (`reasoning.wire` / `preserveThinking`), (b) capability metadata says the
   template supports them, or (c) an explicit family profile (Qwen) declares
   support (`supportsThinkingKwargs: true`). Tests assert absence for the
   default unknown path and presence for the explicit/profile paths.
4. ✅ Existing Qwen behavior preserved when explicitly selected — the Qwen
   profile keeps `chat-template-kwargs` (test fixtures default to
   `modelFamily: 'qwen'`; the web profile and example scripts declare it
   explicitly; e2e against the real Qwen server shows
   `thinking_budget_tokens: 4096` still sent). 229 tests green.
5. ✅ The architecture is explainable to upstream: a llama.cpp provider with
   optional model-family compatibility profiles.

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
        ├── Qwen        (validated; supportsThinkingKwargs: true)
        └── unknown     (default; inherits nothing Qwen-specific)
```

- `familyProfileFor(id)` — explicit selector (`'auto' | 'qwen'`); `'auto'`
  and anything unrecognized → `UNKNOWN_PROFILE`.
- `UNKNOWN_PROFILE.reasoning.supportsThinkingKwargs` is **undefined**
  (unknown), never assumed true.
- `defaultReasoningWire(profile)` — the family's default wire mode; explicit
  `reasoning.wire` always wins (resolved in `resolveAdapterOptions`).
- A future family (Llama/Gemma/DeepSeek/…) adds a profile and an explicit
  selector instead of editing the adapter.

## Acceptance mapping

1. ✅ Core adapter does not rely on undocumented Qwen assumptions for generic
   operation — proven by the non-Qwen model-name tests
   (`tests/compat.test.ts`: serialize / listModels / resolveModel / stream
   with `granite-3.3`).
2. ✅ Qwen-specific reasoning/template behavior is identifiable
   (`src/compat.ts` QWEN_PROFILE + `src/serialize.ts` wire layer comments)
   and isolated behind the profile seam.
3. ✅ Unknown/non-Qwen models receive no Qwen-only wire parameters unless
   explicitly configured — `preserve_thinking` absent for unknown families by
   default; explicit configuration opts in.
4. ✅ Existing Qwen behavior and tests unchanged (218 + 8 new green; default
   resolution regression-guarded).
5. ✅ The architecture is explainable to upstream: a llama.cpp provider with
   optional model-family compatibility profiles.

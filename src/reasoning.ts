/**
 * Qwen reasoning/thinking controls: semantic presets and policy resolution.
 *
 * Implemented in issue #4 (presets + expert overrides) and extended by issue
 * #6 (adaptive budget policy). Kept isolated from transport (`client.ts`) and
 * Harness core logic; only the adapter/request-builder layer translates a
 * resolved policy into llama.cpp request fields.
 *
 * @module llm-llamacpp/reasoning
 */

/**
 * The `LlmAdapter` implementation bridging DeepSeek Harness and the llama.cpp
 * transport client.
 *
 * The adapter owns Harness-specific translation only: `GenerateOptions` into
 * llama.cpp request bodies and llama.cpp streamed responses back into Harness
 * `StreamChunk` values. Transport stays in `client.ts`; reasoning policy stays
 * in `reasoning.ts`. No package-internal agent-loop code is imported.
 *
 * Implemented in issue #3. Until then `stream()` is a placeholder that fails
 * explicitly, so the scaffold is loadable and registered while nothing is
 * silently half-wired.
 *
 * @module llm-llamacpp/adapter
 */
import {
  LlmAdapter,
  LlmError,
  type LlmModelInfo,
  type LlmProviderInfo,
  type LlmResolvedModelInfo,
  type GenerateOptions,
  type StreamChunk,
} from '@deepseek-ai/dsh-llm';
import type { ResolvedAdapterOptions } from './config.ts';

/** The adapter's dependency surface, injected by the registering plugin. */
export interface LlamacppAdapterDeps {
  /** Per-operation re-read of validated connection facts. */
  readonly options: () => ResolvedAdapterOptions;
  /** Per-request API key resolution; `undefined` means no auth header. */
  readonly resolveApiKey: () => Promise<string | undefined>;
}

/**
 * Placeholder adapter for the `llamacpp-local` provider route. One instance
 * serves every model name: the harness model id IS the wire model id.
 */
export class LlamacppAdapter extends LlmAdapter {
  readonly deps: LlamacppAdapterDeps;

  constructor(deps: LlamacppAdapterDeps) {
    super();
    this.deps = deps;
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: this.deps.options().providerName };
  }

  override providerRetryPolicy(): ReturnType<LlmAdapter['providerRetryPolicy']> {
    return this.deps.options().retryPolicy;
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const { model } = this.deps.options();
    return Promise.resolve([
      {
        provider,
        id: model,
        name: model,
        inputModalities: ['text'],
      },
    ]);
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      inputModalities: ['text'],
    });
  }

  override async *stream(_options: GenerateOptions): AsyncIterable<StreamChunk> {
    throw new LlmError(
      'llm-llamacpp: streaming not implemented yet (issue #2/#3)',
      'NOT_IMPLEMENTED',
    );
  }
}

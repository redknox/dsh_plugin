/**
 * Configuration validation tests for issue #1: `resolveAdapterOptions` is the
 * single explicit resolve step from raw config to validated connection facts,
 * so every default and every rejection lives here.
 */
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_API_KEY_HEADER,
  DEFAULT_BASE_URL,
  DEFAULT_MODEL,
  DEFAULT_PROVIDER_NAME,
  DEFAULT_STREAM_IDLE_TIMEOUT_MS,
  resolveAdapterOptions,
  type ConfigType,
} from '../src/config.ts';

describe('resolveAdapterOptions', () => {
  it('applies every default for an empty config', () => {
    const options = resolveAdapterOptions({});
    expect(options.baseURL).toBe(DEFAULT_BASE_URL);
    expect(options.providerName).toBe(DEFAULT_PROVIDER_NAME);
    expect(options.model).toBe(DEFAULT_MODEL);
    expect(options.apiKeyHeader).toBe(DEFAULT_API_KEY_HEADER);
    expect(options.apiKeyEnv).toBeUndefined();
    expect(options.streamIdleTimeoutMs).toBe(DEFAULT_STREAM_IDLE_TIMEOUT_MS);
    expect(options.reasoning).toMatchObject({ enabled: true, preset: 'medium', wire: 'chat-template-kwargs' });
  });

  it('passes through configured values', () => {
    const options = resolveAdapterOptions({
      baseURL: 'http://10.0.0.5:8081/',
      providerName: 'My Qwen Box',
      model: 'qwen3-14b',
      apiKeyEnv: 'MY_LLAMACPP_KEY',
      apiKeyHeader: 'x-api-key',
      streamIdleTimeoutMs: 60_000,
    });
    expect(options.baseURL).toBe('http://10.0.0.5:8081');
    expect(options.providerName).toBe('My Qwen Box');
    expect(options.model).toBe('qwen3-14b');
    expect(options.apiKeyEnv).toBe('MY_LLAMACPP_KEY');
    expect(options.apiKeyHeader).toBe('x-api-key');
    expect(options.streamIdleTimeoutMs).toBe(60_000);
  });

  it('rejects a baseURL that is not a URL', () => {
    expect(() => resolveAdapterOptions({ baseURL: 'not a url' })).toThrow(/baseURL/);
    expect(() => resolveAdapterOptions({ baseURL: '127.0.0.1:8080' })).toThrow(/baseURL/);
  });

  it('rejects a baseURL with a non-http scheme', () => {
    expect(() => resolveAdapterOptions({ baseURL: 'ftp://example.com' })).toThrow(/http or https/);
    expect(() => resolveAdapterOptions({ baseURL: 'file:///tmp/model' })).toThrow(/http or https/);
  });

  it('normalizes a trailing slash on baseURL', () => {
    expect(resolveAdapterOptions({ baseURL: 'http://127.0.0.1:8080///' }).baseURL).toBe('http://127.0.0.1:8080');
  });

  it('rejects an empty model id', () => {
    expect(() => resolveAdapterOptions({ model: '   ' })).toThrow(/model/);
  });

  it('rejects an empty provider name', () => {
    expect(() => resolveAdapterOptions({ providerName: '' })).toThrow(/providerName/);
  });

  it('rejects a non-positive streamIdleTimeoutMs', () => {
    expect(() => resolveAdapterOptions({ streamIdleTimeoutMs: 0 })).toThrow(/streamIdleTimeoutMs/);
    expect(() => resolveAdapterOptions({ streamIdleTimeoutMs: Number.NaN })).toThrow(/streamIdleTimeoutMs/);
  });

  it('omits an apiKeyEnv that is empty or whitespace', () => {
    expect(resolveAdapterOptions({ apiKeyEnv: '' }).apiKeyEnv).toBeUndefined();
    expect(resolveAdapterOptions({ apiKeyEnv: '   ' }).apiKeyEnv).toBeUndefined();
  });

  it('keeps a single endpoint when only baseURL is set (issue #7)', () => {
    const options = resolveAdapterOptions({ baseURL: 'http://127.0.0.1:9999' });
    expect(options.endpoints).toEqual(['http://127.0.0.1:9999']);
    expect(options.baseURL).toBe('http://127.0.0.1:9999');
  });

  it('resolves ordered fallback endpoints and dedupes them (issue #7)', () => {
    const options = resolveAdapterOptions({
      endpoints: ['http://10.0.0.1:8080', 'http://10.0.0.2:8080', 'http://10.0.0.1:8080/'],
    });
    expect(options.endpoints).toEqual(['http://10.0.0.1:8080', 'http://10.0.0.2:8080']);
    expect(options.baseURL).toBe('http://10.0.0.1:8080');
  });

  it('resolves endpoint capability profiles (issue #9)', () => {
    const options = resolveAdapterOptions({
      endpoints: [
        'http://10.0.0.1:8080',
        { url: 'http://10.0.0.2:8080', capabilities: { models: ['qwen3'], contextWindow: 32768, tools: true, reasoning: true, workload: ['chat'] } },
      ],
    });
    expect(options.endpoints).toEqual(['http://10.0.0.1:8080', 'http://10.0.0.2:8080']);
    expect(options.endpointProfiles).toEqual([
      { baseURL: 'http://10.0.0.1:8080' },
      { baseURL: 'http://10.0.0.2:8080', capabilities: { models: ['qwen3'], contextWindow: 32768, tools: true, reasoning: true, workload: ['chat'] } },
    ]);
  });

  it('rejects invalid endpoint capability config (issue #9)', () => {
    expect(() => resolveAdapterOptions({ endpoints: [{ url: 'http://a', capabilities: { contextWindow: 0 } }] }))
      .toThrow(/contextWindow/);
    expect(() => resolveAdapterOptions({ endpoints: [{ url: 'not-a-url' }] })).toThrow(/URL/);
  });

  it('rejects an invalid fallback endpoint (issue #7)', () => {
    expect(() => resolveAdapterOptions({ endpoints: ['not-a-url'] })).toThrow(/URL/);
    expect(() => resolveAdapterOptions({ endpoints: ['ftp://x'] })).toThrow(/http or https/);
  });

  it('validates requestTimeoutMs (issue #7)', () => {
    expect(resolveAdapterOptions({ requestTimeoutMs: 60_000 }).requestTimeoutMs).toBe(60_000);
    expect(() => resolveAdapterOptions({ requestTimeoutMs: 0 })).toThrow(/requestTimeoutMs/);
  });

  it('resolves the provider-owned retry policy (issue #7)', () => {
    const always = resolveAdapterOptions({ retryPolicy: { mode: 'always' } });
    expect(always.retryPolicy.mode).toBe('always');
    const normal = resolveAdapterOptions({ retryPolicy: { mode: 'normal', maxRetries: 3, retryableCodes: ['SERVER'] } });
    expect(normal.retryPolicy.mode).toBe('normal');
    if (normal.retryPolicy.mode === 'normal') {
      expect(normal.retryPolicy.maxRetries).toBe(3);
      expect(normal.retryPolicy.retryableCodes).toEqual(['SERVER']);
    }
  });
});

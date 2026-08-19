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
});

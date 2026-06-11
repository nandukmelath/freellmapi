import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { initDb } from '../../db/index.js';
import {
  computeCacheKey,
  getCachedResponse,
  storeCachedResponse,
  getCacheStats,
  clearCache,
  isCacheableTemperature,
  parseCacheDirective,
  cacheActive,
  isCacheEnabled,
} from '../../services/cache.js';
import type { ChatMessage } from '@freellmapi/shared/types.js';

const CACHE_ENV = [
  'RESPONSE_CACHE',
  'RESPONSE_CACHE_TTL_SECONDS',
  'RESPONSE_CACHE_MAX_TEMPERATURE',
  'RESPONSE_CACHE_MAX_ENTRIES',
] as const;

function msg(role: ChatMessage['role'], content: string): ChatMessage {
  return { role, content } as ChatMessage;
}

const sampleBody = (text: string) => ({
  id: 'chatcmpl-test',
  object: 'chat.completion',
  choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
});

const store = (key: string, text: string, now?: number) =>
  storeCachedResponse(key, {
    body: sampleBody(text),
    platform: 'groq',
    modelId: 'llama-3.3-70b',
    keyId: 7,
    promptTokens: 10,
    completionTokens: 5,
  }, now);

describe('response cache', () => {
  const saved: Record<string, string | undefined> = {};

  beforeEach(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    for (const k of CACHE_ENV) saved[k] = process.env[k];
    // Fresh in-memory DB per test → isolated response_cache table.
    initDb(':memory:');
  });

  afterEach(() => {
    for (const k of CACHE_ENV) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k]!;
    }
  });

  describe('computeCacheKey', () => {
    it('is stable regardless of object key order', () => {
      const messages = [msg('user', 'hello')];
      const a = computeCacheKey({ model: 'auto', messages, temperature: 0.2, top_p: 1 });
      const b = computeCacheKey({ messages, top_p: 1, temperature: 0.2, model: 'auto' });
      expect(a).toBe(b);
    });

    it('treats omitted model and "auto" as the same bucket', () => {
      const messages = [msg('user', 'hello')];
      expect(computeCacheKey({ model: undefined, messages }))
        .toBe(computeCacheKey({ model: 'auto', messages }));
    });

    it('changes when the prompt changes', () => {
      expect(computeCacheKey({ model: 'auto', messages: [msg('user', 'a')] }))
        .not.toBe(computeCacheKey({ model: 'auto', messages: [msg('user', 'b')] }));
    });

    it('changes when sampling params change', () => {
      const messages = [msg('user', 'hello')];
      expect(computeCacheKey({ model: 'auto', messages, temperature: 0 }))
        .not.toBe(computeCacheKey({ model: 'auto', messages, temperature: 0.9 }));
    });

    it('distinguishes a pinned model from auto', () => {
      const messages = [msg('user', 'hello')];
      expect(computeCacheKey({ model: 'gpt-oss-120b', messages }))
        .not.toBe(computeCacheKey({ model: 'auto', messages }));
    });
  });

  describe('store / get round-trip', () => {
    it('returns null on a miss', () => {
      expect(getCachedResponse('does-not-exist')).toBeNull();
    });

    it('returns the stored body on a hit', () => {
      const key = computeCacheKey({ model: 'auto', messages: [msg('user', 'hi')] });
      store(key, 'cached answer');
      const hit = getCachedResponse(key);
      expect(hit).not.toBeNull();
      expect((hit!.body as any).choices[0].message.content).toBe('cached answer');
      expect(hit!.platform).toBe('groq');
      expect(hit!.modelId).toBe('llama-3.3-70b');
      expect(hit!.keyId).toBe(7);
      expect(hit!.promptTokens).toBe(10);
      expect(hit!.completionTokens).toBe(5);
    });

    it('overwrites and refreshes an existing entry', () => {
      const key = computeCacheKey({ model: 'auto', messages: [msg('user', 'hi')] });
      store(key, 'first');
      store(key, 'second');
      expect((getCachedResponse(key)!.body as any).choices[0].message.content).toBe('second');
      expect(getCacheStats().entries).toBe(1);
    });
  });

  describe('TTL expiry', () => {
    it('treats an entry older than the TTL as a miss and deletes it', () => {
      process.env.RESPONSE_CACHE_TTL_SECONDS = '1';
      const key = computeCacheKey({ model: 'auto', messages: [msg('user', 'hi')] });
      const t0 = 1_000_000;
      store(key, 'stale', t0);
      // 2s later, past the 1s TTL → miss.
      expect(getCachedResponse(key, t0 + 2000)).toBeNull();
      // The expired row was purged.
      expect(getCacheStats().entries).toBe(0);
    });

    it('serves an entry still within the TTL', () => {
      process.env.RESPONSE_CACHE_TTL_SECONDS = '60';
      const key = computeCacheKey({ model: 'auto', messages: [msg('user', 'hi')] });
      const t0 = 1_000_000;
      store(key, 'fresh', t0);
      expect(getCachedResponse(key, t0 + 30_000)).not.toBeNull();
    });
  });

  describe('hit accounting', () => {
    it('counts hits and tallies saved tokens', () => {
      const key = computeCacheKey({ model: 'auto', messages: [msg('user', 'hi')] });
      store(key, 'answer');
      getCachedResponse(key);
      getCachedResponse(key);
      getCachedResponse(key);
      const stats = getCacheStats();
      expect(stats.entries).toBe(1);
      expect(stats.totalHits).toBe(3);
      expect(stats.savedPromptTokens).toBe(30); // 3 hits × 10
      expect(stats.savedCompletionTokens).toBe(15); // 3 hits × 5
    });
  });

  describe('entry cap eviction', () => {
    it('evicts the oldest entries past RESPONSE_CACHE_MAX_ENTRIES', () => {
      process.env.RESPONSE_CACHE_MAX_ENTRIES = '3';
      // Realistic, increasing timestamps so reads stay inside the default TTL —
      // the point under test is eviction by count, not TTL expiry.
      const base = 1_700_000_000_000;
      const keys = ['a', 'b', 'c', 'd', 'e'].map(t =>
        computeCacheKey({ model: 'auto', messages: [msg('user', t)] }));
      keys.forEach((k, i) => store(k, `v${i}`, base + i)); // increasing created_at_ms

      expect(getCacheStats().entries).toBe(3);
      // Two oldest gone, three newest survive.
      expect(getCachedResponse(keys[0], base + 10)).toBeNull();
      expect(getCachedResponse(keys[1], base + 10)).toBeNull();
      expect(getCachedResponse(keys[4], base + 10)).not.toBeNull();
    });
  });

  describe('clearCache', () => {
    it('removes every entry', () => {
      store(computeCacheKey({ model: 'auto', messages: [msg('user', 'x')] }), 'x');
      store(computeCacheKey({ model: 'auto', messages: [msg('user', 'y')] }), 'y');
      expect(clearCache()).toBe(2);
      expect(getCacheStats().entries).toBe(0);
    });
  });

  describe('temperature guard', () => {
    it('caches omitted temperature', () => {
      expect(isCacheableTemperature(undefined)).toBe(true);
    });
    it('respects RESPONSE_CACHE_MAX_TEMPERATURE', () => {
      process.env.RESPONSE_CACHE_MAX_TEMPERATURE = '0.5';
      expect(isCacheableTemperature(0.2)).toBe(true);
      expect(isCacheableTemperature(0.5)).toBe(true);
      expect(isCacheableTemperature(0.9)).toBe(false);
    });
  });

  describe('directive parsing', () => {
    it('parses off/on aliases', () => {
      expect(parseCacheDirective('off')).toBe('off');
      expect(parseCacheDirective('bypass')).toBe('off');
      expect(parseCacheDirective('on')).toBe('on');
      expect(parseCacheDirective('force')).toBe('on');
      expect(parseCacheDirective(undefined)).toBe('default');
    });

    it('treats Cache-Control: no-store as off', () => {
      expect(parseCacheDirective(undefined, 'no-store')).toBe('off');
    });

    it('takes the first value of a repeated header', () => {
      expect(parseCacheDirective(['off', 'on'])).toBe('off');
    });
  });

  describe('cacheActive', () => {
    it('off directive always wins; on directive forces; default follows env', () => {
      delete process.env.RESPONSE_CACHE;
      expect(isCacheEnabled()).toBe(false);
      expect(cacheActive('off')).toBe(false);
      expect(cacheActive('on')).toBe(true);
      expect(cacheActive('default')).toBe(false);
      process.env.RESPONSE_CACHE = 'on';
      expect(cacheActive('default')).toBe(true);
      expect(cacheActive('off')).toBe(false);
    });
  });
});

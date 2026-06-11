import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import type { Express } from 'express';
import { createApp } from '../../app.js';
import { initDb, getDb, getUnifiedApiKey } from '../../db/index.js';
import { mintDashboardToken, isGatedApiPath } from '../helpers/auth.js';

let dashToken = '';

async function request(app: Express, method: string, path: string, body?: any, headers: Record<string, string> = {}) {
  const server = app.listen(0);
  const addr = server.address() as any;
  const url = `http://127.0.0.1:${addr.port}${path}`;
  const res = await fetch(url, {
    method,
    headers: {
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...(isGatedApiPath(path) && !('Authorization' in headers) ? { Authorization: `Bearer ${dashToken}` } : {}),
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const raw = await res.text();
  server.close();
  let json: any = null;
  try { json = JSON.parse(raw); } catch {}
  return { status: res.status, body: json, headers: res.headers, raw };
}

function authHeaders() {
  return { Authorization: `Bearer ${getUnifiedApiKey()}` };
}

// Mock every groq chat-completion call, counting how many actually reach the
// provider. The whole point of the cache is that a repeat request does NOT.
function mockGroq(content: string) {
  const origFetch = global.fetch;
  const counter = { calls: 0 };
  vi.spyOn(global, 'fetch').mockImplementation(async (url, init) => {
    const urlStr = typeof url === 'string' ? url : url.toString();
    if (urlStr.includes('api.groq.com/openai/v1/chat/completions')) {
      counter.calls++;
      return {
        ok: true,
        json: () => Promise.resolve({
          id: 'chatcmpl-cache-it',
          object: 'chat.completion',
          created: 123,
          model: 'openai/gpt-oss-120b',
          choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        }),
      } as any;
    }
    return origFetch(url, init);
  });
  return counter;
}

describe('Response cache (proxy integration)', () => {
  let app: Express;

  beforeAll(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    process.env.RESPONSE_CACHE = 'on';
    initDb(':memory:');
    app = createApp();
    dashToken = mintDashboardToken();
  });

  afterAll(() => {
    delete process.env.RESPONSE_CACHE;
  });

  beforeEach(async () => {
    const db = getDb();
    db.prepare('DELETE FROM api_keys').run();
    db.prepare('DELETE FROM requests').run();
    db.prepare('DELETE FROM response_cache').run();
    const addKey = await request(app, 'POST', '/api/keys', { platform: 'groq', key: 'gsk_cache_test', label: 'cache' });
    expect(addKey.status).toBe(201);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const chat = (extra: Record<string, unknown> = {}, headers: Record<string, string> = {}) =>
    request(app, 'POST', '/v1/chat/completions', {
      model: 'auto',
      messages: [{ role: 'user', content: 'what is 2+2?' }],
      ...extra,
    }, { ...authHeaders(), ...headers });

  it('serves an identical second request from cache without calling the provider', async () => {
    const counter = mockGroq('the answer is four');

    const first = await chat();
    expect(first.status).toBe(200);
    expect(first.body.choices[0].message.content).toBe('the answer is four');
    expect(first.headers.get('x-freellm-cache')).toBe('MISS');
    expect(counter.calls).toBe(1);

    const second = await chat();
    expect(second.status).toBe(200);
    expect(second.body.choices[0].message.content).toBe('the answer is four');
    expect(second.headers.get('x-freellm-cache')).toBe('HIT');
    // The provider was NOT hit again — the whole point.
    expect(counter.calls).toBe(1);
  });

  it('does not record rate-limit usage for a cache hit (quota preserved)', async () => {
    mockGroq('four');
    await chat();
    const usageAfterMiss = (getDb().prepare("SELECT COUNT(*) AS c FROM rate_limit_usage WHERE kind='request'").get() as { c: number }).c;
    await chat(); // hit
    const usageAfterHit = (getDb().prepare("SELECT COUNT(*) AS c FROM rate_limit_usage WHERE kind='request'").get() as { c: number }).c;
    expect(usageAfterHit).toBe(usageAfterMiss); // no new provider request recorded
  });

  it('treats a different prompt as a miss and calls the provider again', async () => {
    const counter = mockGroq('depends on the prompt');
    await chat({ messages: [{ role: 'user', content: 'first question' }] });
    await chat({ messages: [{ role: 'user', content: 'second question' }] });
    expect(counter.calls).toBe(2);
  });

  it('X-FreeLLM-Cache: off bypasses the cache on both read and write', async () => {
    const counter = mockGroq('bypassed');
    await chat({}, { 'X-FreeLLM-Cache': 'off' });
    await chat({}, { 'X-FreeLLM-Cache': 'off' });
    // Neither call read nor populated the cache → provider hit twice.
    expect(counter.calls).toBe(2);
  });

  it('does not cache a high-temperature request', async () => {
    process.env.RESPONSE_CACHE_MAX_TEMPERATURE = '0.2';
    try {
      const counter = mockGroq('varies');
      await chat({ temperature: 0.9 });
      await chat({ temperature: 0.9 });
      expect(counter.calls).toBe(2); // above the cap → never cached
    } finally {
      delete process.env.RESPONSE_CACHE_MAX_TEMPERATURE;
    }
  });
});

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import type { Express } from 'express';

// Mock embeddings so the semantic path is deterministic and offline: every
// prompt maps to the SAME vector, so two DIFFERENT prompts (distinct exact
// keys) collide in the semantic layer — exactly what we want to prove.
vi.mock('../../services/embeddings.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../services/embeddings.js')>();
  return {
    ...actual,
    runEmbeddings: vi.fn(async () => ({
      vectors: [[1, 0, 0]],
      family: 'test-embed',
      platform: 'test',
      modelId: 'test',
      dimensions: 3,
      inputTokens: 3,
    })),
  };
});

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
          id: 'chatcmpl-sem',
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

describe('Semantic response cache (proxy integration)', () => {
  let app: Express;

  beforeAll(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    process.env.RESPONSE_CACHE = 'on';
    process.env.RESPONSE_CACHE_SEMANTIC = 'on';
    process.env.RESPONSE_CACHE_SEMANTIC_THRESHOLD = '0.9';
    initDb(':memory:');
    app = createApp();
    dashToken = mintDashboardToken();
  });

  afterAll(() => {
    delete process.env.RESPONSE_CACHE;
    delete process.env.RESPONSE_CACHE_SEMANTIC;
    delete process.env.RESPONSE_CACHE_SEMANTIC_THRESHOLD;
  });

  beforeEach(async () => {
    const db = getDb();
    db.prepare('DELETE FROM api_keys').run();
    db.prepare('DELETE FROM requests').run();
    db.prepare('DELETE FROM response_cache').run();
    const addKey = await request(app, 'POST', '/api/keys', { platform: 'groq', key: 'gsk_sem_test', label: 'sem' });
    expect(addKey.status).toBe(201);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const chat = (content: string) =>
    request(app, 'POST', '/v1/chat/completions', {
      model: 'auto',
      messages: [{ role: 'user', content }],
    }, authHeaders());

  it('serves a DIFFERENTLY-worded prompt from the semantic cache', async () => {
    const counter = mockGroq('four');

    // First prompt → exact + semantic miss → provider called, answer stored.
    const first = await chat('what is 2+2?');
    expect(first.status).toBe(200);
    expect(first.headers.get('x-freellm-cache')).toBe('MISS');
    expect(counter.calls).toBe(1);

    // Different wording → exact MISS, but semantic match (same mock vector).
    const second = await chat('hey, can you tell me two plus two?');
    expect(second.status).toBe(200);
    expect(second.body.choices[0].message.content).toBe('four');
    expect(second.headers.get('x-freellm-cache')).toBe('HIT-SEMANTIC');
    // Provider NOT called again.
    expect(counter.calls).toBe(1);
  });

  it('still bypasses the semantic layer for tool requests', async () => {
    const counter = mockGroq('tooly');
    const toolReq = (content: string) =>
      request(app, 'POST', '/v1/chat/completions', {
        model: 'auto',
        messages: [{ role: 'user', content }],
        tools: [{ type: 'function', function: { name: 'noop', description: 'x', parameters: { type: 'object', properties: {} } } }],
      }, authHeaders());

    await toolReq('do a thing');
    await toolReq('do a different thing');
    // Tool turns are excluded from semantic matching → both hit the provider.
    expect(counter.calls).toBe(2);
  });
});

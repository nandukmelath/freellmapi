// Exact-match response cache.
//
// A free-tier-stacking proxy lives or dies by how far it stretches scarce
// quota. Re-asking a model the *same* prompt burns a free-tier slot for an
// answer we already have — pure waste. This cache stores successful completions
// keyed by a canonical hash of the request and serves an identical later
// request straight from SQLite, without touching any provider: zero quota cost,
// near-zero latency, and one fewer 429 on the way to the daily reset.
//
// Design choices that keep it SAFE:
//   - Exact match only. No embeddings / fuzzy matching — a near-miss must never
//     return a different prompt's answer. The key is a SHA-256 over the
//     canonicalized request, so a single token of difference is a miss.
//   - The key is the REQUEST, not the route. Any model's good answer to an
//     identical prompt is a valid hit, which is what maximizes the hit rate for
//     auto-routed traffic. platform/model_id/key_id are stored for analytics
//     attribution only.
//   - Opt-in. Off unless RESPONSE_CACHE is truthy, so existing installs see no
//     behavior change. A per-request header overrides either way.
//   - Temperature-gated. High-temperature requests are asking for variety;
//     replaying one frozen answer would defeat that. Cached only when the
//     temperature is omitted or <= RESPONSE_CACHE_MAX_TEMPERATURE.
//   - DB-absent safe. Every DB touch is wrapped; if the database is somehow
//     unavailable the cache silently disables rather than throwing in the proxy
//     hot path (mirrors services/ratelimit.ts).

import crypto from 'crypto';
import { getDb } from '../db/index.js';
import type { ChatMessage } from '@freellmapi/shared/types.js';

type CacheDb = ReturnType<typeof getDb>;

function withDb<T>(fn: (db: CacheDb) => T): T | undefined {
  try {
    return fn(getDb());
  } catch {
    return undefined;
  }
}

// ── Config (read from env each call so tests and the dashboard can toggle) ──

function envFlag(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  return /^(1|true|on|yes)$/i.test(raw.trim());
}

function envNum(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

/** Master switch. Default off so adopting the cache is an explicit choice. */
export function isCacheEnabled(): boolean {
  return envFlag('RESPONSE_CACHE', false);
}

/** Entry lifetime. Default 1h — long enough to absorb retries and agent
 *  re-runs, short enough that a refreshed catalog/key changes answers soon. */
export function cacheTtlMs(): number {
  return envNum('RESPONSE_CACHE_TTL_SECONDS', 3600) * 1000;
}

/** Above this temperature a request wants variety, so it is never cached.
 *  Default 1.0 caches everything when enabled (max quota savings); lower it to
 *  restrict caching to (near-)deterministic calls. */
export function cacheMaxTemperature(): number {
  return envNum('RESPONSE_CACHE_MAX_TEMPERATURE', 1.0);
}

/** Hard cap on stored entries; oldest are evicted past this. Bounds disk use. */
export function cacheMaxEntries(): number {
  return Math.floor(envNum('RESPONSE_CACHE_MAX_ENTRIES', 5000));
}

// A request is cacheable only when its temperature is omitted (caller accepts
// the provider default and is fine with a stable answer) or at/below the cap.
export function isCacheableTemperature(temperature?: number): boolean {
  if (temperature === undefined || temperature === null) return true;
  return temperature <= cacheMaxTemperature();
}

// ── Per-request directive ──
// `X-FreeLLM-Cache: off|on` (and the standard `Cache-Control: no-store`) let a
// caller override the global switch for one request — e.g. force a fresh
// generation, or opt a single call into caching on an otherwise cache-off install.
export type CacheDirective = 'default' | 'off' | 'on';

export function parseCacheDirective(
  header: string | string[] | undefined,
  cacheControl?: string | string[] | undefined,
): CacheDirective {
  const cc = (Array.isArray(cacheControl) ? cacheControl[0] : cacheControl)?.toLowerCase() ?? '';
  if (cc.includes('no-store') || cc.includes('no-cache')) return 'off';
  const raw = (Array.isArray(header) ? header[0] : header)?.trim().toLowerCase();
  if (!raw) return 'default';
  if (/^(off|no|0|false|bypass|skip)$/.test(raw)) return 'off';
  if (/^(on|yes|1|true|force)$/.test(raw)) return 'on';
  return 'default';
}

/** Resolve the global switch + per-request directive into a single yes/no. */
export function cacheActive(directive: CacheDirective): boolean {
  if (directive === 'off') return false;
  if (directive === 'on') return true;
  return isCacheEnabled();
}

// ── Canonical key ──

// Deterministic JSON: object keys sorted recursively and undefined dropped, so
// two requests that differ only in key order or omitted-vs-undefined fields
// hash identically. (JSON.stringify alone preserves insertion order, which
// varies between clients and would scatter otherwise-identical requests.)
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'null';
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  const obj = value as Record<string, unknown>;
  const parts = Object.keys(obj)
    .sort()
    .filter(k => obj[k] !== undefined)
    .map(k => `${JSON.stringify(k)}:${stableStringify(obj[k])}`);
  return `{${parts.join(',')}}`;
}

export interface CacheKeyInput {
  model: string | undefined; // the client's `model` field ('auto'/pinned/omitted)
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  tools?: unknown;
  tool_choice?: unknown;
}

export function computeCacheKey(input: CacheKeyInput): string {
  // Normalize the model field: omitted and the explicit "auto" sentinel mean the
  // same thing (let the router decide), so they must share a cache bucket.
  const model = !input.model || input.model === 'auto' ? 'auto' : input.model;
  const canonical = stableStringify({
    v: 1, // bump to invalidate every entry if the cached shape ever changes
    model,
    messages: input.messages,
    temperature: input.temperature,
    top_p: input.top_p,
    max_tokens: input.max_tokens,
    tools: input.tools,
    tool_choice: input.tool_choice,
  });
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

// ── Read / write ──

export interface CachedResponse {
  body: unknown; // the full OpenAI-shaped completion JSON, replayed verbatim
  platform: string;
  modelId: string;
  keyId: number | null;
  promptTokens: number;
  completionTokens: number;
}

interface CacheRow {
  response_json: string;
  platform: string;
  model_id: string;
  key_id: number | null;
  prompt_tokens: number;
  completion_tokens: number;
  created_at_ms: number;
}

/**
 * Look up a cached completion. Returns null on miss or when the entry has aged
 * past the TTL (expired rows are deleted lazily on read). A hit bumps the
 * hit_count / last_hit_at_ms columns for the analytics surface.
 */
export function getCachedResponse(cacheKey: string, now = Date.now()): CachedResponse | null {
  const row = withDb(db =>
    db.prepare(`
      SELECT response_json, platform, model_id, key_id, prompt_tokens, completion_tokens, created_at_ms
        FROM response_cache
       WHERE cache_key = ?
    `).get(cacheKey) as CacheRow | undefined,
  );
  if (!row) return null;

  if (now - row.created_at_ms > cacheTtlMs()) {
    withDb(db => db.prepare('DELETE FROM response_cache WHERE cache_key = ?').run(cacheKey));
    return null;
  }

  let body: unknown;
  try {
    body = JSON.parse(row.response_json);
  } catch {
    // Corrupt row — drop it and treat as a miss.
    withDb(db => db.prepare('DELETE FROM response_cache WHERE cache_key = ?').run(cacheKey));
    return null;
  }

  withDb(db =>
    db.prepare(`
      UPDATE response_cache
         SET hit_count = hit_count + 1, last_hit_at_ms = ?
       WHERE cache_key = ?
    `).run(now, cacheKey),
  );

  return {
    body,
    platform: row.platform,
    modelId: row.model_id,
    keyId: row.key_id,
    promptTokens: row.prompt_tokens,
    completionTokens: row.completion_tokens,
  };
}

export interface StoreInput {
  body: unknown;
  platform: string;
  modelId: string;
  keyId: number | null;
  promptTokens: number;
  completionTokens: number;
}

/**
 * Store a successful completion. Overwrites any existing entry for the key
 * (a re-generation refreshes the cached answer and its TTL). Enforces the
 * entry cap by evicting the oldest rows. Best-effort: failures are swallowed so
 * caching can never break a request that already succeeded.
 */
export function storeCachedResponse(cacheKey: string, input: StoreInput, now = Date.now()): void {
  let json: string;
  try {
    json = JSON.stringify(input.body);
  } catch {
    return; // unserializable body — skip silently
  }

  withDb(db => {
    db.prepare(`
      INSERT INTO response_cache
        (cache_key, response_json, platform, model_id, key_id, prompt_tokens, completion_tokens, hit_count, created_at_ms, last_hit_at_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, NULL)
      ON CONFLICT(cache_key) DO UPDATE SET
        response_json = excluded.response_json,
        platform = excluded.platform,
        model_id = excluded.model_id,
        key_id = excluded.key_id,
        prompt_tokens = excluded.prompt_tokens,
        completion_tokens = excluded.completion_tokens,
        created_at_ms = excluded.created_at_ms,
        hit_count = 0,
        last_hit_at_ms = NULL
    `).run(
      cacheKey, json, input.platform, input.modelId, input.keyId,
      input.promptTokens, input.completionTokens, now,
    );

    // Evict oldest beyond the cap. Cheap because the count only drifts by one
    // per insert, so at most one row is removed per call in steady state.
    const cap = cacheMaxEntries();
    const { cnt } = db.prepare('SELECT COUNT(*) AS cnt FROM response_cache').get() as { cnt: number };
    if (cnt > cap) {
      db.prepare(`
        DELETE FROM response_cache
         WHERE cache_key IN (
           SELECT cache_key FROM response_cache
            ORDER BY created_at_ms ASC
            LIMIT ?
         )
      `).run(cnt - cap);
    }
  });
}

export interface CacheStats {
  entries: number;
  totalHits: number;
  savedPromptTokens: number;
  savedCompletionTokens: number;
}

/**
 * Aggregate cache stats for the dashboard/analytics. "saved" tokens are the
 * provider tokens that hits avoided spending: hit_count × the entry's token
 * counts, summed — i.e. the free-tier quota the cache gave back.
 */
export function getCacheStats(): CacheStats {
  const row = withDb(db =>
    db.prepare(`
      SELECT
        COUNT(*) AS entries,
        COALESCE(SUM(hit_count), 0) AS totalHits,
        COALESCE(SUM(hit_count * prompt_tokens), 0) AS savedPromptTokens,
        COALESCE(SUM(hit_count * completion_tokens), 0) AS savedCompletionTokens
      FROM response_cache
    `).get() as CacheStats | undefined,
  );
  return row ?? { entries: 0, totalHits: 0, savedPromptTokens: 0, savedCompletionTokens: 0 };
}

/** Drop every cached entry. Returns the number removed. */
export function clearCache(): number {
  const res = withDb(db => db.prepare('DELETE FROM response_cache').run());
  return res?.changes ?? 0;
}

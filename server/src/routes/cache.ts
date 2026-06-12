import { Router } from 'express';
import type { Request, Response } from 'express';
import {
  getCacheStats,
  clearCache,
  isCacheEnabled,
  isSemanticEnabled,
  cacheTtlMs,
  cacheMaxEntries,
  semanticThreshold,
} from '../services/cache.js';

export const cacheRouter = Router();

// Cache status + savings for the dashboard. "saved" tokens are provider tokens
// that hits avoided spending — the free-tier quota the cache gave back.
cacheRouter.get('/stats', (_req: Request, res: Response) => {
  const stats = getCacheStats();
  res.json({
    enabled: isCacheEnabled(),
    semantic: isSemanticEnabled(),
    ttlSeconds: Math.round(cacheTtlMs() / 1000),
    maxEntries: cacheMaxEntries(),
    semanticThreshold: semanticThreshold(),
    ...stats,
    savedTokens: stats.savedPromptTokens + stats.savedCompletionTokens,
  });
});

// Flush the cache (e.g. after changing keys/models, or to force fresh answers).
cacheRouter.delete('/', (_req: Request, res: Response) => {
  const removed = clearCache();
  res.json({ cleared: removed });
});

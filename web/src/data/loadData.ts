// ─────────────────────────────────────────────────────────────
// Data loading: injects the generated /data/*.js files exactly
// like the legacy monolith (sequential <script> tags with
// cache-busting), with the same per-file fallbacks:
//   tracker missing  → TRACKER_PICKS=[] / TRACKER_METRICS=null
//   backtest missing → BACKTEST_DATA=null
//   results missing  → RESOLVED_RESULTS={}
// predictions.js missing → throws (App shows an empty state).
// ─────────────────────────────────────────────────────────────
import type {
  BacktestData,
  MatchPred,
  PredictionsMeta,
  ResolvedResult,
  TrackerMetrics,
  TrackerPick,
} from '../types'

declare global {
  interface Window {
    TRACKER_PICKS?: TrackerPick[]
    TRACKER_METRICS?: TrackerMetrics | null
    BACKTEST_DATA?: BacktestData | null
    RESOLVED_RESULTS?: Record<string, ResolvedResult>
  }
  // predictions.js declares these with top-level `const` (global
  // lexical bindings, shared with module scripts).
  const ALL_MATCHES: MatchPred[]
  const DATES: string[]
  const DATE_LABELS: Record<string, string>
  const PREDICTIONS_META: PredictionsMeta | undefined
}

export interface AppData {
  matches: MatchPred[]
  dates: string[]
  dateLabels: Record<string, string>
  meta: PredictionsMeta
  trackerPicks: TrackerPick[]
  trackerMetrics: TrackerMetrics | null
  backtest: BacktestData | null
  results: Record<string, ResolvedResult>
}

function injectScript(src: string): Promise<boolean> {
  return new Promise((resolve) => {
    const s = document.createElement('script')
    s.src = src
    s.async = false
    s.onload = () => {
      s.remove()
      resolve(true)
    }
    s.onerror = () => {
      s.remove()
      resolve(false)
    }
    document.head.appendChild(s)
  })
}

let _loaded: AppData | null = null

export async function loadData(): Promise<AppData> {
  if (_loaded) return _loaded

  const v = '?v=' + Date.now()

  // Fallback defaults first — survive a 404 on any individual file.
  window.TRACKER_PICKS = []
  window.TRACKER_METRICS = null
  window.BACKTEST_DATA = null
  window.RESOLVED_RESULTS = {}

  await injectScript('/data/predictions.js' + v)
  await injectScript('/data/tracker_data.js' + v)
  await injectScript('/data/backtest_data.js' + v)
  await injectScript('/data/results.js' + v)

  if (typeof ALL_MATCHES === 'undefined' || typeof DATES === 'undefined') {
    throw new Error('predictions.js not found — run `python main.py` first')
  }

  _loaded = {
    matches: ALL_MATCHES,
    dates: DATES,
    dateLabels: typeof DATE_LABELS !== 'undefined' ? DATE_LABELS : {},
    meta:
      typeof PREDICTIONS_META !== 'undefined' && PREDICTIONS_META
        ? PREDICTIONS_META
        : { generatedAt: null, minStarsTracked: 3 },
    trackerPicks: window.TRACKER_PICKS ?? [],
    trackerMetrics: window.TRACKER_METRICS ?? null,
    backtest: window.BACKTEST_DATA ?? null,
    results: window.RESOLVED_RESULTS ?? {},
  }
  return _loaded
}

/**
 * 15 s HEAD poll on /data/predictions.js — full page reload when the
 * file changes (same behaviour as the monolith).
 * Returns a cleanup function.
 */
export function startAutoReload(): () => void {
  let lastTag: string | null = null
  const check = () => {
    fetch('/data/predictions.js', { method: 'HEAD', cache: 'no-store' })
      .then((r) => {
        const tag =
          r.headers.get('etag') || r.headers.get('last-modified') || r.headers.get('content-length')
        if (lastTag === null) {
          lastTag = tag
          return
        }
        if (tag && tag !== lastTag) location.reload()
      })
      .catch(() => {})
  }
  const id = setInterval(check, 15000)
  check()
  return () => clearInterval(id)
}

// ─────────────────────────────────────────────────────────────
// Data-layer types.
// MatchPred mirrors reporter.py::_prediction_to_dict exactly.
// TrackerPick mirrors tracker.py::_pick_to_js exactly.
// Storage shapes mirror the legacy visualizador/index.html
// localStorage records byte-for-byte (bwg_* keys).
// ─────────────────────────────────────────────────────────────

export type Outcome = 'home' | 'draw' | 'away' | 'over25' | 'btts'
export type Result1X2 = 'home' | 'draw' | 'away'

export interface SubProbs {
  p1: number
  pX: number
  p2: number
}

export interface ValueBet {
  outcome: Outcome
  edge: number // %
  bkOdds: number
  kelly: number // %
  modelProb: number // %
  impliedProb: number // %
  sharpMoney?: boolean
  bestBook?: string
  oddsMovement?: number | null
  pinnacleProb?: number | null
  clvVsPinnacle?: number | null
  sharpnessScore?: number
  mis?: number
  corrDiscount?: number
  kellyPortfolio?: number // %
}

export interface MatchPred {
  date: string
  home: string
  homeShort: string
  away: string
  awayShort: string
  league: string
  time: string | null
  stars: number
  prob1: number
  probX: number
  prob2: number
  over25: number
  over25yn: boolean
  btts: number
  bttsyn: boolean
  corners: number | null
  cornersOver: boolean | null
  cards: number | null
  cardsHome: number | null
  cardsAway: number | null
  cardsOver: boolean | null
  goalsHome: number | null
  goalsAway: number | null
  eloConf: number | null
  formHome: string | null
  formAway: string | null
  bestScore: string | null
  bestScoreProb: number
  fairOdds: number | null
  marketOdds: number | null
  h2h: string | null
  valueBets: ValueBet[] | null
  subModels: Record<string, SubProbs> | null
  eloHome: number | null
  eloAway: number | null
  fatigueHome: number | null
  fatigueAway: number | null
  homePos: number | null
  awayPos: number | null
  scoreGrid: number[][] | null
  aiNote?: string | null
  aiFactors?: string[] | null
  contextTags?: string[]
  // Monte Carlo secondary markets (absent in very old predictions.js)
  over15?: number
  over35?: number
  over45?: number
  bttsAndOver25?: number
  ahHomeMinus1Win?: number
  ahHomeMinus1Push?: number
  ahAwayPlus1Win?: number
}

export interface PredictionsMeta {
  generatedAt: string | null
  minStarsTracked: number
}

// ── tracker_data.js ──────────────────────────────────────────

export interface TrackerSubPred {
  ph: number
  pd: number
  pa: number
}

export interface TrackerPick {
  matchId: number
  runDate: string
  date: string
  home: string
  away: string
  league: string
  stars: number
  bestOutcome: Result1X2
  bestProb: number
  probHome: number
  probDraw: number
  probAway: number
  over25: number
  btts: number
  fairOdds: number | null
  marketOdds: number | null
  actualResult: Result1X2 | null
  actualOver25: number | boolean | null
  actualBtts: number | boolean | null
  subPreds: Record<string, TrackerSubPred> | null
}

export interface LeagueStats {
  n: number
  accuracy: number | null
  roi: number | null
  brier: number | null
  accuracy_over25: number | null
  accuracy_btts: number | null
}

export interface StarStats {
  n: number
  accuracy: number | null
  roi: number | null
}

export interface TagStats {
  n: number
  accuracy: number | null
  roi: number | null
  brier?: number | null
}

export interface PsiInfo {
  psi_max?: number | null
  drift_detected?: boolean
  ks_home?: number
  ks_draw?: number
  ks_away?: number
  n_recent?: number
  n_baseline?: number
}

export interface TimingWindow {
  n: number
  avg_clv: number
}

export type TimingLeague = {
  window_best_clv?: string
} & {
  [window: string]: TimingWindow | string | undefined
}

export interface SubmodelAcc {
  n: number
  accuracy: number
  brier?: number
}

export type SubmodelAccuracy30 = {
  n_window?: number
} & {
  [model: string]: SubmodelAcc | number | undefined
}

export interface BankrollPoint {
  date: string
  bankroll: number
}

export interface TrackerMetrics {
  n_resolved: number
  n_pending: number
  n_total: number
  accuracy_1x2: number | null
  brier_score: number | null
  roi_flat: number | null
  roi_kelly: number | null
  accuracy_over25: number | null
  accuracy_btts: number | null
  bankroll_history: BankrollPoint[]
  calibration_ready?: boolean
  n_for_calibration?: number
  vb_n?: number
  vb_accuracy?: number | null
  vb_roi_flat?: number | null
  per_league?: Record<string, LeagueStats>
  per_stars?: Record<string, StarStats>
  per_market?: Record<string, { accuracy: number | null; roi?: number | null }>
  per_tag?: Record<string, TagStats>
  avg_clv?: number | null
  avg_clv_by_league?: Record<string, number>
  n_clv_picks?: number
  hindsight_edge_by_league?: Record<string, number>
  hindsight_edge_by_stars?: Record<string, number>
  psi?: PsiInfo
  timing_analysis?: Record<string, TimingLeague>
  submodel_accuracy_30?: SubmodelAccuracy30
}

// ── backtest_data.js ─────────────────────────────────────────

export interface BacktestCurvePoint {
  date: string
  bankroll: number
  acc: number
  league?: string
}

export interface BacktestMetrics {
  n_matches: number
  accuracy_1x2?: number
  brier_score?: number
  log_loss?: number
  roi_flat?: number
  accuracy_over25?: number
  accuracy_btts?: number
  vb_n?: number
  vb_accuracy?: number | null
  vb_roi?: number | null
}

export interface BacktestData {
  league: string
  seasons: number[]
  generatedAt: string
  metrics: BacktestMetrics
  curve: BacktestCurvePoint[]
}

// ── results.js ───────────────────────────────────────────────

export interface ResolvedResult {
  result: 'H' | 'D' | 'A'
  over25: boolean | number
  btts: boolean | number
  league?: string
  matchDate?: string
}

// ── localStorage shapes (LEGACY-COMPATIBLE — do not change) ──

export interface Bankroll {
  initial: number
  current: number
  setAt: string
}

export interface SlipItem {
  id: string
  homeName: string
  awayName: string
  league: string
  matchDate: string
  outcome: Outcome
  oddsUser: number
  stake: number
  kellyStake: number | null
  addedAt: string
}

export interface ParlayLeg {
  homeName: string
  awayName: string
  league: string
  matchDate: string
  outcome: Outcome
  oddsUser: number
}

export type BetResult = 'won' | 'lost' | null

export interface HistorySimpleBet extends SlipItem {
  type: 'simple'
  placedAt: string
  settled: boolean
  result: BetResult
  pnl: number
  settledAt?: string
}

export interface HistoryParlayBet {
  id: string
  type: 'combinada'
  legs: ParlayLeg[]
  combinedOdds: number
  stake: number
  placedAt: string
  settled: boolean
  result: BetResult
  pnl: number
  settledAt?: string
}

export type HistoryBet = HistorySimpleBet | HistoryParlayBet

export interface NavStateStored {
  activeDate?: string
  activeLeague?: string
  sortMode?: string
}

export type RiskLevel = 'conservative' | 'balanced' | 'aggressive'

export interface AutopilotProfile {
  riskLevel: RiskLevel
  maxBetsPerDay: number
  leagues: string[]
  activeDates: string[]
  markets: string[]
  minOdds: number
  minStars: number
  maxExposurePct: number
}

export type SortMode = 'stars' | 'prob' | 'time'
export type SlipTab = 'active' | 'history' | 'combined'

export const VIEW_IDS = ['ALL', 'BEST', 'VALUE', 'TRACK', 'BACK', 'WIN', 'BETNOW', 'SLIP', 'AUTO'] as const
export type ViewId = (typeof VIEW_IDS)[number]

// ─────────────────────────────────────────────────────────────
// Pure betting logic — 1:1 ports from the monolith:
//   calcBestBets, getSuggestedParlays (V2 parlay engine),
//   getWiniela, selectAutopilotBets, sorting/day-score helpers.
// ─────────────────────────────────────────────────────────────
import type {
  AutopilotProfile,
  Bankroll,
  MatchPred,
  Outcome,
  SortMode,
  TrackerMetrics,
  ValueBet,
} from '../types'

export function winnerTeam(m: MatchPred): 'home' | 'draw' | 'away' {
  if (m.prob1 > m.prob2 && m.prob1 > m.probX) return 'home'
  if (m.prob2 > m.prob1 && m.prob2 > m.probX) return 'away'
  return 'draw'
}

export function bestWinProb(m: MatchPred): number {
  return Math.max(m.prob1, m.prob2)
}

export function dayScore(matches: MatchPred[], date: string): number {
  const top = matches.filter((m) => m.date === date && m.stars === 5)
  return top.reduce((s, m) => s + 25 * Math.max(m.prob1, m.prob2), 0)
}

export function bestDay(matches: MatchPred[], dates: string[]): string | null {
  if (!dates.length) return null
  const scores = Object.fromEntries(dates.map((d) => [d, dayScore(matches, d)]))
  return dates.reduce((a, b) => (scores[a] >= scores[b] ? a : b))
}

export function sortMatches(arr: MatchPred[], mode: SortMode): MatchPred[] {
  return [...arr].sort((a, b) => {
    if (mode === 'stars') {
      if (b.stars !== a.stars) return b.stars - a.stars
      return bestWinProb(b) - bestWinProb(a)
    }
    if (mode === 'prob') return bestWinProb(b) - bestWinProb(a)
    if (mode === 'time') {
      if (!a.time && !b.time) return 0
      if (!a.time) return 1
      if (!b.time) return -1
      return a.time.localeCompare(b.time)
    }
    return 0
  })
}

// ── BEST BETS (monolith calcBestBets) ────────────────────────

export type BestBetType = 'victoria' | 'over25' | 'btts'

export interface BestBet {
  match: MatchPred
  type: BestBetType
  team: string | null // short name of the team backed (victoria only)
  prob: number
  score: number
  fairOdds: number | null
}

export function calcBestBets(matches: MatchPred[]): BestBet[] {
  const bets: BestBet[] = []
  matches.forEach((m) => {
    // Result bet (team most likely to win, skip draws)
    const winner = winnerTeam(m)
    const betProb = winner === 'home' ? m.prob1 : m.prob2
    if (betProb >= 48) {
      const betTeam = winner === 'home' ? m.homeShort : m.awayShort
      bets.push({
        match: m,
        type: 'victoria',
        team: betTeam,
        prob: betProb,
        score: (m.stars * m.stars * betProb) / 100,
        fairOdds: m.fairOdds,
      })
    }
    if (m.over25yn && m.over25 >= 48) {
      bets.push({
        match: m,
        type: 'over25',
        team: null,
        prob: m.over25,
        score: (m.stars * m.stars * m.over25) / 100,
        fairOdds: m.over25 > 0 ? parseFloat((100 / m.over25).toFixed(2)) : null,
      })
    }
    if (m.bttsyn && m.btts >= 48) {
      bets.push({
        match: m,
        type: 'btts',
        team: null,
        prob: m.btts,
        score: (m.stars * m.stars * m.btts) / 100,
        fairOdds: m.btts > 0 ? parseFloat((100 / m.btts).toFixed(2)) : null,
      })
    }
  })
  return bets.sort((a, b) => b.score - a.score)
}

/** Outcome key for a BestBet leg (used for slip + edge lookup) */
export function bestBetOutcome(b: BestBet): Outcome {
  if (b.type === 'over25') return 'over25'
  if (b.type === 'btts') return 'btts'
  const w = winnerTeam(b.match)
  return w === 'home' ? 'home' : w === 'away' ? 'away' : 'draw'
}

// ── V2 PARLAY ENGINE (monolith getSuggestedParlays) ──────────

export interface EnrichedBet extends BestBet {
  edgeBonus: number
  legScore: number
  outcomeKey: Outcome
  matchKey: string
  league: string
  matchDate: string
}

export type ParlayRisk = 'safe' | 'medium' | 'risky' | 'extreme'

export interface Parlay {
  title: string
  riskLevel: ParlayRisk
  legs: EnrichedBet[]
  combinedProb: string
  combinedOdds: string | null
  correlationPenalty: number
}

function buildEdgeMap(matches: MatchPred[]): Map<string, number> {
  const em = new Map<string, number>()
  matches.forEach((m) => {
    if (!m.valueBets || !m.valueBets.length) return
    const mk = m.date + '|' + (m.home || m.homeShort)
    m.valueBets.forEach((vb) => {
      const k = mk + '|' + vb.outcome
      if (vb.edge > (em.get(k) || 0)) em.set(k, vb.edge)
    })
  })
  return em
}

function corrPenalty(legs: EnrichedBet[]): number {
  let penalty = 1.0
  for (let i = 0; i < legs.length; i++) {
    for (let j = i + 1; j < legs.length; j++) {
      if (legs[i].league === legs[j].league && legs[i].matchDate === legs[j].matchDate) {
        penalty *= 0.96
      }
    }
  }
  return penalty
}

export function getSuggestedParlays(bets: BestBet[], matches: MatchPred[]): Parlay[] {
  const edgeMap = buildEdgeMap(matches)

  const enriched: EnrichedBet[] = bets.map((b) => {
    const mk = b.match.date + '|' + (b.match.home || b.match.homeShort)
    const outcomeKey: Outcome =
      b.type === 'victoria' ? (winnerTeam(b.match) === 'home' ? 'home' : 'away') : b.type
    const edgePct = edgeMap.get(mk + '|' + outcomeKey) || 0
    const edgeBonus = edgePct / 100
    const legScore = b.match.stars * b.match.stars * b.prob * (1 + edgeBonus)
    const fairOdds = b.fairOdds || (b.prob > 0 ? parseFloat((100 / b.prob).toFixed(2)) : null)
    return {
      ...b,
      edgeBonus,
      legScore,
      fairOdds,
      outcomeKey,
      matchKey: mk,
      league: b.match.league,
      matchDate: b.match.date,
    }
  })

  function buildPool(minStars: number, minProb: number): EnrichedBet[] {
    const seen = new Set<string>()
    return enriched
      .filter((b) => b.match.stars >= minStars && b.prob >= minProb)
      .sort((a, b) => b.prob - a.prob)
      .filter((b) => {
        if (seen.has(b.matchKey)) return false
        seen.add(b.matchKey)
        return true
      })
  }

  function exhaustiveSearch(
    pool: EnrichedBet[],
    nLegs: number,
    minCombinedProb: number,
  ): EnrichedBet[] | null {
    let bestCombo: number[] | null = null
    let bestScore = -Infinity
    const n = pool.length

    function search(start: number, current: number[]): void {
      if (current.length === nLegs) {
        const legs = current.map((i) => pool[i])
        const rawProb = legs.reduce((p, b) => (p * b.prob) / 100, 1) * 100
        if (rawProb < minCombinedProb) return
        const score = rawProb * corrPenalty(legs)
        if (score > bestScore) {
          bestScore = score
          bestCombo = [...current]
        }
        return
      }
      const remaining = nLegs - current.length
      for (let i = start; i <= n - remaining; i++) {
        current.push(i)
        search(i + 1, current)
        current.pop()
      }
    }

    search(0, [])
    return bestCombo ? (bestCombo as number[]).map((i) => pool[i]) : null
  }

  function selectBestLegs(
    pool: EnrichedBet[],
    nLegs: number,
    minCombinedProb: number,
  ): EnrichedBet[] | null {
    if (pool.length < nLegs) return null
    if (pool.length <= 15) return exhaustiveSearch(pool, nLegs, minCombinedProb)
    return exhaustiveSearch(pool.slice(0, 15), nLegs, minCombinedProb)
  }

  function buildParlay(legs: EnrichedBet[], title: string, riskLevel: ParlayRisk): Parlay {
    const combinedProb = legs.reduce((p, b) => (p * b.prob) / 100, 1) * 100
    const hasOdds = legs.every((b) => b.fairOdds != null && !isNaN(b.fairOdds))
    const combinedOdds = hasOdds ? legs.reduce((o, b) => o * (b.fairOdds as number), 1) : null
    return {
      title,
      riskLevel,
      legs,
      combinedProb: combinedProb.toFixed(1),
      combinedOdds: combinedOdds ? combinedOdds.toFixed(2) : null,
      correlationPenalty: corrPenalty(legs),
    }
  }

  const configs: [number, string, ParlayRisk, number, number, number][] = [
    [2, 'Doble Segura', 'safe', 3, 62, 38],
    [3, 'Triple Sólida', 'medium', 3, 58, 22],
    [4, 'Cuádruple Firme', 'risky', 2, 55, 12],
    [5, 'Quíntuple Valiente', 'extreme', 2, 55, 8],
  ]

  const parlays: Parlay[] = []
  for (const [nLegs, title, riskLevel, minStars, minProb, minCombinedProb] of configs) {
    const pool = buildPool(minStars, minProb)
    const bestLegs = selectBestLegs(pool, nLegs, minCombinedProb)
    if (bestLegs) parlays.push(buildParlay(bestLegs, title, riskLevel))
  }
  return parlays
}

// ── WINIELA (monolith getWiniela) ────────────────────────────

export type WinielaPick = '1' | 'X' | '2' | 'O25' | 'BTTS'

export interface WinielaLeg {
  match: MatchPred
  pick: WinielaPick
  label: string
  prob: number
  odds: number | null
}

export interface Winiela {
  legs: WinielaLeg[]
  probDisplay: string
  combinedOdds: string | null
}

export function winielaOutcome(pick: WinielaPick): Outcome {
  return pick === '1' ? 'home' : pick === '2' ? 'away' : pick === 'X' ? 'draw' : pick === 'O25' ? 'over25' : 'btts'
}

export function getWiniela(matches: MatchPred[]): Winiela | null {
  const pdMatches = matches
    .filter((m) => m.league === 'PD')
    .sort((a, b) => (a.date + (a.time || '') < b.date + (b.time || '') ? -1 : 1))
  if (pdMatches.length < 2) return null

  const legs: WinielaLeg[] = pdMatches.map((m) => {
    const w = winnerTeam(m)
    const resultProb = w === 'home' ? m.prob1 : w === 'away' ? m.prob2 : m.probX
    const resultPick: WinielaPick = w === 'home' ? '1' : w === 'away' ? '2' : 'X'
    const resultLabel = w === 'home' ? m.homeShort : w === 'away' ? m.awayShort : 'X'

    const candidates: { pick: WinielaPick; label: string; prob: number }[] = [
      { pick: resultPick, label: resultLabel, prob: resultProb },
    ]
    if (m.over25yn && m.over25 > 0) candidates.push({ pick: 'O25', label: 'Over 2.5', prob: m.over25 })
    if (m.bttsyn && m.btts > 0) candidates.push({ pick: 'BTTS', label: 'BTTS', prob: m.btts })

    const best = candidates.reduce((a, b) => (b.prob > a.prob ? b : a))
    const odds = best.prob > 0 ? parseFloat((100 / best.prob).toFixed(2)) : null
    return { match: m, pick: best.pick, label: best.label, prob: best.prob, odds }
  })

  const combinedProb = legs.reduce((p, l) => (p * l.prob) / 100, 1) * 100
  const hasOdds = legs.every((l) => l.odds !== null)
  const rawOdds = hasOdds ? legs.reduce((o, l) => o * (l.odds as number), 1) : null

  let combinedOdds: string | null = null
  if (rawOdds !== null) {
    combinedOdds =
      rawOdds >= 10000
        ? Math.round(rawOdds).toLocaleString()
        : rawOdds >= 100
          ? rawOdds.toFixed(1)
          : rawOdds.toFixed(2)
  }
  const probDisplay =
    combinedProb >= 0.1
      ? combinedProb.toFixed(2) + '%'
      : combinedProb >= 0.001
        ? combinedProb.toFixed(4) + '%'
        : combinedProb.toExponential(2) + '%'

  return { legs, probDisplay, combinedOdds }
}

// ── AUTOPILOT selection engine (monolith selectAutopilotBets) ─

export interface AutopilotPick {
  m: MatchPred
  vb: ValueBet
  score: number
  league: string
  date: string
  stake: number
  ev: number
  correlated: boolean
}

export interface AutopilotPlan {
  picks: AutopilotPick[]
  relaxedStars: number | null
}

export function selectAutopilotBets(
  matches: MatchPred[],
  profile: AutopilotProfile | null,
  bankroll: Bankroll | null,
  metrics: TrackerMetrics | null,
): AutopilotPlan {
  if (!profile || !bankroll) return { picks: [], relaxedStars: null }

  const kellyMult = { conservative: 0.5, balanced: 1.0, aggressive: 1.5 }[profile.riskLevel] || 1.0
  const leagueRoi = metrics?.per_league || {}
  const wantedMin = profile.minStars || 5

  type Candidate = Omit<AutopilotPick, 'stake' | 'ev' | 'correlated'>

  const collect = (minSt: number): Candidate[] => {
    const out: Candidate[] = []
    matches.forEach((m) => {
      if (!m.valueBets?.length || m.stars < minSt) return
      if (!profile.leagues.includes(m.league)) return
      if (profile.activeDates?.length && !profile.activeDates.includes(m.date)) return
      m.valueBets.forEach((vb) => {
        if (vb.bkOdds < profile.minOdds) return
        const market = vb.outcome === 'over25' ? 'over25' : vb.outcome === 'btts' ? 'btts' : '1x2'
        if (!profile.markets.includes(market)) return
        const lgMult = Math.max(0.5, Math.min(2.0, 1.0 + (leagueRoi[m.league]?.roi ?? 0)))
        out.push({
          m,
          vb,
          score: (vb.edge / 100) * (vb.kelly / 100) * lgMult,
          league: m.league,
          date: m.date,
        })
      })
    })
    return out
  }

  // Progressive fallback: 5★ → 4★ → 3★
  let candidates = collect(wantedMin)
  let effectiveMin = wantedMin
  for (const fallback of [4, 3]) {
    if (candidates.length >= profile.maxBetsPerDay || effectiveMin <= fallback) break
    candidates = collect(fallback)
    effectiveMin = fallback
  }
  const relaxedStars = effectiveMin < wantedMin ? effectiveMin : null

  candidates.sort((a, b) => b.score - a.score)
  const seen = new Set<string>()
  const picked: Candidate[] = []
  for (const c of candidates) {
    const key = `${c.date}|${c.m.home}|${c.m.away}`
    if (!seen.has(key)) {
      seen.add(key)
      picked.push(c)
    }
    if (picked.length >= profile.maxBetsPerDay) break
  }

  const leagueCount: Record<string, number> = {}
  picked.forEach((c) => {
    const k = `${c.league}|${c.date}`
    leagueCount[k] = (leagueCount[k] || 0) + 1
  })
  const maxPerBet = (bankroll.current * (profile.maxExposurePct / 100)) / Math.max(picked.length, 1)

  const picks: AutopilotPick[] = picked.map((c) => {
    const k = `${c.league}|${c.date}`
    const corrPen = leagueCount[k] >= 3 ? 0.65 : leagueCount[k] === 2 ? 0.8 : 1.0
    const stake = +Math.min(
      (c.vb.kelly / 100) * bankroll.current * kellyMult * corrPen,
      maxPerBet,
    ).toFixed(2)
    return {
      ...c,
      stake,
      ev: +((stake * c.vb.edge) / 100).toFixed(2),
      correlated: leagueCount[k] > 1,
    }
  })

  return { picks, relaxedStars }
}

// ── Kelly helper (slip warning) ──────────────────────────────

/** stake / kellyStake ratio, or null when no kelly reference */
export function kellyRatio(stake: number, kellyStake: number | null): number | null {
  if (!kellyStake) return null
  return Number(stake) / kellyStake
}

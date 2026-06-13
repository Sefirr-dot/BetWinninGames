// ─────────────────────────────────────────────────────────────
// Settlement — EXACT port of the monolith resolveOutcome() +
// settleBets() semantics. This touches the user's real betting
// history; do not "improve" the math.
// ─────────────────────────────────────────────────────────────
import type { HistoryBet, Outcome, ResolvedResult } from '../types'
import { loadBankroll, loadHistory, saveBankroll, saveHistory } from './storage'

export function resolveOutcome(outcome: Outcome, res: ResolvedResult): boolean {
  if (outcome === 'home') return res.result === 'H'
  if (outcome === 'draw') return res.result === 'D'
  if (outcome === 'away') return res.result === 'A'
  if (outcome === 'over25') return !!res.over25
  if (outcome === 'btts') return !!res.btts
  return false
}

export interface SettleSummary {
  count: number
  won: number
  lost: number
  netPnl: number
}

/**
 * Cross unsettled bwg_history bets against RESOLVED_RESULTS and
 * update bankroll.current = initial + Σ pnl(settled).
 * Returns a summary of bets settled in this call (or null).
 */
export function settleBets(results: Record<string, ResolvedResult>): SettleSummary | null {
  if (!results || !Object.keys(results).length) return null
  const history = loadHistory().map((b) => ({ ...b })) as HistoryBet[]
  const bankroll = loadBankroll()
  if (!bankroll || !history.length) return null

  let changed = false

  history.forEach((bet) => {
    if (bet.settled) return
    if (bet.type === 'simple') {
      const key = `${bet.homeName}|${bet.awayName}|${bet.matchDate}`
      const res = results[key]
      if (!res) return
      const won = resolveOutcome(bet.outcome, res)
      bet.result = won ? 'won' : 'lost'
      bet.pnl = won ? +((Number(bet.oddsUser) - 1) * Number(bet.stake)).toFixed(2) : -Number(bet.stake)
      bet.settled = true
      bet.settledAt = new Date().toISOString()
      changed = true
    } else if (bet.type === 'combinada') {
      const allResolved = bet.legs.every(
        (leg) => !!results[`${leg.homeName}|${leg.awayName}|${leg.matchDate}`],
      )
      if (!allResolved) return
      const allWon = bet.legs.every((leg) =>
        resolveOutcome(leg.outcome, results[`${leg.homeName}|${leg.awayName}|${leg.matchDate}`]),
      )
      bet.result = allWon ? 'won' : 'lost'
      bet.pnl = allWon
        ? +((Number(bet.combinedOdds) - 1) * Number(bet.stake)).toFixed(2)
        : -Number(bet.stake)
      bet.settled = true
      bet.settledAt = new Date().toISOString()
      changed = true
    }
  })

  if (!changed) return null

  saveHistory(history)
  const totalPnl = history.filter((b) => b.settled).reduce((s, b) => s + b.pnl, 0)
  saveBankroll({ ...bankroll, current: +(bankroll.initial + totalPnl).toFixed(2) })

  const justSettled = history.filter(
    (b) => b.settled && b.settledAt && Date.now() - new Date(b.settledAt).getTime() < 5000,
  )
  if (!justSettled.length) return null
  return {
    count: justSettled.length,
    won: justSettled.filter((b) => b.result === 'won').length,
    lost: justSettled.filter((b) => b.result === 'lost').length,
    netPnl: justSettled.reduce((s, b) => s + b.pnl, 0),
  }
}

// ─────────────────────────────────────────────────────────────
// Slip / bankroll mutations — exact semantics of the monolith
// addToSlip / placeBets / placeCombinada / loadParlayToSlip /
// autopilotAddAll / confirmBankroll / resetBankroll.
// All records stay byte-compatible with the legacy page.
// ─────────────────────────────────────────────────────────────
import { t } from '../i18n'
import type {
  Bankroll,
  HistoryBet,
  HistoryParlayBet,
  HistorySimpleBet,
  Outcome,
  ParlayLeg,
  ResolvedResult,
} from '../types'
import type { AutopilotPick } from './betting'
import { settleBets } from './settlement'
import {
  loadBankroll,
  loadHistory,
  loadSlip,
  removeBankroll,
  removeHistory,
  saveBankroll,
  saveHistory,
  saveSlip,
} from './storage'
import { toast } from './toast'

export type ActionResult = 'OK' | 'NO_BANKROLL' | 'DUP' | 'EMPTY' | 'OVER_BANKROLL'

export interface AddToSlipArgs {
  homeName: string
  awayName: string
  league: string
  matchDate: string
  outcome: Outcome
  suggestedOdds: number | null
  kellyStake: number | null
}

export function addToSlip(args: AddToSlipArgs): ActionResult {
  const brl = loadBankroll()
  if (!brl) return 'NO_BANKROLL'
  const slip = loadSlip()
  if (
    slip.some(
      (s) => s.homeName === args.homeName && s.awayName === args.awayName && s.outcome === args.outcome,
    )
  ) {
    toast(t('toast_in_slip'))
    return 'DUP'
  }
  const defaultStake = args.kellyStake || +(brl.current * 0.02).toFixed(2)
  saveSlip([
    ...slip,
    {
      id: Date.now() + '',
      homeName: args.homeName,
      awayName: args.awayName,
      league: args.league,
      matchDate: args.matchDate,
      outcome: args.outcome,
      oddsUser: args.suggestedOdds || 2.0,
      stake: defaultStake,
      kellyStake: args.kellyStake || null,
      addedAt: new Date().toISOString(),
    },
  ])
  toast(t('toast_added'))
  return 'OK'
}

export function removeFromSlip(id: string): void {
  saveSlip(loadSlip().filter((s) => s.id !== id))
}

export function updateSlipItem(id: string, patch: { oddsUser?: number; stake?: number }): void {
  const slip = loadSlip()
  const i = slip.findIndex((x) => x.id === id)
  if (i < 0) return
  const next = [...slip]
  next[i] = { ...next[i], ...patch }
  saveSlip(next)
}

export function placeBets(results: Record<string, ResolvedResult>): ActionResult {
  const slip = loadSlip()
  if (!slip.length) return 'EMPTY'
  const brl = loadBankroll()
  if (!brl) return 'NO_BANKROLL'
  const hist = loadHistory()
  const total = slip.reduce((s, b) => s + (parseFloat(String(b.stake)) || 0), 0)
  if (total > brl.current) return 'OVER_BANKROLL'
  const placed: HistorySimpleBet[] = slip.map((b) => ({
    ...b,
    type: 'simple' as const,
    placedAt: new Date().toISOString(),
    settled: false,
    result: null,
    pnl: 0,
  }))
  saveHistory([...hist, ...placed] as HistoryBet[])
  saveBankroll({ ...brl, current: +(brl.current - total).toFixed(2) })
  saveSlip([])
  notifySettlement(settleBets(results))
  return 'OK'
}

export function placeParlay(stake: number, results: Record<string, ResolvedResult>): ActionResult {
  const slip = loadSlip()
  if (slip.length < 2) return 'EMPTY'
  const brl = loadBankroll()
  if (!brl) return 'NO_BANKROLL'
  if (stake > brl.current) return 'OVER_BANKROLL'
  const combinedOdds = +slip.reduce((p, b) => p * (parseFloat(String(b.oddsUser)) || 1), 1).toFixed(3)
  const bet: HistoryParlayBet = {
    id: Date.now() + '',
    type: 'combinada',
    legs: slip.map((b) => ({
      homeName: b.homeName,
      awayName: b.awayName,
      league: b.league,
      matchDate: b.matchDate,
      outcome: b.outcome,
      oddsUser: b.oddsUser,
    })),
    combinedOdds,
    stake,
    placedAt: new Date().toISOString(),
    settled: false,
    result: null,
    pnl: 0,
  }
  saveHistory([...loadHistory(), bet])
  saveBankroll({ ...brl, current: +(brl.current - stake).toFixed(2) })
  saveSlip([])
  notifySettlement(settleBets(results))
  return 'OK'
}

/** Replaces the slip with a pre-built parlay (Top Picks / Winiela). */
export function loadParlayToSlip(legs: (ParlayLeg & { oddsUser: number })[]): ActionResult {
  const brl = loadBankroll()
  if (!brl) return 'NO_BANKROLL'
  if (!legs.length) return 'EMPTY'
  saveSlip(
    legs.map((leg) => ({
      id: Date.now() + Math.random() + '',
      homeName: leg.homeName,
      awayName: leg.awayName,
      league: leg.league,
      matchDate: leg.matchDate,
      outcome: leg.outcome,
      oddsUser: leg.oddsUser || 2.0,
      stake: +((loadBankroll()?.current || 100) * 0.02).toFixed(2),
      kellyStake: null,
      addedAt: new Date().toISOString(),
    })),
  )
  return 'OK'
}

export function autopilotAddAll(picks: AutopilotPick[]): ActionResult {
  if (!picks.length) return 'EMPTY'
  const brl = loadBankroll()
  if (!brl) return 'NO_BANKROLL'
  const slip = [...loadSlip()]
  picks.forEach((p) => {
    const key = `${p.m.home || p.m.homeShort}|${p.m.away || p.m.awayShort}|${p.m.date}|${p.vb.outcome}`
    if (slip.some((s) => `${s.homeName}|${s.awayName}|${s.matchDate}|${s.outcome}` === key)) return
    slip.push({
      id: Date.now() + Math.random() + '',
      homeName: p.m.home || p.m.homeShort,
      awayName: p.m.away || p.m.awayShort,
      league: p.m.league,
      matchDate: p.m.date,
      outcome: p.vb.outcome,
      oddsUser: p.vb.bkOdds,
      stake: p.stake,
      kellyStake: +((brl.current * p.vb.kelly) / 100).toFixed(2),
      addedAt: new Date().toISOString(),
    })
  })
  saveSlip(slip)
  return 'OK'
}

export function confirmBankroll(val: number): boolean {
  if (!val || val <= 0) return false
  saveBankroll({ initial: val, current: val, setAt: new Date().toISOString() })
  return true
}

export function resetBankroll(): void {
  removeBankroll()
  removeHistory()
  saveSlip([])
}

export function runSettlement(results: Record<string, ResolvedResult>): void {
  notifySettlement(settleBets(results))
}

function notifySettlement(summary: ReturnType<typeof settleBets>): void {
  if (!summary) return
  const sign = summary.netPnl >= 0 ? '+' : ''
  const parts = [
    `${summary.count} bet${summary.count > 1 ? 's' : ''} ${t('toast_settled')}${summary.count > 1 ? 's' : ''}`,
  ]
  if (summary.won) parts.push(`✅ ${summary.won} ${t('toast_won')}${summary.won > 1 ? 's' : ''}`)
  if (summary.lost) parts.push(`❌ ${summary.lost} ${t('toast_lost')}${summary.lost > 1 ? 's' : ''}`)
  parts.push(`${sign}€${Math.abs(summary.netPnl).toFixed(2)}`)
  toast(parts.join(' · '))
}

// ── Export / Import (new safety feature) ─────────────────────

export function exportBankrollJSON(): void {
  const payload = {
    app: 'BetWinninGames',
    exportedAt: new Date().toISOString(),
    bwg_bankroll: loadBankroll(),
    bwg_history: loadHistory(),
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `bwg_export_${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)
}

export function importBankrollJSON(text: string): boolean {
  try {
    const parsed: unknown = JSON.parse(text)
    if (typeof parsed !== 'object' || parsed === null) return false
    const obj = parsed as Record<string, unknown>
    const hist = obj.bwg_history
    const brl = obj.bwg_bankroll
    if (!Array.isArray(hist)) return false
    if (typeof brl !== 'object' || brl === null) return false
    const b = brl as Record<string, unknown>
    if (typeof b.initial !== 'number' || typeof b.current !== 'number') return false
    saveHistory(hist as HistoryBet[])
    saveBankroll(brl as unknown as Bankroll)
    return true
  } catch {
    return false
  }
}

// ─────────────────────────────────────────────────────────────
// SLIP view — bankroll tracker with Active / History / Parlay
// tabs (monolith renderSlipView + placeBets/placeCombinada +
// analyzeRiskWithOllama). Adds Export/Import JSON.
// ─────────────────────────────────────────────────────────────
import { useRef, useState } from 'react'
import { t, tf, LANG } from '../i18n'
import {
  exportBankrollJSON,
  importBankrollJSON,
  placeBets,
  placeParlay,
  removeFromSlip,
  resetBankroll,
  updateSlipItem,
} from '../lib/actions'
import { kellyRatio } from '../lib/betting'
import { outcomeColor, outcomeLabel, shortStamp } from '../lib/format'
import { resolveOutcome } from '../lib/settlement'
import { ollamaChat } from '../lib/ollama'
import { useBankroll, useHistory, useSlip } from '../lib/storage'
import { toast } from '../lib/toast'
import { useApp } from '../state/AppContext'
import { LeagueBadge } from '../components/ui'
import type { HistoryBet, ResolvedResult, SlipItem, SlipTab } from '../types'

// ── numeric field that commits on blur / Enter ───────────────

function NumField({
  value,
  min,
  step,
  fallback,
  width = 'w-18',
  onCommit,
}: {
  value: number
  min: number
  step: number
  fallback: number
  width?: string
  onCommit: (v: number) => void
}) {
  const [text, setText] = useState(String(value))
  const commit = () => onCommit(parseFloat(text) || fallback)
  return (
    <input
      type="number"
      min={min}
      step={step}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          commit()
          ;(e.target as HTMLInputElement).blur()
        }
      }}
      className={`bg-raised border-line stat-num rounded-md border px-2 py-1.5 text-center text-[15px] outline-none ${width}`}
    />
  )
}

// ── Ollama risk box ──────────────────────────────────────────

function useOllamaRisk() {
  const [busy, setBusy] = useState(false)
  const [text, setText] = useState<string | null>(null)
  const [err, setErr] = useState(false)

  const analyze = async (slip: SlipItem[], brl: { current: number; initial: number } | null, isCombinada: boolean, combStake: number | null) => {
    if (!slip.length) {
      setErr(false)
      setText(t('ollama_add_first'))
      return
    }
    setBusy(true)
    setErr(false)
    setText(t('ollama_querying'))

    const combOdds = isCombinada ? +slip.reduce((p, b) => p * (parseFloat(String(b.oddsUser)) || 1), 1).toFixed(3) : null
    const total = isCombinada ? combStake || 10 : slip.reduce((s, b) => s + (parseFloat(String(b.stake)) || 0), 0)
    const exposure = brl ? ((total / brl.current) * 100).toFixed(1) : '?'

    const picksLines = slip
      .map((b, i) => {
        const kr = !isCombinada && b.kellyStake ? kellyRatio(parseFloat(String(b.stake)), b.kellyStake) : null
        const kw = kr && kr > 1.5 ? ` [SUPERA KELLY x${kr.toFixed(1)}]` : ''
        return `${i + 1}. ${b.homeName} vs ${b.awayName} [${b.league}] | ${outcomeLabel(b.outcome)} @${b.oddsUser}${kw}`
      })
      .join('\n')

    const tipoApuesta = isCombinada
      ? `COMBINADA ${slip.length} piernas — Cuota combinada @${combOdds} — Stake €${combStake || 10}`
      : `SIMPLES (${slip.length} picks independientes)`

    const prompt = `Eres un gestor de riesgo de apuestas. Analiza en ${LANG === 'es' ? 'español' : 'inglés'}.

BANKROLL: €${brl?.current?.toFixed(2) || '?'} (inicial €${brl?.initial?.toFixed(2) || '?'})
TIPO: ${tipoApuesta}
EXPOSICIÓN: €${total.toFixed(2)} (${exposure}% del bankroll)

PICKS:
${picksLines}

Responde SOLO con:
RIESGO: (Bajo/Medio/Alto/Muy Alto)
PROBLEMAS: (lista máx 3, o "Ninguno")
CONSEJO: (1-2 frases concretas)`

    try {
      const txt = await ollamaChat(prompt)
      setText(txt || '—')
    } catch {
      setErr(true)
      setText(t('ollama_unavail'))
    } finally {
      setBusy(false)
    }
  }

  return { busy, text, err, analyze, clear: () => setText(null) }
}

function OllamaBox({ text, err }: { text: string | null; err: boolean }) {
  if (!text) return null
  return (
    <div
      className="bg-raised/60 border-line mt-3 rounded-lg border px-4 py-3 text-[13px] leading-relaxed whitespace-pre-line"
      style={err ? { color: 'var(--color-orange)' } : undefined}
    >
      {text}
    </div>
  )
}

// ── expandable history row ───────────────────────────────────

function HistRow({
  bet,
  results,
}: {
  bet: HistoryBet
  results: Record<string, ResolvedResult>
}) {
  const [open, setOpen] = useState(false)

  const badge = bet.settled ? (
    bet.result === 'won' ? (
      <span className="bg-green/15 text-green rounded-md px-2 py-0.5 text-[11px] font-bold">
        {t('badge_won')} +€{bet.pnl.toFixed(2)}
      </span>
    ) : (
      <span className="bg-red/15 text-red rounded-md px-2 py-0.5 text-[11px] font-bold">
        {t('badge_lost')} -€{Math.abs(bet.pnl).toFixed(2)}
      </span>
    )
  ) : (
    <span className="bg-amber/15 text-amber rounded-md px-2 py-0.5 text-[11px] font-bold">{t('badge_pending')}</span>
  )

  const title =
    bet.type === 'combinada'
      ? `🔗 ${tf('hist_parlay', bet.legs?.length ?? 0)} @${bet.combinedOdds}`
      : `${bet.homeName} vs ${bet.awayName}`

  let detail: React.ReactNode
  if (bet.type === 'combinada') {
    detail = (
      <>
        <div className="text-mute mb-1 text-[11px]">{t('sec_legs')}</div>
        {(bet.legs || []).map((leg, i) => {
          const legRes = results[`${leg.homeName}|${leg.awayName}|${leg.matchDate}`]
          const legWon = legRes ? resolveOutcome(leg.outcome, legRes) : null
          return (
            <div key={i} className="border-line flex items-center gap-2 border-b py-1.5 text-[12px]">
              <span className="text-mute min-w-4">{i + 1}</span>
              <span>
                {leg.homeName} vs {leg.awayName}
              </span>
              <span className="text-[11px] font-semibold" style={{ color: outcomeColor(leg.outcome) }}>
                {outcomeLabel(leg.outcome)}
              </span>
              <span className="text-mute text-[11px]">@{leg.oddsUser}</span>
              <span className="ml-auto">{legWon === true ? '✅' : legWon === false ? '❌' : '⏳'}</span>
            </div>
          )
        })}
        <div className="mt-2 flex flex-wrap gap-4 text-[12px]">
          <span className="text-mute">
            {t('lbl_stake_word')} <b className="text-ink">€{bet.stake}</b>
          </span>
          <span className="text-mute">
            {t('lbl_comb_odds')} <b className="text-gold">@{bet.combinedOdds}</b>
          </span>
          <span className="text-mute">
            {t('lbl_pot_ret')} <b className="text-green">€{(bet.stake * bet.combinedOdds).toFixed(2)}</b>
          </span>
        </div>
      </>
    )
  } else {
    const pResult = bet.settled
      ? bet.result === 'won'
        ? `+€${bet.pnl.toFixed(2)}`
        : `-€${Math.abs(bet.pnl).toFixed(2)}`
      : t('badge_pending')
    const pColor = !bet.settled ? 'var(--color-mute)' : bet.result === 'won' ? 'var(--color-green)' : 'var(--color-orange)'
    detail = (
      <>
        <div className="flex flex-wrap gap-4 text-[12px]">
          <span className="text-mute">
            {t('lbl_league')} <b className="text-ink">{bet.league || '—'}</b>
          </span>
          <span className="text-mute">
            {t('lbl_bet')} <b style={{ color: outcomeColor(bet.outcome) }}>{outcomeLabel(bet.outcome)}</b>
          </span>
          <span className="text-mute">
            {t('lbl_odds')} <b className="text-gold">@{bet.oddsUser}</b>
          </span>
          <span className="text-mute">
            {t('lbl_stake_word')} <b className="text-ink">€{bet.stake}</b>
          </span>
          <span className="text-mute">
            {t('lbl_pot_ret')} <b className="text-green">€{(bet.stake * bet.oddsUser).toFixed(2)}</b>
          </span>
          <span className="text-mute">
            {t('lbl_result')} <b style={{ color: pColor }}>{pResult}</b>
          </span>
          {!bet.settled && (
            <span className="text-mute">
              {t('lbl_match_date')} <b className="text-ink">{bet.matchDate || '—'}</b>
            </span>
          )}
        </div>
        <div className="text-mute mt-1.5 text-[11px]">
          {t('lbl_registered')} {shortStamp(bet.placedAt)}
          {bet.settledAt ? ` · ${t('lbl_settled_at')} ${shortStamp(bet.settledAt)}` : ''}
        </div>
      </>
    )
  }

  return (
    <div className="mb-1">
      <div
        className="bg-raised/40 border-line flex cursor-pointer items-center gap-3 rounded-lg border px-3.5 py-2.5 select-none"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-semibold">{title}</div>
          <div className="text-mute mt-0.5 text-[11px]">
            {(bet.settledAt || bet.placedAt || '').slice(0, 10)} · {t('lbl_stake_word')} €{bet.stake}
            {bet.type !== 'combinada' && bet.league ? ` · ${bet.league}` : ''}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {badge}
          <span className="text-mute text-sm">{open ? '▴' : '▾'}</span>
        </div>
      </div>
      {open && <div className="bg-raised/30 border-line rounded-b-lg border border-t-0 px-3.5 py-2.5">{detail}</div>}
    </div>
  )
}

// ── main view ────────────────────────────────────────────────

export default function SlipView() {
  const { data, slipTab, setSlipTab, setActive, setBankrollModal } = useApp()
  const brl = useBankroll()
  const slip = useSlip()
  const hist = useHistory()
  const ollama = useOllamaRisk()
  const fileRef = useRef<HTMLInputElement>(null)
  const [parlayStake, setParlayStake] = useState(10)

  const settled = hist.filter((b) => b.settled)
  const pending = hist.filter((b) => !b.settled)
  const totalPnl = settled.reduce((s, b) => s + b.pnl, 0)
  const roi = brl ? (totalPnl / brl.initial) * 100 : 0
  const inPlay =
    pending.reduce((s, b) => s + (Number(b.stake) || 0), 0) +
    slip.reduce((s, b) => s + (parseFloat(String(b.stake)) || 0), 0)
  const pnlColor = totalPnl >= 0 ? 'var(--color-green)' : 'var(--color-orange)'

  const doReset = () => {
    if (!window.confirm(t('confirm_reset'))) return
    resetBankroll()
    setBankrollModal(true)
  }

  const doPlaceBets = () => {
    const res = placeBets(data.results)
    if (res === 'OVER_BANKROLL') toast(t('alert_over_bankroll'))
    else if (res === 'NO_BANKROLL') setBankrollModal(true)
    else if (res === 'OK') ollama.clear()
  }

  const doPlaceParlay = () => {
    if (slip.length < 2) {
      toast(t('alert_min_parlay'))
      return
    }
    const res = placeParlay(parlayStake, data.results)
    if (res === 'OVER_BANKROLL') toast(t('alert_over_bankroll_one'))
    else if (res === 'NO_BANKROLL') setBankrollModal(true)
    else if (res === 'OK') ollama.clear()
  }

  const onImportFile = (f: File | undefined) => {
    if (!f) return
    f.text().then((txt) => {
      toast(importBankrollJSON(txt) ? t('import_ok') : t('import_err'))
    })
  }

  const tabBtn = (tab: SlipTab, label: string, badge: React.ReactNode) => (
    <button
      onClick={() => setSlipTab(tab)}
      className={`flex cursor-pointer items-center gap-1.5 border-b-2 px-4 py-2.5 text-[13px] font-semibold transition-colors ${
        slipTab === tab ? 'border-teal text-ink' : 'text-mute hover:text-ink border-transparent'
      }`}
    >
      {label}
      {badge}
    </button>
  )

  // ── header ──
  const header = (
    <div className="mb-2 flex flex-wrap items-center gap-2.5">
      <div className="bg-raised/60 border-line flex items-baseline gap-2 rounded-lg border px-3.5 py-2">
        <span className="micro-label">{t('lbl_bankroll')}</span>
        <span className="stat-num text-2xl">€{brl ? brl.current.toFixed(2) : '—'}</span>
      </div>
      <div className="bg-raised/60 border-line flex items-baseline gap-2 rounded-lg border px-3.5 py-2">
        <span className="micro-label">{t('lbl_pnl')}</span>
        <span className="stat-num text-lg" style={{ color: pnlColor }}>
          {totalPnl >= 0 ? '+' : ''}€{totalPnl.toFixed(2)}
        </span>
        <span className="text-xs" style={{ color: pnlColor }}>
          {(roi >= 0 ? '+' : '') + roi.toFixed(1)}%
        </span>
      </div>
      <div className="bg-raised/60 border-line flex items-baseline gap-2 rounded-lg border px-3.5 py-2">
        <span className="micro-label">{t('lbl_in_play')}</span>
        <span className="stat-num text-amber text-lg">€{inPlay.toFixed(2)}</span>
      </div>
      <div className="ml-auto flex gap-2">
        <button
          onClick={doReset}
          className="border-line text-mute hover:text-ink cursor-pointer rounded-md border px-3 py-1.5 text-xs"
        >
          {t('btn_reset')}
        </button>
        {!brl && (
          <button
            onClick={() => setBankrollModal(true)}
            className="bg-green cursor-pointer rounded-md px-3.5 py-1.5 text-xs font-bold text-black"
          >
            {t('btn_config_bkrl')}
          </button>
        )}
      </div>
    </div>
  )

  // ── ACTIVE tab body ──
  let body: React.ReactNode
  if (slipTab === 'active') {
    if (!slip.length) {
      body = (
        <div className="px-6 py-14 text-center">
          <div className="text-4xl">🎯</div>
          <div className="text-mute mt-3 text-sm whitespace-pre-line">{t('empty_slip')}</div>
          <button
            onClick={() => setActive(data.dates[0] || 'ALL')}
            className="bg-raised border-line hover:border-teal mt-5 cursor-pointer rounded-lg border px-5 py-2.5 text-sm"
          >
            {t('btn_view_pred')}
          </button>
        </div>
      )
    } else {
      const total = slip.reduce((s, b) => s + (parseFloat(String(b.stake)) || 0), 0)
      const retPot = slip.reduce(
        (s, b) => s + (parseFloat(String(b.stake)) || 0) * (parseFloat(String(b.oddsUser)) || 1),
        0,
      )
      const exposurePct = brl ? (total / brl.current) * 100 : 0
      const expColor =
        exposurePct > 20 ? 'var(--color-orange)' : exposurePct > 10 ? 'var(--color-amber)' : 'var(--color-green)'

      body = (
        <div>
          <div className="flex flex-col">
            {slip.map((b) => {
              const kr = kellyRatio(parseFloat(String(b.stake)), b.kellyStake)
              return (
                <div key={b.id} className="border-line flex flex-wrap items-center gap-3 border-b py-3">
                  <div className="min-w-45 flex-1">
                    <div className="text-[13px] font-semibold">
                      {b.homeName} vs {b.awayName}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px]">
                      <LeagueBadge league={b.league} />
                      <span className="font-semibold" style={{ color: outcomeColor(b.outcome) }}>
                        {outcomeLabel(b.outcome)}
                      </span>
                      <span className="text-mute">{b.matchDate}</span>
                    </div>
                    {kr !== null &&
                      (kr > 1.5 ? (
                        <div className="text-orange mt-1 text-[11px]">
                          ⚠️ {t('kelly_above')} {kr.toFixed(1)}
                          {t('kelly_suffix')} (€{b.kellyStake?.toFixed(2)})
                        </div>
                      ) : (
                        <div className="text-green mt-1 text-[11px]">{t('kelly_ok')}</div>
                      ))}
                  </div>
                  <div className="flex items-center gap-2.5">
                    <div>
                      <div className="micro-label mb-0.5 text-center">{t('lbl_odds')}</div>
                      <NumField
                        key={`o-${b.id}-${b.oddsUser}`}
                        value={b.oddsUser}
                        min={1.01}
                        step={0.01}
                        fallback={2}
                        onCommit={(v) => updateSlipItem(b.id, { oddsUser: v })}
                      />
                    </div>
                    <div>
                      <div className="micro-label mb-0.5 text-center">{t('lbl_stake')}</div>
                      <NumField
                        key={`s-${b.id}-${b.stake}`}
                        value={b.stake}
                        min={0.5}
                        step={0.5}
                        fallback={1}
                        onCommit={(v) => updateSlipItem(b.id, { stake: v })}
                      />
                    </div>
                    <div className="min-w-16 text-right">
                      <div className="micro-label">{t('lbl_return')}</div>
                      <div className="stat-num text-green text-base">
                        €{((parseFloat(String(b.stake)) || 0) * (parseFloat(String(b.oddsUser)) || 1)).toFixed(2)}
                      </div>
                    </div>
                    <button
                      onClick={() => removeFromSlip(b.id)}
                      className="border-line text-mute hover:text-red h-7 w-7 cursor-pointer rounded-md border text-xs"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              )
            })}
          </div>

          <div className="bg-raised/60 mt-4 flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl px-4 py-3.5">
            <div>
              <span className="micro-label">{t('lbl_exposure')}</span>
              <div className="stat-num text-lg" style={{ color: expColor }}>
                €{total.toFixed(2)} ({exposurePct.toFixed(1)}%)
              </div>
            </div>
            <div>
              <span className="micro-label">{t('lbl_pot_ret')}</span>
              <div className="stat-num text-green text-lg">€{retPot.toFixed(2)}</div>
            </div>
            <div className="ml-auto flex flex-wrap gap-2">
              <button
                disabled={ollama.busy}
                onClick={() => ollama.analyze(slip, brl, false, null)}
                className="bg-surface border-line cursor-pointer rounded-lg border px-4 py-2.5 text-[13px] font-semibold disabled:opacity-60"
              >
                {ollama.busy ? t('btn_analyzing') : t('btn_ollama')}
              </button>
              <button
                disabled={!brl}
                onClick={doPlaceBets}
                className="bg-green font-display cursor-pointer rounded-lg px-5 py-2.5 text-base font-bold text-black disabled:opacity-50"
              >
                {t('btn_place')}
              </button>
            </div>
          </div>
          <OllamaBox text={ollama.text} err={ollama.err} />
        </div>
      )
    }
  }

  // ── HISTORY tab body ──
  else if (slipTab === 'history') {
    if (!hist.length) {
      body = (
        <div className="px-6 py-14 text-center">
          <div className="text-4xl">📋</div>
          <div className="text-mute mt-3 text-sm">{t('empty_history')}</div>
        </div>
      )
    } else {
      // personal stats
      const sWon = settled.filter((b) => b.result === 'won')
      const winRate = settled.length ? ((sWon.length / settled.length) * 100).toFixed(0) : '—'
      const stakeSum = settled.reduce((s, b) => s + (Number(b.stake) || 0), 0)
      const roiStr = settled.length && stakeSum > 0 ? ((totalPnl / stakeSum) * 100).toFixed(1) : '—'
      const avgOdds = settled.length
        ? (
            settled.reduce(
              (s, b) => s + (Number(b.type === 'combinada' ? b.combinedOdds : b.oddsUser) || 0),
              0,
            ) / settled.length
          ).toFixed(2)
        : '—'
      let streak = 0
      let streakType: string | null = null
      for (const b of [...settled].sort((a, b2) => ((b2.settledAt || '') > (a.settledAt || '') ? 1 : -1))) {
        if (!streakType) streakType = b.result
        if (b.result === streakType) streak++
        else break
      }
      const streakStr =
        streak > 1 ? `${streak} ${streakType === 'won' ? '✅' : '❌'} ${t('stat_streak_suffix')}` : '—'
      const roiColor = roiStr !== '—' && parseFloat(roiStr) >= 0 ? 'var(--color-green)' : 'var(--color-orange)'

      // sparkline
      let runningBrl = brl ? brl.initial : 0
      const curve: number[] = []
      ;[...hist]
        .sort((a, b) => ((a.placedAt || '') < (b.placedAt || '') ? -1 : 1))
        .forEach((b) => {
          if (b.settled) {
            runningBrl += b.pnl
            curve.push(runningBrl)
          }
        })
      const minBrl = Math.min(...curve, brl?.initial || 0)
      const maxBrl = Math.max(...curve, brl?.initial || 0)
      const range = maxBrl - minBrl || 1

      const stats: [string, string, string][] = [
        [t('stat_winrate'), `${winRate}%`, winRate !== '—' && +winRate >= 50 ? 'var(--color-green)' : 'var(--color-orange)'],
        [t('stat_roi'), roiStr === '—' ? '—' : `${parseFloat(roiStr) >= 0 ? '+' : ''}${roiStr}%`, roiColor],
        [t('lbl_pnl'), `${totalPnl >= 0 ? '+' : ''}€${totalPnl.toFixed(2)}`, pnlColor],
        [t('stat_avgodds'), `@${avgOdds}`, 'var(--color-gold)'],
        [t('stat_streak'), streakStr, streakType === 'won' ? 'var(--color-green)' : 'var(--color-orange)'],
        [t('stat_settled'), `${settled.length}/${hist.length}`, 'var(--color-ink)'],
      ]

      const sortedSettled = [...settled].sort((a, b2) =>
        (b2.settledAt || b2.placedAt || '') > (a.settledAt || a.placedAt || '') ? 1 : -1,
      )

      body = (
        <div>
          {settled.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {stats.map(([label, val, col]) => (
                <div key={label} className="bg-raised/60 border-line min-w-21 rounded-lg border px-3.5 py-2 text-center">
                  <div className="micro-label">{label}</div>
                  <div className="stat-num mt-0.5 text-lg" style={{ color: col }}>
                    {val}
                  </div>
                </div>
              ))}
            </div>
          )}

          {curve.length > 1 && (
            <div className="bg-raised/60 border-line mb-3 rounded-lg border px-4 py-3">
              <div className="micro-label mb-2">{t('sec_bankroll_curve')}</div>
              <div className="flex h-10 items-end gap-0.5">
                {curve.map((v, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-sm opacity-80"
                    style={{
                      height: Math.max(4, Math.round(((v - minBrl) / range) * 36)),
                      background: v >= (brl?.initial || 0) ? 'var(--color-green)' : 'var(--color-orange)',
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          {/* export / import */}
          <div className="mb-3 flex flex-wrap gap-2">
            <button
              onClick={exportBankrollJSON}
              className="border-line text-mute hover:text-ink cursor-pointer rounded-md border px-3 py-1.5 text-xs font-semibold"
            >
              {t('btn_export')}
            </button>
            <button
              onClick={() => fileRef.current?.click()}
              className="border-line text-mute hover:text-ink cursor-pointer rounded-md border px-3 py-1.5 text-xs font-semibold"
            >
              {t('btn_import')}
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="application/json"
              className="hidden"
              onChange={(e) => {
                onImportFile(e.target.files?.[0])
                e.target.value = ''
              }}
            />
          </div>

          {pending.length > 0 && (
            <div className="mb-3">
              <div className="text-amber mb-1.5 text-[11px] font-bold tracking-wider uppercase">
                {t('sec_pending')} ({pending.length})
              </div>
              {pending.map((b) => (
                <HistRow key={b.id} bet={b} results={data.results} />
              ))}
            </div>
          )}

          {settled.length > 0 && (
            <div>
              <div className="micro-label mb-1.5 font-bold">
                {t('sec_settled')} ({settled.length})
              </div>
              {sortedSettled.map((b) => (
                <HistRow key={b.id} bet={b} results={data.results} />
              ))}
            </div>
          )}
        </div>
      )
    }
  }

  // ── PARLAY tab body ──
  else {
    if (slip.length < 2) {
      body = (
        <div className="px-6 py-14 text-center">
          <div className="text-4xl">🔗</div>
          <div className="text-mute mt-3 text-sm">{t('empty_parlay')}</div>
        </div>
      )
    } else {
      const combinedOdds = +slip.reduce((p, b) => p * (parseFloat(String(b.oddsUser)) || 1), 1).toFixed(3)
      body = (
        <div>
          <div className="flex flex-col">
            {slip.map((b, i) => (
              <div key={b.id} className="border-line flex items-center gap-3 border-b py-2.5">
                <span className="stat-num text-mute text-lg">{i + 1}</span>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-semibold">
                    {b.homeName} vs {b.awayName}
                  </div>
                  <div className="mt-0.5 text-[11px]" style={{ color: outcomeColor(b.outcome) }}>
                    {outcomeLabel(b.outcome)}
                  </div>
                </div>
                <div className="stat-num text-lg">@{b.oddsUser}</div>
                <button
                  onClick={() => removeFromSlip(b.id)}
                  className="border-line text-mute hover:text-red h-7 w-7 cursor-pointer rounded-md border text-xs"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          <div className="bg-raised/60 mt-5 flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl px-4 py-4">
            <div>
              <div className="micro-label">{t('lbl_comb_odds')}</div>
              <div className="stat-num text-gold text-3xl">@{combinedOdds}</div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-mute text-[13px]">{t('lbl_stake')}</span>
              <NumField
                value={parlayStake}
                min={1}
                step={1}
                fallback={10}
                width="w-22"
                onCommit={setParlayStake}
              />
            </div>
            <div>
              <div className="micro-label">{t('lbl_pot_ret')}</div>
              <div className="stat-num text-green text-xl">€{(parlayStake * combinedOdds).toFixed(2)}</div>
            </div>
            <div className="ml-auto flex flex-wrap gap-2">
              <button
                disabled={ollama.busy}
                onClick={() => ollama.analyze(slip, brl, true, parlayStake)}
                className="bg-surface border-line cursor-pointer rounded-lg border px-4 py-2.5 text-[13px] font-semibold disabled:opacity-60"
              >
                {ollama.busy ? t('btn_analyzing') : t('btn_ollama')}
              </button>
              <button
                disabled={!brl}
                onClick={doPlaceParlay}
                className="bg-green font-display cursor-pointer rounded-lg px-5 py-2.5 text-base font-bold text-black disabled:opacity-50"
              >
                {t('btn_place_comb')}
              </button>
            </div>
          </div>
          <OllamaBox text={ollama.text} err={ollama.err} />
        </div>
      )
    }
  }

  return (
    <div className="mx-auto max-w-210">
      {header}
      <div className="panel overflow-hidden">
        <div className="border-line flex border-b">
          {tabBtn(
            'active',
            t('tab_active'),
            slip.length > 0 && (
              <span className="bg-teal rounded-full px-1.5 py-px text-[10px] font-bold text-black">{slip.length}</span>
            ),
          )}
          {tabBtn(
            'history',
            t('tab_history'),
            hist.length > 0 && (
              <span className="bg-raised border-line rounded-full border px-1.5 py-px text-[10px]">{hist.length}</span>
            ),
          )}
          {tabBtn(
            'combined',
            t('tab_combined'),
            slip.length >= 2 && (
              <span className="bg-gold rounded-full px-1.5 py-px text-[10px] font-bold text-black">{slip.length}</span>
            ),
          )}
        </div>
        <div className="px-4 py-4 md:px-5">{body}</div>
      </div>
    </div>
  )
}

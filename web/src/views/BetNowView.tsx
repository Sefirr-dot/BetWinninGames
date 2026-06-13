// ─────────────────────────────────────────────────────────────
// BETNOW view — prioritized value picks with capped exposure
// (monolith renderBetNowView).
// ─────────────────────────────────────────────────────────────
import { useMemo, useState } from 'react'
import { t, tf } from '../i18n'
import { outcomeLabel, shortDate } from '../lib/format'
import { useBankroll } from '../lib/storage'
import { useApp } from '../state/AppContext'
import { EmptyState, LeagueBadge } from '../components/ui'
import { BankrollInput } from './ValueBetsView'
import type { MatchPred, ValueBet } from '../types'

interface Entry {
  m: MatchPred
  vb: ValueBet
}

export default function BetNowView() {
  const { data, openMatch } = useApp()
  const brl = useBankroll()
  const [bankroll, setBankroll] = useState<number>(() => brl?.current ?? 100)

  const { high, low } = useMemo(() => {
    const high: Entry[] = []
    const low: Entry[] = []
    data.matches.forEach((m) => {
      if (!m.valueBets || !m.valueBets.length) return
      const best = [...m.valueBets].sort((a, b) => b.edge - a.edge)[0]
      if (m.stars >= 4) high.push({ m, vb: best })
      else if (m.stars === 3) low.push({ m, vb: best })
    })
    high.sort((a, b) => b.vb.edge - a.vb.edge)
    low.sort((a, b) => b.vb.edge - a.vb.edge)
    return { high, low }
  }, [data.matches])

  const all = [...high, ...low]
  if (all.length === 0) {
    return <EmptyState icon="💰" msg={`${t('bn_empty')}\n${t('bn_empty_sub')}`} />
  }

  const rawKelly = all.reduce((s, { vb }) => s + vb.kelly / 100, 0)
  const capFactor = Math.min(1, (bankroll * 0.1) / Math.max(0.01, rawKelly * bankroll))
  const totalStake = (rawKelly * bankroll * capFactor).toFixed(2)
  const totalPct = (rawKelly * 100 * capFactor).toFixed(1)

  const betCard = ({ m, vb }: Entry, priority: boolean) => {
    const stake = ((vb.kelly / 100) * bankroll * capFactor).toFixed(2)
    const tierColor = priority ? 'var(--color-green)' : 'var(--color-amber)'
    const tierLabel = priority ? (m.stars === 5 ? t('bn_max_conf') : t('bn_high_prio')) : t('bn_low_prio')
    return (
      <div
        key={`${m.date}|${m.home}|${m.away}`}
        className="panel flex cursor-pointer flex-wrap items-stretch gap-4 p-4 transition-colors hover:brightness-110"
        style={{ borderLeft: `3px solid ${tierColor}` }}
        onClick={() => openMatch(m)}
      >
        <div className="min-w-0 flex-1">
          <div className="mb-1.5 flex flex-wrap items-center gap-2">
            <LeagueBadge league={m.league} />
            <span className="text-[10px] font-bold tracking-wider uppercase" style={{ color: tierColor }}>
              {tierLabel}
            </span>
            <span className="text-mute text-[11px]">
              {shortDate(m.date)}
              {m.time ? ' · ' + m.time : ''}
            </span>
          </div>
          <div className="font-display text-lg font-bold tracking-wide">
            {m.home} vs {m.away}
          </div>
          <div className="mt-1.5 text-[15px] font-bold">▶ {outcomeLabel(vb.outcome)}</div>
          <div className="mt-2.5 flex flex-wrap gap-5">
            <div>
              <div className="micro-label">{t('vb_market_odds')}</div>
              <div className="stat-num text-gold text-2xl">@{vb.bkOdds}</div>
              {vb.bestBook && <div className="text-mute mt-px text-[10px]">{vb.bestBook}</div>}
            </div>
            <div>
              <div className="micro-label">{t('lbl_edge')}</div>
              <div className="stat-num text-green text-2xl">+{vb.edge}%</div>
            </div>
            <div>
              <div className="micro-label">{t('vb_model')}</div>
              <div className="stat-num text-2xl">{vb.modelProb}%</div>
            </div>
            <div>
              <div className="micro-label">{t('vb_implied')}</div>
              <div className="stat-num text-mute text-2xl">{vb.impliedProb}%</div>
            </div>
          </div>
        </div>
        <div className="bg-green/8 border-green/20 flex min-w-25 shrink-0 flex-col items-center justify-center gap-1 rounded-xl border px-4 py-3 text-center">
          <div className="micro-label">{t('lbl_stake_word')}</div>
          <div className="stat-num text-green text-3xl">€{stake}</div>
          <div className="text-mute text-[10px]">Kelly {vb.kelly}%</div>
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-230">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div>
          <div className="font-display text-2xl font-extrabold tracking-wide">{t('bn_title')}</div>
          <div className="text-mute text-xs">{tf('bn_subtitle', high.length, low.length)}</div>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2.5">
          <BankrollInput value={bankroll} onChange={setBankroll} />
          <div className="bg-green/8 border-green/25 rounded-lg border px-3.5 py-1.5 text-center">
            <div className="micro-label">{t('bn_exposure_today')}</div>
            <div className="stat-num text-green text-xl">
              €{totalStake} <span className="text-mute text-[13px] font-normal">({totalPct}%)</span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-2.5">
        {high.map((b) => betCard(b, true))}
        {low.length > 0 && (
          <>
            <div className="micro-label mt-3 px-1 font-bold">{t('bn_secondary')}</div>
            {low.map((b) => betCard(b, false))}
          </>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// VALUE view — all value bets sorted by edge
// (monolith renderValueBetsView).
// ─────────────────────────────────────────────────────────────
import { useMemo, useState } from 'react'
import { t } from '../i18n'
import { outcomeLabel, shortDate } from '../lib/format'
import { useBankroll } from '../lib/storage'
import { useApp } from '../state/AppContext'
import { useAddToSlip } from '../state/useAddToSlip'
import { EdgeBadge, EmptyState, KellyBadge, LeagueBadge, SharpnessBadge, StarsInline } from '../components/ui'
import type { MatchPred, ValueBet } from '../types'

export function BankrollInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const [text, setText] = useState(String(value))
  const commit = () => onChange(Math.max(1, parseInt(text) || 100))
  return (
    <div className="bg-raised border-line flex items-center gap-1.5 rounded-lg border px-3 py-1.5">
      <span className="micro-label">{t('lbl_bankroll')}</span>
      <span className="text-mute text-[13px]">€</span>
      <input
        type="number"
        min={1}
        max={100000}
        step={10}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            commit()
            ;(e.target as HTMLInputElement).blur()
          }
        }}
        className="stat-num w-20 bg-transparent text-lg outline-none"
      />
    </div>
  )
}

export default function ValueBetsView() {
  const { data, openMatch } = useApp()
  const add = useAddToSlip()
  const brl = useBankroll()
  const [bankroll, setBankroll] = useState<number>(() => brl?.current ?? 100)

  const all = useMemo(() => {
    const out: { m: MatchPred; vb: ValueBet }[] = []
    data.matches.forEach((m) => {
      ;(m.valueBets || []).forEach((vb) => out.push({ m, vb }))
    })
    return out.sort((a, b) => b.vb.edge - a.vb.edge)
  }, [data.matches])

  if (all.length === 0) {
    return <EmptyState icon="💎" msg={`${t('vb_empty')}\n${t('vb_empty_sub')}`} />
  }

  const totalKelly = all.reduce((s, { vb }) => s + vb.kelly / 100, 0)
  const totalStake = Math.min(totalKelly * bankroll, bankroll).toFixed(2)

  return (
    <div className="mx-auto max-w-230">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div>
          <div className="font-display text-2xl font-extrabold tracking-wide">{t('vb_title')}</div>
          <div className="text-mute text-xs">
            {all.length} {t('vb_subtitle')}
          </div>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2.5">
          <BankrollInput value={bankroll} onChange={setBankroll} />
          <div className="bg-green/8 border-green/25 rounded-lg border px-3.5 py-1.5 text-center">
            <div className="micro-label">{t('vb_total_stake')}</div>
            <div className="stat-num text-green text-xl">€{totalStake}</div>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-2.5">
        {all.map(({ m, vb }, idx) => {
          const kellyEur = ((vb.kelly / 100) * bankroll).toFixed(2)
          const slipStake = +((((vb.kellyPortfolio || vb.kelly) as number) / 100) * bankroll).toFixed(2)
          return (
            <div
              key={idx}
              className="panel hover:border-green/40 flex cursor-pointer flex-wrap items-stretch gap-4 p-4 transition-colors"
              onClick={() => openMatch(m)}
            >
              <div className="min-w-0 flex-1">
                <div className="font-display text-lg font-bold tracking-wide">
                  {m.homeShort} vs {m.awayShort}
                </div>
                <div className="text-green mt-0.5 text-[14px] font-bold">▶ {outcomeLabel(vb.outcome)}</div>
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                  <LeagueBadge league={m.league} />
                  <span className="text-mute text-[11px]">
                    {shortDate(m.date)}
                    {m.time ? ' · ' + m.time : ''}
                  </span>
                  <EdgeBadge edge={vb.edge} />
                  <KellyBadge>
                    Kelly {vb.kelly}% → €{kellyEur}
                  </KellyBadge>
                  <SharpnessBadge score={vb.sharpnessScore} />
                  {!!vb.corrDiscount && vb.corrDiscount < 0.95 && (
                    <span className="text-amber rounded-md border border-amber-400/30 bg-black/30 px-1.5 py-px text-[10px]">
                      🔗 {t('lbl_corr')} ×{vb.corrDiscount}
                    </span>
                  )}
                </div>
                <div className="mt-2.5 flex gap-5">
                  <span>
                    <div className="micro-label">{t('vb_model')}</div>
                    <div className="stat-num text-green text-lg">{vb.modelProb}%</div>
                  </span>
                  <span>
                    <div className="micro-label">{t('vb_implied')}</div>
                    <div className="stat-num text-lg">{vb.impliedProb}%</div>
                  </span>
                  {vb.pinnacleProb ? (
                    <span>
                      <div className="micro-label">{t('lbl_pinnacle')}</div>
                      <div className="stat-num text-teal text-lg">{vb.pinnacleProb}%</div>
                    </span>
                  ) : (
                    <span>
                      <div className="micro-label">{t('vb_fair')}</div>
                      <div className="stat-num text-gold text-lg">{m.fairOdds ? '@' + m.fairOdds : '—'}</div>
                    </span>
                  )}
                </div>
                {vb.clvVsPinnacle !== null && vb.clvVsPinnacle !== undefined && (
                  <div
                    className="mt-1 text-[11px]"
                    style={{ color: vb.clvVsPinnacle >= 0 ? 'var(--color-teal)' : 'var(--color-orange)' }}
                  >
                    {t('vb_clv_pin')} {vb.clvVsPinnacle >= 0 ? '+' : ''}
                    {vb.clvVsPinnacle}%
                  </div>
                )}
                {!!vb.kellyPortfolio && vb.kellyPortfolio !== vb.kelly && (
                  <div className="text-mute mt-1 text-[11px]">
                    {t('lbl_markowitz')}: <span className="text-green font-bold">{vb.kellyPortfolio}%</span>{' '}
                    <span className="opacity-60">(vs Kelly {vb.kelly}%)</span>
                  </div>
                )}
              </div>
              <div className="flex shrink-0 flex-col items-end justify-center text-right">
                <div className="stat-num text-gold text-3xl">@{vb.bkOdds}</div>
                <div className="micro-label">{t('vb_market_odds')}</div>
                {vb.bestBook && <div className="text-gold mt-0.5 text-[11px] font-semibold">{vb.bestBook}</div>}
                <div className="mt-1.5">
                  <StarsInline n={m.stars} />
                </div>
                <button
                  className="bg-teal font-display mt-2 w-full cursor-pointer rounded-md px-3 py-1.5 text-[13px] font-bold text-black"
                  onClick={(e) => {
                    e.stopPropagation()
                    add({
                      homeName: m.home || m.homeShort,
                      awayName: m.away || m.awayShort,
                      league: m.league,
                      matchDate: m.date,
                      outcome: vb.outcome,
                      suggestedOdds: vb.bkOdds,
                      kellyStake: slipStake,
                    })
                  }}
                >
                  {t('btn_add_slip')}
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// BEST view — calcBestBets ranking + suggested parlays
// (monolith renderBestBetsView).
// ─────────────────────────────────────────────────────────────
import { useMemo } from 'react'
import { t } from '../i18n'
import { calcBestBets, getSuggestedParlays } from '../lib/betting'
import type { BestBet, BestBetType, Parlay, ParlayRisk } from '../lib/betting'
import { loadParlayToSlip } from '../lib/actions'
import { shortDate } from '../lib/format'
import { useApp } from '../state/AppContext'
import { LeagueBadge, StarsInline } from '../components/ui'

const BET_TYPE_CONFIG: Record<BestBetType, { color: string; bg: string }> = {
  victoria: { color: '#22c55e', bg: 'rgba(34,197,94,0.12)' },
  over25: { color: '#3b82f6', bg: 'rgba(59,130,246,0.12)' },
  btts: { color: '#a855f7', bg: 'rgba(168,85,247,0.12)' },
}

const RISK_META: Record<ParlayRisk, { color: string; icon: string; descKey: string }> = {
  safe: { color: 'var(--color-green)', icon: '🟢', descKey: 'risk_safe' },
  medium: { color: 'var(--color-amber)', icon: '🟡', descKey: 'risk_medium' },
  risky: { color: 'var(--color-orange)', icon: '🟠', descKey: 'risk_risky' },
  extreme: { color: 'var(--color-red)', icon: '🔴', descKey: 'risk_extreme' },
}

function betLabel(b: BestBet): string {
  if (b.type === 'victoria') return `${t('victory_word')} ${b.team}`
  if (b.type === 'over25') return t('card_over25')
  return t('out_btts')
}

function ParlayBox({ parlay }: { parlay: Parlay }) {
  const { setActive, setSlipTab, setBankrollModal } = useApp()
  const meta = RISK_META[parlay.riskLevel] || RISK_META.medium

  const addToBankroll = () => {
    const res = loadParlayToSlip(
      parlay.legs.map((b) => ({
        homeName: b.match.home || b.match.homeShort,
        awayName: b.match.away || b.match.awayShort,
        league: b.match.league,
        matchDate: b.match.date,
        outcome: b.outcomeKey,
        oddsUser: b.fairOdds || 2.0,
      })),
    )
    if (res === 'NO_BANKROLL') {
      setBankrollModal(true)
      return
    }
    setSlipTab('combined')
    setActive('SLIP')
  }

  return (
    <div className="mb-3">
      <div className="font-display mb-1.5 text-base font-bold tracking-wide" style={{ color: meta.color }}>
        {meta.icon} {parlay.title}
        <span className="text-mute ml-2 text-[11px] font-normal">{t(meta.descKey)}</span>
      </div>
      <div className="panel overflow-hidden">
        <div className="px-4">
          {parlay.legs.map((b, i) => (
            <div key={i} className="border-line flex items-center gap-3 border-b py-2.5 last:border-b-0">
              <span
                className="stat-num flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[12px]"
                style={{ background: meta.color + '20', color: meta.color }}
              >
                {i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-semibold">
                  {b.match.homeShort} vs {b.match.awayShort}
                </div>
                <div className="text-mute mt-0.5 flex flex-wrap items-center gap-1.5 text-[11px]">
                  <LeagueBadge league={b.match.league} />
                  {shortDate(b.match.date)}
                  {b.match.time ? ' · ' + b.match.time : ''}
                  <StarsInline n={b.match.stars} />
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <span className="text-[12px] font-semibold">{betLabel(b)}</span>
                <span className="stat-num text-[15px]" style={{ color: meta.color }}>
                  {b.prob}%
                </span>
              </div>
            </div>
          ))}
        </div>
        <div className="bg-raised/50 border-line flex flex-wrap items-center gap-x-6 gap-y-2 border-t px-4 py-3">
          <div>
            <div className="micro-label">{t('lbl_comb_prob')}</div>
            <div className="stat-num text-lg" style={{ color: meta.color }}>
              {parlay.combinedProb}%
            </div>
          </div>
          {parlay.combinedOdds && (
            <div>
              <div className="micro-label">{t('lbl_comb_odds')}</div>
              <div className="stat-num text-gold text-lg">@{parlay.combinedOdds}</div>
            </div>
          )}
          <button
            onClick={addToBankroll}
            className="bg-teal font-display ml-auto cursor-pointer rounded-lg px-4 py-1.5 text-sm font-bold text-black"
          >
            {t('btn_add_bkrl')}
          </button>
          <div className="text-mute/70 w-full text-right text-[10px]">{t('parlay_disclaimer')}</div>
        </div>
      </div>
    </div>
  )
}

export default function BestBetsView() {
  const { data, openMatch } = useApp()
  const bets = useMemo(() => calcBestBets(data.matches), [data.matches])
  const parlays = useMemo(() => getSuggestedParlays(bets, data.matches), [bets, data.matches])
  const maxScore = bets.length ? bets[0].score : 1

  const categories: { type: BestBetType; title: string; icon: string }[] = [
    { type: 'victoria', title: t('best_victories'), icon: '🏆' },
    { type: 'over25', title: t('card_over25'), icon: '⚽' },
    { type: 'btts', title: t('out_btts'), icon: '🎯' },
  ]

  return (
    <div className="mx-auto max-w-260">
      {parlays.length > 0 && (
        <div className="mb-6">
          <div className="micro-label mb-3 font-bold">{t('best_parlays_title')}</div>
          {parlays.map((p) => (
            <ParlayBox key={p.title} parlay={p} />
          ))}
        </div>
      )}

      <div className="micro-label mb-3 font-bold">{t('best_ranking')}</div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 2xl:grid-cols-3">
        {categories.map((cat) => {
          const catBets = bets.filter((b) => b.type === cat.type)
          if (!catBets.length) return null
          const cfg = BET_TYPE_CONFIG[cat.type]
          return (
            <div key={cat.type}>
              <div className="font-display mb-2 text-base font-bold tracking-wide" style={{ color: cfg.color }}>
                {cat.icon} {cat.title}{' '}
                <span className="text-mute text-xs font-normal">
                  {catBets.length} {t('best_bets_word')}
                </span>
              </div>
              <div className="flex flex-col gap-1.5">
                {catBets.map((b, i) => {
                  const rankColor = i === 0 ? '#f0b429' : i === 1 ? '#9ca3af' : i === 2 ? '#b45309' : 'var(--color-mute)'
                  const barW = Math.round((b.score / maxScore) * 100)
                  return (
                    <div
                      key={`${b.match.date}|${b.match.home}|${b.type}`}
                      className="panel hover:border-gold/40 relative cursor-pointer overflow-hidden px-3.5 py-2.5 transition-colors"
                      onClick={() => openMatch(b.match)}
                    >
                      <div className="flex items-center gap-3">
                        <div className="stat-num w-8 shrink-0 text-base" style={{ color: rankColor }}>
                          #{i + 1}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="text-[13px] font-semibold">
                            {b.match.homeShort} <span className="text-mute">vs</span> {b.match.awayShort}
                          </div>
                          <div className="text-[12px] font-semibold" style={{ color: cfg.color }}>
                            ▶ {betLabel(b)}
                          </div>
                          <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
                            <LeagueBadge league={b.match.league} />
                            <span className="text-mute text-[11px]">
                              {shortDate(b.match.date)}
                              {b.match.time ? ' · ' + b.match.time : ''}
                            </span>
                            <StarsInline n={b.match.stars} />
                          </div>
                        </div>
                        <div className="shrink-0 text-right">
                          <div className="stat-num text-lg">{b.prob}%</div>
                          {b.fairOdds && <div className="text-gold stat-num text-[12px]">@{b.fairOdds}</div>}
                        </div>
                      </div>
                      <div className="absolute bottom-0 left-0 h-0.5 w-full bg-black/20">
                        <div className="h-full" style={{ width: `${barW}%`, background: cfg.color }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

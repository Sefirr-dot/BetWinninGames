// ─────────────────────────────────────────────────────────────
// Match card — port of the monolith renderCard().
// ─────────────────────────────────────────────────────────────
import { t } from '../i18n'
import { winnerTeam } from '../lib/betting'
import { outcomeColor, starsLabel } from '../lib/format'
import { useApp } from '../state/AppContext'
import { useAddToSlip } from '../state/useAddToSlip'
import type { MatchPred, Outcome } from '../types'
import { FormDots, LeagueBadge, Stars, YesNo } from './ui'

export function ProbBar({ m, height = 24 }: { m: MatchPred; height?: number }) {
  const winner = winnerTeam(m)
  const total = m.prob1 + m.probX + m.prob2
  const w1 = (m.prob1 / total) * 100
  const wX = (m.probX / total) * 100
  const w2 = (m.prob2 / total) * 100
  const seg = (w: number, color: string, label: string, show: boolean, win: boolean) => (
    <div
      className="font-display flex items-center justify-center overflow-hidden text-[11px] font-bold whitespace-nowrap text-white/90"
      style={{
        width: `${w.toFixed(1)}%`,
        background: color,
        opacity: win ? 1 : 0.55,
        boxShadow: win ? 'inset 0 0 0 1px rgba(255,255,255,0.25)' : undefined,
      }}
    >
      {show ? label : ''}
    </div>
  )
  return (
    <div className="flex w-full overflow-hidden rounded-md" style={{ height }}>
      {seg(w1, 'var(--color-home)', `${m.prob1}%`, m.prob1 > 10, winner === 'home')}
      {seg(wX, 'var(--color-draw)', `${m.probX}%`, m.probX > 8, false)}
      {seg(w2, 'var(--color-away)', `${m.prob2}%`, m.prob2 > 10, winner === 'away')}
    </div>
  )
}

function StatBox({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="bg-raised/60 border-line/60 rounded-md border px-2.5 py-2">
      <div className="micro-label mb-1">{label}</div>
      <div className="flex items-center gap-1.5 text-[13px] font-semibold">{children}</div>
    </div>
  )
}

export default function MatchCard({ m }: { m: MatchPred }) {
  const { data, openMatch } = useApp()
  const add = useAddToSlip()
  const winner = winnerTeam(m)
  const untracked = m.stars < (data.meta.minStarsTracked ?? 3)
  const hasValue = !!(m.valueBets && m.valueBets.length)

  const slipOpts: { outcome: Outcome; label: string; prob: number }[] = [
    { outcome: 'home', label: `${m.homeShort} ${t('wins_lbl')}`, prob: m.prob1 },
    { outcome: 'draw', label: t('out_draw'), prob: m.probX },
    { outcome: 'away', label: `${m.awayShort} ${t('wins_lbl')}`, prob: m.prob2 },
    ...(m.over25yn ? [{ outcome: 'over25' as Outcome, label: t('out_over25'), prob: m.over25 }] : []),
    ...(m.bttsyn ? [{ outcome: 'btts' as Outcome, label: t('out_btts'), prob: m.btts }] : []),
  ]

  return (
    <div
      className="panel rise-in hover:border-gold/40 cursor-pointer p-4 transition-colors"
      onClick={() => openMatch(m)}
    >
      {/* header */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="font-display text-lg leading-tight font-bold tracking-wide">
            {m.homeShort} <span className="text-mute font-normal">vs</span> {m.awayShort}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <LeagueBadge league={m.league} />
            {m.time && <span className="text-mute text-[11px]">🕐 {m.time}</span>}
            {hasValue && (
              <span className="bg-green/12 text-green rounded px-1.5 py-px text-[10px] font-bold">⚡ VALUE</span>
            )}
            {untracked && (
              <span
                className="text-mute/70 border-line rounded border border-dashed px-1.5 py-px text-[9px] tracking-wide uppercase"
                title={t('badge_untracked_tip')}
              >
                {t('badge_untracked')}
              </span>
            )}
          </div>
          {m.homePos && m.awayPos && (
            <div className="text-mute mt-1 text-[11px]">
              <span className="bg-raised rounded px-1 py-px">{m.homePos}°</span>
              <span className="mx-1 opacity-60">vs</span>
              <span className="bg-raised rounded px-1 py-px">{m.awayPos}°</span>
            </div>
          )}
        </div>
        <div className="shrink-0 text-right">
          <Stars n={m.stars} />
          <div className="text-mute mt-px text-[10px]">{starsLabel(m.stars)}</div>
          {m.fairOdds && (
            <div className="text-gold border-gold/30 bg-gold/10 stat-num mt-1 inline-block rounded border px-1.5 py-px text-[12px]">
              @{m.fairOdds}
            </div>
          )}
        </div>
      </div>

      {/* prob bar */}
      <div className="mt-3">
        <div className="text-mute mb-1 flex justify-between text-[10px]">
          <span>1 — {m.homeShort}</span>
          <span>X — {t('out_draw')}</span>
          <span>{m.awayShort} — 2</span>
        </div>
        <ProbBar m={m} />
        <div className="stat-num mt-1 flex justify-between text-[13px]">
          <span style={{ color: 'var(--color-home)', opacity: winner === 'home' ? 1 : 0.6 }}>{m.prob1}%</span>
          <span className="text-mute">{m.probX}%</span>
          <span style={{ color: 'var(--color-away)', opacity: winner === 'away' ? 1 : 0.6 }}>{m.prob2}%</span>
        </div>
      </div>

      {/* stats grid */}
      <div className="mt-3 grid grid-cols-2 gap-1.5">
        <StatBox label={t('card_over25')}>
          <YesNo yes={m.over25yn} />
          <span className="text-mute text-[11px]">{m.over25}%</span>
        </StatBox>
        <StatBox label={t('card_btts')}>
          <YesNo yes={m.bttsyn} />
          <span className="text-mute text-[11px]">{m.btts}%</span>
        </StatBox>
        <StatBox label={t('card_corners')}>
          {m.corners !== null ? (
            <>
              <span>{m.cornersOver ? 'Over' : 'Under'} 9.5</span>
              <span className="text-mute text-[11px]">
                {m.corners} {t('card_exp')}
              </span>
            </>
          ) : (
            <span className="text-mute">—</span>
          )}
        </StatBox>
        <StatBox label={t('card_cards')}>
          {m.cards !== null ? (
            <>
              <span>{m.cardsOver ? 'Over' : 'Under'} 3.5</span>
              <span className="text-mute text-[11px]">
                {m.cards} {t('card_exp')}
              </span>
            </>
          ) : (
            <span className="text-mute">—</span>
          )}
        </StatBox>
        {m.goalsHome !== null && (
          <StatBox label={t('card_goals')}>
            <span style={{ color: 'var(--color-home)' }}>{m.goalsHome}</span>
            <span className="text-mute">–</span>
            <span style={{ color: 'var(--color-away)' }}>{m.goalsAway}</span>
          </StatBox>
        )}
        {m.bestScore !== null && (
          <StatBox label={t('card_score')}>
            <span className="text-gold stat-num text-[15px]">{m.bestScore}</span>
            <span className="text-mute text-[11px]">({m.bestScoreProb}%)</span>
          </StatBox>
        )}
      </div>

      {/* detail row */}
      {(m.eloConf !== null || m.formHome !== null) && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {m.eloConf !== null && (
            <div className="bg-raised/60 border-line/60 rounded-md border px-2 py-1">
              <div className="micro-label">{t('card_elo_conf')}</div>
              <div className="stat-num text-[13px]">{m.eloConf}%</div>
            </div>
          )}
          {m.formHome !== null && (
            <>
              <div className="bg-raised/60 border-line/60 rounded-md border px-2 py-1">
                <div className="micro-label">{m.homeShort}</div>
                <div className="mt-0.5">
                  <FormDots form={m.formHome} />
                </div>
              </div>
              <div className="bg-raised/60 border-line/60 rounded-md border px-2 py-1">
                <div className="micro-label">{m.awayShort}</div>
                <div className="mt-0.5">
                  <FormDots form={m.formAway} />
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* add-to-slip buttons */}
      <div
        className="border-line mt-2.5 flex flex-wrap gap-1.5 border-t pt-2.5"
        onClick={(e) => e.stopPropagation()}
      >
        {slipOpts.map((opt) => {
          const odds =
            m.fairOdds && opt.outcome === (m.prob1 >= m.prob2 ? 'home' : 'away')
              ? m.fairOdds
              : +(100 / opt.prob).toFixed(2)
          return (
            <button
              key={opt.outcome}
              className="bg-raised border-line hover:border-teal/60 flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-[11px] font-semibold transition-colors"
              onClick={() =>
                add({
                  homeName: m.home || m.homeShort,
                  awayName: m.away || m.awayShort,
                  league: m.league,
                  matchDate: m.date,
                  outcome: opt.outcome,
                  suggestedOdds: odds,
                  kellyStake: null,
                })
              }
            >
              <span style={{ color: outcomeColor(opt.outcome) }}>+</span> {opt.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

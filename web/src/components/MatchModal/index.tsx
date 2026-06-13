// ─────────────────────────────────────────────────────────────
// Match detail modal — port of the monolith renderMatchModal().
// ─────────────────────────────────────────────────────────────
import { useEffect } from 'react'
import { t, tagLabel } from '../../i18n'
import { winnerTeam } from '../../lib/betting'
import { useApp } from '../../state/AppContext'
import type { MatchPred } from '../../types'
import { ProbBar } from '../MatchCard'
import { LeagueBadge, Stars, YesNo } from '../ui'
import ScoreGrid from './ScoreGrid'
import ValueBetsSection from './ValueBetsSection'

// Display weights as shown in the legacy modal
const MODEL_WEIGHTS_LBL: Record<string, string> = { dc: '45%', elo: '20%', form: '20%', h2h: '5%' }

function modelName(key: string): string {
  if (key === 'dc') return 'Dixon-Coles'
  if (key === 'elo') return 'Elo'
  if (key === 'form') return t('model_form')
  if (key === 'h2h') return 'H2H'
  return key
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-line border-b px-5 py-4 last:border-b-0">
      <div className="micro-label mb-2.5 font-bold tracking-[0.1em]">{title}</div>
      {children}
    </div>
  )
}

function StatChip({ value, label, color }: { value: React.ReactNode; label: string; color?: string }) {
  return (
    <div className="bg-raised/70 border-line/60 min-w-20 rounded-md border px-2.5 py-1.5 text-center">
      <div className="stat-num text-base" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="micro-label mt-0.5">{label}</div>
    </div>
  )
}

function fatColor(d: number | null): string {
  if (d === null || d === undefined) return 'var(--color-mute)'
  return d < 4 ? 'var(--color-red)' : d < 7 ? 'var(--color-amber)' : 'var(--color-green)'
}

export default function MatchModal() {
  const { modalMatch, closeMatch } = useApp()

  useEffect(() => {
    if (!modalMatch) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeMatch()
    }
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [modalMatch, closeMatch])

  if (!modalMatch) return null
  const m: MatchPred = modalMatch
  const winner = winnerTeam(m)
  const hasAi = !!(m.aiNote || (m.aiFactors && m.aiFactors.length))

  return (
    <div
      className="fixed inset-0 z-[9000] flex items-start justify-center overflow-y-auto bg-black/75 p-3 md:p-8"
      onClick={closeMatch}
    >
      <div
        className="modal-pop panel w-full max-w-180 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="border-line bg-raised/40 sticky top-0 z-10 flex items-start justify-between gap-3 border-b px-5 py-4 backdrop-blur-sm">
          <div>
            <div className="font-display text-2xl leading-tight font-extrabold tracking-wide">
              {m.home} <span className="text-mute font-normal">vs</span> {m.away}
            </div>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <LeagueBadge league={m.league} />
              {m.time && <span className="text-mute text-[11px]">🕐 {m.time}</span>}
              <Stars n={m.stars} size={14} />
              {m.homePos && m.awayPos && (
                <span className="bg-raised text-mute rounded px-1.5 py-px text-[11px]">
                  {m.homePos}° vs {m.awayPos}°
                </span>
              )}
            </div>
          </div>
          <button
            className="border-line text-mute hover:text-ink h-8 w-8 shrink-0 cursor-pointer rounded-md border text-sm"
            onClick={closeMatch}
            aria-label="close"
          >
            ✕
          </button>
        </div>

        {/* ensemble */}
        <Section title={t('modal_ensemble')}>
          <ProbBar m={m} height={32} />
          <div className="stat-num mt-1.5 flex justify-between text-sm">
            <span style={{ color: 'var(--color-home)', opacity: winner === 'home' ? 1 : 0.6 }}>
              {m.homeShort} {m.prob1}%
            </span>
            <span className="text-mute">
              {t('out_draw')} {m.probX}%
            </span>
            <span style={{ color: 'var(--color-away)', opacity: winner === 'away' ? 1 : 0.6 }}>
              {m.awayShort} {m.prob2}%
            </span>
          </div>
          {m.fairOdds && (
            <div className="text-mute mt-2 text-xs">
              {t('modal_fair')} <span className="text-gold font-bold">@{m.fairOdds}</span>
              {m.marketOdds && (
                <>
                  {' · '}
                  {t('modal_market')} <span className="font-bold">@{m.marketOdds}</span>
                </>
              )}
            </div>
          )}
        </Section>

        {/* context tags */}
        {m.contextTags && m.contextTags.length > 0 && (
          <Section title={t('modal_context')}>
            <div className="flex flex-wrap gap-1">
              {m.contextTags.map((tag) => (
                <span
                  key={tag}
                  className="bg-teal/12 border-teal/30 text-teal m-0.5 inline-block rounded-md border px-2 py-0.5 text-[11px]"
                >
                  {tagLabel(tag)}
                </span>
              ))}
            </div>
          </Section>
        )}

        {/* sub-models */}
        {m.subModels && Object.keys(m.subModels).length > 0 && (
          <Section title={t('modal_submodels')}>
            <div className="thin-scroll overflow-x-auto">
              <table className="w-full text-left text-[13px]">
                <thead>
                  <tr className="micro-label">
                    <th className="py-1 pr-2 font-semibold">{t('th_model')}</th>
                    <th className="py-1 pr-2 font-semibold">{t('th_weight')}</th>
                    <th className="py-1 pr-2 font-semibold" style={{ color: '#60a5fa' }}>
                      {t('th_local')}
                    </th>
                    <th className="py-1 pr-2 font-semibold">{t('th_draw')}</th>
                    <th className="py-1 pr-2 font-semibold" style={{ color: '#c084fc' }}>
                      {t('th_visit')}
                    </th>
                    <th className="py-1 font-semibold">{t('th_dist')}</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(m.subModels).map(([key, v]) => (
                    <tr key={key} className="border-line/50 border-t">
                      <td className="py-1.5 pr-2 font-semibold">{modelName(key)}</td>
                      <td className="text-mute py-1.5 pr-2 text-[10px]">{MODEL_WEIGHTS_LBL[key] || ''}</td>
                      <td className="py-1.5 pr-2" style={{ color: '#60a5fa' }}>
                        {v.p1}%
                      </td>
                      <td className="text-mute py-1.5 pr-2">{v.pX}%</td>
                      <td className="py-1.5 pr-2" style={{ color: '#c084fc' }}>
                        {v.p2}%
                      </td>
                      <td className="py-1.5">
                        <div className="bg-raised flex h-2 w-full min-w-24 overflow-hidden rounded">
                          <div style={{ width: `${v.p1}%`, background: 'var(--color-home)' }} />
                          <div style={{ width: `${v.pX}%`, background: 'var(--color-draw)' }} />
                          <div style={{ width: `${v.p2}%`, background: 'var(--color-away)' }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        )}

        {/* elo */}
        {m.eloHome !== null && m.eloAway !== null && (
          <Section title={t('modal_elo')}>
            <div className="flex flex-wrap gap-1.5">
              <StatChip value={m.eloHome} label={m.homeShort} color="var(--color-home)" />
              <StatChip
                value={(m.eloHome - m.eloAway > 0 ? '+' : '') + (m.eloHome - m.eloAway)}
                label={t('modal_elo_diff')}
                color={m.eloHome - m.eloAway > 0 ? 'var(--color-green)' : 'var(--color-red)'}
              />
              <StatChip value={m.eloAway} label={m.awayShort} color="var(--color-away)" />
            </div>
          </Section>
        )}

        {/* stats */}
        <Section title={t('modal_stats')}>
          <div className="flex flex-wrap gap-1.5">
            <StatChip value={<YesNo yes={m.over25yn} />} label={`Over 2.5 (${m.over25}%)`} />
            <StatChip value={<YesNo yes={m.bttsyn} />} label={`BTTS (${m.btts}%)`} />
            {!!m.over15 && <StatChip value={`${m.over15}%`} label="Over 1.5" color="var(--color-green)" />}
            {!!m.over35 && (
              <StatChip
                value={`${m.over35}%`}
                label="Over 3.5"
                color={m.over35 >= 50 ? 'var(--color-teal)' : 'var(--color-mute)'}
              />
            )}
            {!!m.over45 && <StatChip value={`${m.over45}%`} label="Over 4.5" color="var(--color-mute)" />}
            {!!m.bttsAndOver25 && (
              <StatChip value={`${m.bttsAndOver25}%`} label="BTTS + O2.5" color="var(--color-amber)" />
            )}
            <StatChip value={m.corners !== null ? m.corners : '—'} label={`${t('card_corners')} ${t('card_exp')}`} />
            {m.cards !== null && (
              <StatChip
                value={`${m.cardsHome}+${m.cardsAway}`}
                label={`${t('card_cards')} ${t('card_exp')}`}
                color="var(--color-orange)"
              />
            )}
            {m.goalsHome !== null && (
              <StatChip
                value={
                  <>
                    <span style={{ color: 'var(--color-home)' }}>{m.goalsHome}</span>
                    <span className="text-mute">-</span>
                    <span style={{ color: 'var(--color-away)' }}>{m.goalsAway}</span>
                  </>
                }
                label={t('card_goals')}
              />
            )}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <StatChip
              value={m.fatigueHome !== null ? `${m.fatigueHome}d` : '—'}
              label={`${t('modal_rest')} ${m.homeShort}`}
              color={fatColor(m.fatigueHome)}
            />
            <StatChip
              value={m.fatigueAway !== null ? `${m.fatigueAway}d` : '—'}
              label={`${t('modal_rest')} ${m.awayShort}`}
              color={fatColor(m.fatigueAway)}
            />
            {m.bestScore && (
              <StatChip
                value={m.bestScore}
                label={`${t('modal_best_score')} (${m.bestScoreProb}%)`}
                color="var(--color-gold)"
              />
            )}
            {m.eloConf !== null && <StatChip value={`${m.eloConf}%`} label={t('card_elo_conf')} />}
          </div>
          {(!!m.ahHomeMinus1Win || !!m.ahAwayPlus1Win) && (
            <div className="bg-raised/60 mt-2.5 rounded-lg p-2.5">
              <div className="micro-label mb-2">{t('modal_ah')}</div>
              <div className="flex flex-wrap gap-4">
                {!!m.ahHomeMinus1Win && (
                  <div className="text-center">
                    <div className="stat-num text-base" style={{ color: 'var(--color-home)' }}>
                      {m.ahHomeMinus1Win}%
                    </div>
                    <div className="text-mute text-[10px]">
                      {m.homeShort} -1 {t('modal_ah_win')}
                    </div>
                  </div>
                )}
                {!!m.ahHomeMinus1Push && (
                  <div className="text-center">
                    <div className="stat-num text-amber text-base">{m.ahHomeMinus1Push}%</div>
                    <div className="text-mute text-[10px]">
                      {m.homeShort} -1 {t('modal_ah_push')}
                    </div>
                  </div>
                )}
                {!!m.ahAwayPlus1Win && (
                  <div className="text-center">
                    <div className="stat-num text-base" style={{ color: 'var(--color-away)' }}>
                      {m.ahAwayPlus1Win}%
                    </div>
                    <div className="text-mute text-[10px]">
                      {m.awayShort} +1 {t('modal_ah_win')}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </Section>

        {/* score grid */}
        {m.scoreGrid && m.scoreGrid.length > 0 && (
          <Section title={t('modal_grid')}>
            <ScoreGrid m={m} />
          </Section>
        )}

        {/* h2h */}
        {m.h2h && (
          <Section title={t('modal_h2h')}>
            <div className="text-[13px] leading-relaxed">{m.h2h}</div>
          </Section>
        )}

        {/* value bets */}
        {m.valueBets && m.valueBets.length > 0 && (
          <Section title={t('modal_vb')}>
            <ValueBetsSection m={m} />
          </Section>
        )}

        {/* AI advisor */}
        {hasAi && (
          <Section title={t('modal_ai')}>
            {m.aiNote && <div className="text-[13px] leading-relaxed whitespace-pre-line">{m.aiNote}</div>}
            {m.aiFactors && m.aiFactors.length > 0 && (
              <ul className="text-mute mt-2 list-inside list-disc text-xs leading-relaxed">
                {m.aiFactors.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            )}
          </Section>
        )}
      </div>
    </div>
  )
}

// Value-bets block inside the match modal (monolith vbSection)
import { t } from '../../i18n'
import { loadBankroll } from '../../lib/storage'
import { outcomeLabel } from '../../lib/format'
import { useAddToSlip } from '../../state/useAddToSlip'
import type { MatchPred } from '../../types'
import { EdgeBadge, KellyBadge, SharpnessBadge } from '../ui'

export default function ValueBetsSection({ m }: { m: MatchPred }) {
  const add = useAddToSlip()
  if (!m.valueBets || !m.valueBets.length) return null
  const brl = loadBankroll()?.current || 100

  return (
    <div className="flex flex-col">
      {m.valueBets.map((vb, i) => {
        const kellyEur = +((((vb.kellyPortfolio || vb.kelly) as number) / 100) * brl).toFixed(2)
        return (
          <div key={i} className="border-line flex items-center justify-between border-b py-2 last:border-b-0">
            <div className="min-w-0 flex-1">
              <div className="mb-1 flex flex-wrap items-center gap-1.5">
                <span className="text-[13px] font-semibold">{outcomeLabel(vb.outcome)}</span>
                <EdgeBadge edge={vb.edge} />
                <KellyBadge>
                  Kelly {vb.kelly}%
                  {vb.kellyPortfolio && vb.kellyPortfolio !== vb.kelly && (
                    <>
                      {' · '}
                      <span className="text-green">
                        {t('lbl_markowitz')} {vb.kellyPortfolio}%
                      </span>
                    </>
                  )}
                </KellyBadge>
                {vb.sharpMoney && <span className="text-amber text-[11px]">{t('vb_sharp')}</span>}
                <SharpnessBadge score={vb.sharpnessScore} />
              </div>
              <div className="text-mute text-[10px]">
                {t('vb_model')} {vb.modelProb}% vs {t('vb_implied')} {vb.impliedProb}%
                {vb.pinnacleProb ? ` vs Pin. ${vb.pinnacleProb}%` : ''}
              </div>
              {!!vb.oddsMovement && (
                <div className="text-amber text-[10px]">
                  {t('vb_shortened')} ×{vb.oddsMovement}
                </div>
              )}
              {vb.clvVsPinnacle !== null && vb.clvVsPinnacle !== undefined && (
                <div
                  className="text-[10px]"
                  style={{ color: vb.clvVsPinnacle >= 0 ? 'var(--color-teal)' : 'var(--color-orange)' }}
                >
                  {t('vb_clv_pin')} {vb.clvVsPinnacle >= 0 ? '+' : ''}
                  {vb.clvVsPinnacle}%
                </div>
              )}
            </div>
            <div className="ml-3 flex flex-col items-end gap-1.5">
              <div className="stat-num text-xl">
                @{vb.bkOdds}
                {vb.bestBook && <span className="text-gold ml-1 text-[11px] font-semibold">{vb.bestBook}</span>}
              </div>
              <button
                className="bg-teal font-display cursor-pointer rounded-md px-3 py-1 text-[13px] font-bold text-black"
                onClick={(e) => {
                  e.stopPropagation()
                  add({
                    homeName: m.home || m.homeShort,
                    awayName: m.away || m.awayShort,
                    league: m.league,
                    matchDate: m.date,
                    outcome: vb.outcome,
                    suggestedOdds: vb.bkOdds,
                    kellyStake: kellyEur,
                  })
                }}
              >
                {t('btn_add_slip')} €{kellyEur}
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}

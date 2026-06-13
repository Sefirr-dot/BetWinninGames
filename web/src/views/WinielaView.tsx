// ─────────────────────────────────────────────────────────────
// WIN view — La Liga quiniela (monolith renderWinielaView).
// ─────────────────────────────────────────────────────────────
import { useMemo } from 'react'
import { t } from '../i18n'
import { getWiniela, winielaOutcome } from '../lib/betting'
import type { WinielaPick } from '../lib/betting'
import { loadParlayToSlip } from '../lib/actions'
import { useApp } from '../state/AppContext'
import { EmptyState } from '../components/ui'

const TYPE_COLOR: Record<WinielaPick, string> = {
  '1': 'var(--color-green)',
  X: 'var(--color-amber)',
  '2': 'var(--color-red)',
  O25: 'var(--color-teal)',
  BTTS: 'var(--color-gold)',
}

function typeLabel(pick: WinielaPick): string {
  if (pick === '1') return t('win_local')
  if (pick === '2') return t('win_away')
  if (pick === 'X') return t('out_draw')
  if (pick === 'O25') return t('win_o25')
  return t('win_btts_lbl')
}

export default function WinielaView() {
  const { data, setActive, setSlipTab, setBankrollModal } = useApp()
  const winiela = useMemo(() => getWiniela(data.matches), [data.matches])

  if (!winiela) {
    return <EmptyState icon="🇪🇸" msg={t('win_empty')} code="python main.py" >
      <div className="text-mute mt-2 text-xs">{t('win_empty_sub')}</div>
    </EmptyState>
  }

  const counts: Record<WinielaPick, number> = { '1': 0, X: 0, '2': 0, O25: 0, BTTS: 0 }
  winiela.legs.forEach((l) => {
    counts[l.pick] = (counts[l.pick] || 0) + 1
  })
  const typeSummary = [
    counts['1'] ? `${counts['1']} ${t('win_locals')}` : '',
    counts['X'] ? `${counts['X']} ${t('win_draws')}` : '',
    counts['2'] ? `${counts['2']} ${t('win_visits')}` : '',
    counts['O25'] ? `${counts['O25']} ${t('win_overs')}` : '',
    counts['BTTS'] ? `${counts['BTTS']} ${t('win_btts_n')}` : '',
  ]
    .filter(Boolean)
    .join(' · ')

  const addToBankroll = () => {
    const res = loadParlayToSlip(
      winiela.legs.map((l) => ({
        homeName: l.match.home || l.match.homeShort,
        awayName: l.match.away || l.match.awayShort,
        league: l.match.league,
        matchDate: l.match.date,
        outcome: winielaOutcome(l.pick),
        oddsUser: l.odds || 2.0,
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
    <div className="mx-auto max-w-175 py-4">
      <div className="panel overflow-hidden">
        <div className="border-line flex flex-wrap items-baseline gap-3 border-b px-4 py-3.5">
          <span className="font-display text-red text-xl font-extrabold tracking-wider">🇪🇸 WINIELA</span>
          <span className="text-mute text-xs">
            {winiela.legs.length} {t('win_subtitle')}
          </span>
        </div>
        <div className="border-line text-mute border-b px-4 py-1.5 text-[10px] tracking-wide">{typeSummary}</div>

        <div className="px-4">
          {winiela.legs.map((l, i) => {
            const d = l.match.date.split('-')
            const color = TYPE_COLOR[l.pick]
            const isWide = l.pick === 'O25' || l.pick === 'BTTS'
            return (
              <div key={i} className="border-line flex items-center gap-3 border-b py-2.5 last:border-b-0">
                <div className="text-mute w-5 shrink-0 text-center text-xs">{i + 1}</div>
                <div
                  className={`stat-num flex shrink-0 items-center justify-center rounded-md border px-1 py-1 ${
                    isWide ? 'min-w-11 text-[11px]' : 'min-w-9 text-base'
                  }`}
                  style={{ color, borderColor: color + '55', background: color + '15' }}
                >
                  {l.pick}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-semibold">
                    {l.match.homeShort} <span className="text-mute">vs</span> {l.match.awayShort}
                  </div>
                  <div className="text-mute mt-0.5 text-[11px]">
                    <span className="font-semibold" style={{ color }}>
                      {typeLabel(l.pick)}
                    </span>{' '}
                    · {l.prob.toFixed(1)}%{l.match.time ? ' · ' + l.match.time + 'h' : ''} · {d[2]}/{d[1]}
                  </div>
                </div>
                {l.odds && <div className="stat-num text-gold shrink-0 text-[15px]">@{l.odds}</div>}
              </div>
            )
          })}
        </div>

        <div className="bg-raised/50 border-line flex flex-wrap items-center gap-x-6 gap-y-2 border-t px-4 py-3.5">
          <div>
            <div className="micro-label">{t('lbl_comb_prob')}</div>
            <div className="stat-num text-red text-xl">{winiela.probDisplay}</div>
          </div>
          {winiela.combinedOdds && (
            <div>
              <div className="micro-label">{t('lbl_comb_odds')}</div>
              <div className="stat-num text-gold text-xl">@{winiela.combinedOdds}</div>
            </div>
          )}
          <button
            onClick={addToBankroll}
            className="bg-teal font-display ml-auto cursor-pointer rounded-lg px-4 py-2 text-sm font-bold text-black"
          >
            {t('btn_add_bkrl')}
          </button>
          <div className="text-mute/70 w-full text-right text-[10px]">{t('win_fun')}</div>
        </div>
      </div>
    </div>
  )
}

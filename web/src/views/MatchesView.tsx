// ─────────────────────────────────────────────────────────────
// Date / ALL view — league chips, sort select, day summary,
// match-card grid (monolith render() main branch).
// ─────────────────────────────────────────────────────────────
import { useMemo } from 'react'
import { t } from '../i18n'
import { bestWinProb, sortMatches, winnerTeam } from '../lib/betting'
import { leagueColor } from '../lib/format'
import { useApp } from '../state/AppContext'
import MatchCard from '../components/MatchCard'
import { EmptyState } from '../components/ui'
import type { SortMode } from '../types'

const LEAGUES = ['ALL', 'PL', 'PD', 'BL1', 'FL1']

export default function MatchesView() {
  const { data, active, league, setLeague, sort, setSort } = useApp()

  const dayMatches = useMemo(
    () => (active === 'ALL' ? data.matches : data.matches.filter((m) => m.date === active)),
    [data.matches, active],
  )
  const filtered = useMemo(
    () => (league === 'ALL' ? dayMatches : dayMatches.filter((m) => m.league === league)),
    [dayMatches, league],
  )
  const sorted = useMemo(() => sortMatches(filtered, sort), [filtered, sort])

  const highConf = dayMatches.filter((m) => m.stars === 5).length
  const medConf = dayMatches.filter((m) => m.stars === 3).length
  const topPick = [...dayMatches]
    .filter((m) => m.stars === 5)
    .sort((a, b) => bestWinProb(b) - bestWinProb(a))[0]

  return (
    <div>
      {/* controls */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="flex flex-wrap gap-1.5">
          {LEAGUES.map((lg) => {
            const isActive = league === lg
            const c = lg === 'ALL' ? 'var(--color-gold)' : leagueColor(lg)
            return (
              <button
                key={lg}
                onClick={() => setLeague(lg)}
                className="font-display cursor-pointer rounded-full border px-3 py-1 text-xs font-bold tracking-wide uppercase transition-colors"
                style={
                  isActive
                    ? { background: c + '22', color: c, borderColor: c + '88' }
                    : { borderColor: 'var(--color-line)', color: 'var(--color-mute)' }
                }
              >
                {lg === 'ALL' ? t('all_leagues') : lg}
              </button>
            )
          })}
        </div>
        <label className="text-mute ml-auto flex items-center gap-2 text-xs">
          {t('lbl_sort')}
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortMode)}
            className="bg-raised border-line text-ink cursor-pointer rounded-md border px-2 py-1.5 text-xs outline-none"
          >
            <option value="stars">{t('sort_stars')}</option>
            <option value="prob">{t('sort_prob')}</option>
            <option value="time">{t('sort_time')}</option>
          </select>
        </label>
      </div>

      {/* summary */}
      <div className="panel mb-4 flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-3.5">
        <div>
          <div className="stat-num text-2xl">{dayMatches.length}</div>
          <div className="micro-label">{t('sum_matches')}</div>
        </div>
        <div className="bg-line h-8 w-px" />
        <div>
          <div className="stat-num text-green text-2xl">{highConf}</div>
          <div className="micro-label">{t('sum_high')}</div>
        </div>
        <div className="bg-line h-8 w-px" />
        <div>
          <div className="stat-num text-amber text-2xl">{medConf}</div>
          <div className="micro-label">{t('sum_med')}</div>
        </div>
        {topPick && (
          <div className="border-gold/30 bg-gold/6 ml-auto rounded-lg border px-4 py-2">
            <div className="micro-label text-gold">{t('sum_best_pick')}</div>
            <div className="font-display mt-0.5 text-base font-bold">
              {topPick.homeShort} vs {topPick.awayShort}
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="stat-num text-gold text-lg">{bestWinProb(topPick).toFixed(1)}%</span>
              <span className="text-mute text-[11px]">
                {t('sum_victory')}{' '}
                {winnerTeam(topPick) === 'home' ? topPick.homeShort : topPick.awayShort}
              </span>
              {topPick.fairOdds && <span className="text-gold stat-num text-[12px]">@{topPick.fairOdds}</span>}
            </div>
          </div>
        )}
      </div>

      {/* grid */}
      {sorted.length === 0 ? (
        <EmptyState icon="⚽" msg={t('empty_matches')} />
      ) : (
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2 2xl:grid-cols-3">
          {sorted.map((m) => (
            <MatchCard key={`${m.date}|${m.home}|${m.away}`} m={m} />
          ))}
        </div>
      )}
    </div>
  )
}

// 6×6 score-distribution heatmap (Dixon-Coles prob_matrix)
import { t } from '../../i18n'
import type { MatchPred } from '../../types'

export default function ScoreGrid({ m }: { m: MatchPred }) {
  if (!m.scoreGrid || !m.scoreGrid.length) return null
  const grid = m.scoreGrid
  const maxP = Math.max(...grid.flat())
  const awayGoals = grid[0].length

  return (
    <div>
      <div className="text-mute mb-2 text-[11px]">{t('modal_grid_sub')}</div>
      <div className="thin-scroll overflow-x-auto">
        <table className="w-full border-separate" style={{ borderSpacing: 2 }}>
          <thead>
            <tr>
              <th />
              {Array.from({ length: awayGoals }, (_, i) => (
                <th key={i} className="stat-num text-[12px]" style={{ color: 'var(--color-away)' }}>
                  {i}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.map((row, h) => (
              <tr key={h}>
                <th className="stat-num text-[12px]" style={{ color: 'var(--color-home)' }}>
                  {h}
                </th>
                {row.map((p, a) => {
                  const intensity = maxP > 0 ? p / maxP : 0
                  const isBest = m.bestScore === `${h}-${a}`
                  return (
                    <td
                      key={a}
                      className="rounded px-1.5 py-1 text-center text-[11px]"
                      style={{
                        background: `rgba(20,184,166,${(intensity * 0.85).toFixed(2)})`,
                        border: isBest ? '2px solid var(--color-gold)' : '1px solid transparent',
                        color: intensity > 0.5 ? '#fff' : 'var(--color-ink)',
                        fontWeight: isBest ? 700 : 400,
                      }}
                    >
                      {p.toFixed(1)}%
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

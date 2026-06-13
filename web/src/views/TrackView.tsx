// ─────────────────────────────────────────────────────────────
// TRACK view — live-pick performance dashboard
// (monolith renderTrackingView).
// ─────────────────────────────────────────────────────────────
import { useMemo, useState } from 'react'
import { t, tf, tagLabel } from '../i18n'
import { useApp } from '../state/AppContext'
import { BankrollChart, ReliabilityDiagram } from '../components/charts'
import { EmptyState, LeagueBadge, accColor, roiColor } from '../components/ui'
import type { SubmodelAcc, TimingWindow, TrackerPick } from '../types'

type TrackFilter = 'all' | 'resolved' | 'pending'

const fmtAcc = (v: number | null | undefined) => (v !== null && v !== undefined ? (v * 100).toFixed(1) + '%' : '—')
const fmtRoi = (v: number | null | undefined) =>
  v !== null && v !== undefined ? (v * 100 > 0 ? '+' : '') + (v * 100).toFixed(1) + '%' : '—'
const pct = (v: number | null | undefined) => (v !== null && v !== undefined ? v * 100 : null)

function MetricCard({
  value,
  label,
  sub,
  color,
  alert,
}: {
  value: string | number
  label: string
  sub?: string
  color?: string
  alert?: boolean
}) {
  return (
    <div
      className="panel px-4 py-3.5 text-center"
      style={alert ? { borderColor: 'rgba(239,68,68,0.4)', background: 'rgba(239,68,68,0.05)' } : undefined}
    >
      <div className="stat-num text-3xl" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="font-display mt-1 text-[13px] font-bold tracking-wide uppercase">{label}</div>
      {sub && (
        <div className="text-mute mt-0.5 text-[10px]" style={alert ? { color } : undefined}>
          {sub}
        </div>
      )}
    </div>
  )
}

function ChartPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="panel mt-4 px-4 py-4">
      <div className="micro-label mb-3 font-bold tracking-[0.1em]">{title}</div>
      {children}
    </div>
  )
}

function Bar({ pctVal, color }: { pctVal: number; color: string }) {
  return (
    <div className="bg-raised h-2 w-24 overflow-hidden rounded">
      <div className="h-full rounded transition-[width]" style={{ width: `${Math.min(100, Math.max(0, pctVal))}%`, background: color }} />
    </div>
  )
}

const TH = ({ children }: { children: React.ReactNode }) => (
  <th className="micro-label py-1.5 pr-3 text-left font-bold">{children}</th>
)
const TD = ({ children, className = '', style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) => (
  <td className={`border-line/40 border-t py-1.5 pr-3 text-[13px] ${className}`} style={style}>
    {children}
  </td>
)

const OUT_LABELS: Record<string, () => string> = {
  home: () => t('out_home'),
  draw: () => t('out_draw'),
  away: () => t('out_away'),
}

function PicksTable({ picks }: { picks: TrackerPick[] }) {
  const [filter, setFilter] = useState<TrackFilter>('all')
  const filtered = picks.filter((p) => {
    if (filter === 'resolved') return p.actualResult !== null
    if (filter === 'pending') return p.actualResult === null
    return true
  })

  return (
    <div className="panel mt-4 overflow-hidden">
      <div className="border-line flex flex-wrap items-center gap-3 border-b px-4 py-3">
        <span className="font-display text-base font-bold tracking-wide">
          {t('trk_picks_history')} ({filtered.length})
        </span>
        <div className="ml-auto flex gap-1.5">
          {(['all', 'resolved', 'pending'] as TrackFilter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`cursor-pointer rounded-md border px-2.5 py-1 text-[11px] font-semibold transition-colors ${
                filter === f ? 'border-teal/60 bg-teal/10 text-teal' : 'border-line text-mute hover:text-ink'
              }`}
            >
              {f === 'all' ? t('trk_filter_all') : f === 'resolved' ? t('trk_filter_resolved') : t('trk_filter_pending')}
            </button>
          ))}
        </div>
      </div>
      <div className="thin-scroll overflow-x-auto px-4 pb-3">
        <table className="w-full min-w-180">
          <thead>
            <tr>
              <TH>{t('th_date')}</TH>
              <TH>{t('th_match')}</TH>
              <TH>{t('lbl_league')}</TH>
              <TH>★</TH>
              <TH>{t('th_pred')}</TH>
              <TH>{t('th_prob')}</TH>
              <TH>{t('th_odds')}</TH>
              <TH>{t('th_real')}</TH>
              <TH>{t('lbl_result')}</TH>
              <TH>{t('th_profit')}</TH>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={10} className="text-mute py-6 text-center text-[13px]">
                  {t('trk_no_picks_filter')}
                </td>
              </tr>
            ) : (
              filtered.map((p) => {
                const isResolved = p.actualResult !== null
                const isWin = isResolved && p.actualResult === p.bestOutcome
                const d = p.date.split('-')
                return (
                  <tr key={p.matchId} style={{ opacity: isResolved ? 1 : 0.65 }}>
                    <TD>
                      {d[2]}/{d[1]}
                    </TD>
                    <TD>
                      <span className="font-semibold">{p.home}</span> vs {p.away}
                    </TD>
                    <TD>
                      <LeagueBadge league={p.league} />
                    </TD>
                    <TD>
                      <span className="text-amber">{'★'.repeat(p.stars)}</span>
                    </TD>
                    <TD>
                      <span className="text-teal">{OUT_LABELS[p.bestOutcome]?.() || p.bestOutcome}</span>
                    </TD>
                    <TD>
                      <span className="stat-num text-[15px]">{p.bestProb}%</span>
                    </TD>
                    <TD>{p.fairOdds ? '@' + p.fairOdds : '—'}</TD>
                    <TD>{p.actualResult ? OUT_LABELS[p.actualResult]?.() || p.actualResult : '—'}</TD>
                    <TD>
                      {isResolved ? (
                        <span
                          className="rounded px-1.5 py-px text-[11px] font-bold"
                          style={{
                            color: isWin ? 'var(--color-green)' : 'var(--color-red)',
                            background: isWin ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.1)',
                          }}
                        >
                          {isWin ? t('pick_win') : t('pick_loss')}
                        </span>
                      ) : (
                        <span className="text-mute text-[11px]">{t('pick_pending')}</span>
                      )}
                    </TD>
                    <TD>
                      {isResolved ? (
                        isWin ? (
                          <span className="text-green">+{p.fairOdds ? (p.fairOdds - 1).toFixed(2) : '?'}</span>
                        ) : (
                          <span className="text-red">-1.00</span>
                        )
                      ) : (
                        <span className="text-mute">—</span>
                      )}
                    </TD>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function TrackView() {
  const { data } = useApp()
  const m = data.trackerMetrics
  const picks = data.trackerPicks

  const resolvedWithSub = useMemo(
    () => picks.filter((p) => p.actualResult !== null && p.subPreds && Object.keys(p.subPreds).length > 0),
    [picks],
  )

  if (!m || picks.length === 0) {
    return <EmptyState icon="📊" title={t('trk_empty_title')} msg={t('trk_empty_msg')} code="python main.py" />
  }

  const psi = m.psi || {}
  const psiVal = psi.psi_max
  const psiDrift = !!psi.drift_detected
  const psiColor = psiDrift
    ? 'var(--color-red)'
    : (psiVal ?? 0) > 0.1
      ? 'var(--color-amber)'
      : 'var(--color-green)'
  const psiLabel = psiDrift ? t('trk_psi_drift') : (psiVal ?? 0) > 0.1 ? t('trk_psi_mod') : t('trk_psi_stable')

  // sub-model accuracy from raw picks
  const allModels = [...new Set(resolvedWithSub.flatMap((p) => Object.keys(p.subPreds!)))]
  const modelName = (k: string) =>
    k === 'dc' ? 'Dixon-Coles' : k === 'elo' ? 'Elo' : k === 'form' ? t('model_form') : k === 'h2h' ? 'H2H' : k

  const sm30 = m.submodel_accuracy_30 || {}
  const sm30Keys = Object.keys(sm30).filter((k) => k !== 'n_window')
  const timing = m.timing_analysis || {}
  const timingLeagues = Object.keys(timing)
  const WINDOWS = ['0-6h', '6-24h', '24-48h', '48h+']

  return (
    <div className="mx-auto max-w-260">
      {/* metric cards */}
      <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4 xl:grid-cols-5">
        <MetricCard
          value={fmtAcc(m.accuracy_1x2)}
          label={t('trk_acc')}
          sub={t('trk_acc_sub')}
          color={m.accuracy_1x2 !== null && m.accuracy_1x2 >= 0.5 ? 'var(--color-green)' : 'var(--color-red)'}
        />
        <MetricCard
          value={m.brier_score !== null ? m.brier_score.toFixed(3) : '—'}
          label={t('trk_brier')}
          sub={t('trk_brier_sub')}
          color={
            m.brier_score !== null && m.brier_score < 0.5
              ? 'var(--color-green)'
              : m.brier_score !== null && m.brier_score < 0.6
                ? 'var(--color-gold)'
                : 'var(--color-red)'
          }
        />
        <MetricCard
          value={fmtRoi(m.roi_flat)}
          label={t('trk_roi')}
          sub={t('trk_roi_sub')}
          color={roiColor(pct(m.roi_flat))}
        />
        <MetricCard
          value={m.n_resolved}
          label={t('trk_resolved')}
          sub={`${m.n_pending} ${t('trk_pending_n')}`}
          color="var(--color-teal)"
        />
        {psiVal !== null && psiVal !== undefined && (
          <MetricCard value={psiVal.toFixed(3)} label={t('trk_psi')} sub={psiLabel} color={psiColor} alert={psiDrift} />
        )}
      </div>

      {/* value bets live record */}
      {!!m.vb_n && m.vb_n > 0 && (
        <div className="bg-green/5 border-green/20 mt-3 rounded-xl border px-4 py-3">
          <div className="text-green mb-2.5 text-[11px] font-bold tracking-wider uppercase">{t('trk_vb_hist')}</div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <div
                className="stat-num text-2xl"
                style={{ color: (m.vb_accuracy ?? 0) >= 0.5 ? 'var(--color-green)' : 'var(--color-red)' }}
              >
                {fmtAcc(m.vb_accuracy)}
              </div>
              <div className="micro-label mt-0.5">{t('trk_vb_acc')}</div>
            </div>
            <div>
              <div className="stat-num text-2xl" style={{ color: roiColor(pct(m.vb_roi_flat)) }}>
                {fmtRoi(m.vb_roi_flat)}
              </div>
              <div className="micro-label mt-0.5">{t('trk_vb_roi')}</div>
            </div>
            <div>
              <div className="stat-num text-teal text-2xl">{m.vb_n}</div>
              <div className="micro-label mt-0.5">{t('trk_vb_n')}</div>
            </div>
          </div>
        </div>
      )}

      {/* per-league */}
      {m.per_league && Object.keys(m.per_league).length > 0 && (
        <ChartPanel title={t('trk_by_league')}>
          <div className="thin-scroll overflow-x-auto">
            <table className="w-full min-w-140">
              <thead>
                <tr>
                  <TH>{t('lbl_league')}</TH>
                  <TH>{t('th_picks')}</TH>
                  <TH>{t('th_accuracy')}</TH>
                  <TH>ROI</TH>
                  <TH>Brier</TH>
                  <TH>O2.5%</TH>
                  <TH>BTTS%</TH>
                  <TH>{t('th_edge_real')}</TH>
                </tr>
              </thead>
              <tbody>
                {Object.entries(m.per_league).map(([lg, d]) => {
                  const he = m.hindsight_edge_by_league?.[lg]
                  return (
                    <tr key={lg}>
                      <TD>
                        <LeagueBadge league={lg} />
                      </TD>
                      <TD className="text-mute">{d.n}</TD>
                      <TD>
                        <span className="stat-num text-[15px]" style={{ color: accColor(pct(d.accuracy)) }}>
                          {fmtAcc(d.accuracy)}
                        </span>
                      </TD>
                      <TD>
                        <span className="stat-num text-[15px]" style={{ color: roiColor(pct(d.roi)) }}>
                          {fmtRoi(d.roi)}
                        </span>
                      </TD>
                      <TD className="text-mute">{d.brier !== null ? d.brier.toFixed(3) : '—'}</TD>
                      <TD className="text-mute">{fmtAcc(d.accuracy_over25)}</TD>
                      <TD className="text-mute">{fmtAcc(d.accuracy_btts)}</TD>
                      <TD style={{ color: roiColor(pct(he ?? null)) }}>{fmtRoi(he)}</TD>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </ChartPanel>
      )}

      {/* per-stars */}
      {m.per_stars && Object.keys(m.per_stars).length > 0 && (
        <ChartPanel title={t('trk_by_stars')}>
          <div className="thin-scroll overflow-x-auto">
            <table className="w-full min-w-130">
              <thead>
                <tr>
                  <TH>{t('th_tier')}</TH>
                  <TH>{t('th_picks')}</TH>
                  <TH>{t('th_accuracy')}</TH>
                  <TH>ROI</TH>
                  <TH>{t('th_edge_real')}</TH>
                  <TH>{t('th_bar')}</TH>
                </tr>
              </thead>
              <tbody>
                {Object.entries(m.per_stars)
                  .sort((a, b) => Number(b[0]) - Number(a[0]))
                  .map(([s, d]) => {
                    const he = m.hindsight_edge_by_stars?.[s]
                    const accNum = pct(d.accuracy)
                    return (
                      <tr key={s}>
                        <TD>
                          <span className="font-display text-amber text-[15px]">
                            {'★'.repeat(Number(s))}
                            {'☆'.repeat(5 - Number(s))}
                          </span>
                        </TD>
                        <TD className="text-mute">{d.n}</TD>
                        <TD>
                          <span className="stat-num text-[15px]" style={{ color: accColor(accNum) }}>
                            {fmtAcc(d.accuracy)}
                          </span>
                        </TD>
                        <TD>
                          <span className="stat-num text-[15px]" style={{ color: roiColor(pct(d.roi)) }}>
                            {fmtRoi(d.roi)}
                          </span>
                        </TD>
                        <TD style={{ color: roiColor(pct(he ?? null)) }}>{fmtRoi(he)}</TD>
                        <TD>
                          <Bar pctVal={accNum ?? 0} color={accColor(accNum)} />
                        </TD>
                      </tr>
                    )
                  })}
                {[
                  { min: 4, label: t('trk_combined4') },
                  { min: 5, label: t('trk_combined5') },
                ].map(({ min, label }) => {
                  const tiers = Object.entries(m.per_stars!).filter(([s]) => Number(s) >= min)
                  if (!tiers.length) return null
                  const totalN = tiers.reduce((s, [, d]) => s + d.n, 0)
                  if (!totalN) return null
                  const wAcc = tiers.reduce((s, [, d]) => s + (d.accuracy ?? 0) * d.n, 0) / totalN
                  const wRoi = tiers.reduce((s, [, d]) => s + (d.roi ?? 0) * d.n, 0) / totalN
                  const accC = wAcc >= 0.65 ? 'var(--color-green)' : wAcc >= 0.55 ? 'var(--color-amber)' : 'var(--color-red)'
                  return (
                    <tr key={min} className="bg-raised/50">
                      <TD>
                        <span className="font-display text-teal text-[14px] font-bold">{label}</span>
                      </TD>
                      <TD className="text-mute">{totalN}</TD>
                      <TD>
                        <span className="stat-num text-[15px]" style={{ color: accC }}>
                          {(wAcc * 100).toFixed(1)}%
                        </span>
                      </TD>
                      <TD>
                        <span className="stat-num text-[15px]" style={{ color: roiColor(wRoi * 100) }}>
                          {wRoi > 0 ? '+' : ''}
                          {(wRoi * 100).toFixed(1)}%
                        </span>
                      </TD>
                      <TD className="text-mute text-[11px]">{t('trk_combined')}</TD>
                      <TD>
                        <Bar pctVal={wAcc * 100} color={accC} />
                      </TD>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </ChartPanel>
      )}

      {/* per-tag */}
      {m.per_tag && Object.keys(m.per_tag).length > 0 && (
        <ChartPanel title={t('trk_by_tag')}>
          <div className="thin-scroll overflow-x-auto">
            <table className="w-full min-w-110">
              <thead>
                <tr>
                  <TH>{t('th_tag')}</TH>
                  <TH>{t('th_picks')}</TH>
                  <TH>{t('th_accuracy')}</TH>
                  <TH>ROI</TH>
                </tr>
              </thead>
              <tbody>
                {Object.entries(m.per_tag)
                  .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
                  .map(([tag, d]) => (
                    <tr key={tag}>
                      <TD className="text-[12px]">{tagLabel(tag)}</TD>
                      <TD className="text-mute">{d.n}</TD>
                      <TD>
                        <span className="stat-num text-[15px]" style={{ color: accColor(pct(d.accuracy)) }}>
                          {fmtAcc(d.accuracy)}
                        </span>
                      </TD>
                      <TD>
                        <span className="stat-num text-[15px]" style={{ color: roiColor(pct(d.roi)) }}>
                          {fmtRoi(d.roi)}
                        </span>
                      </TD>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </ChartPanel>
      )}

      {/* bankroll curve */}
      <ChartPanel title={t('trk_curve_title')}>
        <BankrollChart history={m.bankroll_history} />
      </ChartPanel>

      {/* reliability */}
      <ChartPanel title={t('trk_reliability')}>
        <ReliabilityDiagram picks={picks} />
      </ChartPanel>

      {/* sub-model accuracy (all resolved with subPreds) */}
      {resolvedWithSub.length >= 5 && (
        <ChartPanel title={tf('trk_submodel', resolvedWithSub.length)}>
          <table className="w-full max-w-130">
            <thead>
              <tr>
                <TH>{t('th_model')}</TH>
                <TH>{t('th_n_picks')}</TH>
                <TH>{t('th_accuracy')}</TH>
                <TH>{t('th_bar')}</TH>
              </tr>
            </thead>
            <tbody>
              {allModels.map((key) => {
                const relevant = resolvedWithSub.filter((p) => p.subPreds![key])
                if (!relevant.length) return null
                const correct = relevant.filter((p) => {
                  const sp = p.subPreds![key]
                  const best =
                    sp.ph >= sp.pd && sp.ph >= sp.pa ? 'home' : sp.pa >= sp.ph && sp.pa >= sp.pd ? 'away' : 'draw'
                  return best === p.actualResult
                }).length
                const acc = (correct / relevant.length) * 100
                const barColor = acc >= 50 ? 'var(--color-green)' : acc >= 40 ? 'var(--color-amber)' : 'var(--color-red)'
                return (
                  <tr key={key}>
                    <TD className="font-semibold">{modelName(key)}</TD>
                    <TD className="text-mute">{relevant.length}</TD>
                    <TD>
                      <span className="stat-num text-[16px]" style={{ color: barColor }}>
                        {acc.toFixed(1)}%
                      </span>
                    </TD>
                    <TD>
                      <Bar pctVal={acc} color={barColor} />
                    </TD>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </ChartPanel>
      )}

      {/* sub-model accuracy — last 30 window */}
      {sm30Keys.length > 0 && (
        <ChartPanel title={tf('trk_submodel30', (sm30.n_window as number) || 30)}>
          <table className="w-full max-w-140">
            <thead>
              <tr>
                <TH>{t('th_model')}</TH>
                <TH>N</TH>
                <TH>{t('th_accuracy')}</TH>
                <TH>Brier</TH>
                <TH>{t('th_bar')}</TH>
              </tr>
            </thead>
            <tbody>
              {sm30Keys.map((key) => {
                const v = sm30[key] as SubmodelAcc
                if (!v || typeof v !== 'object') return null
                const acc = v.accuracy * 100
                const barColor = acc >= 50 ? 'var(--color-green)' : 'var(--color-red)'
                return (
                  <tr key={key}>
                    <TD className="font-semibold">{modelName(key)}</TD>
                    <TD className="text-mute">{v.n}</TD>
                    <TD>
                      <span className="font-bold" style={{ color: barColor }}>
                        {acc.toFixed(1)}%
                      </span>
                    </TD>
                    <TD className="text-mute">{v.brier !== undefined ? v.brier.toFixed(3) : '—'}</TD>
                    <TD>
                      <Bar pctVal={acc} color={barColor} />
                    </TD>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </ChartPanel>
      )}

      {/* CLV timing */}
      {timingLeagues.length > 0 && (
        <ChartPanel title={t('trk_timing')}>
          <div className="thin-scroll overflow-x-auto">
            <table className="w-full min-w-130">
              <thead>
                <tr>
                  <TH>{t('lbl_league')}</TH>
                  {WINDOWS.map((w) => (
                    <TH key={w}>{w}</TH>
                  ))}
                  <TH>{t('trk_best_window')}</TH>
                </tr>
              </thead>
              <tbody>
                {timingLeagues.map((lg) => {
                  const ld = timing[lg]
                  const best = (ld.window_best_clv as string) || '—'
                  return (
                    <tr key={lg}>
                      <TD className="font-bold">{lg}</TD>
                      {WINDOWS.map((w) => {
                        const wd = ld[w] as TimingWindow | undefined
                        if (!wd || typeof wd === 'string') return <TD key={w} className="text-mute">—</TD>
                        const clvStr = (wd.avg_clv >= 0 ? '+' : '') + (wd.avg_clv * 100).toFixed(1)
                        return (
                          <TD
                            key={w}
                            style={{
                              color: wd.avg_clv > 0 ? 'var(--color-green)' : 'var(--color-red)',
                              fontWeight: w === best ? 800 : 400,
                            }}
                          >
                            {clvStr}% (n={wd.n})
                          </TD>
                        )
                      })}
                      <TD>
                        <span className="text-green font-bold">{best}</span>
                      </TD>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </ChartPanel>
      )}

      <PicksTable picks={picks} />
    </div>
  )
}

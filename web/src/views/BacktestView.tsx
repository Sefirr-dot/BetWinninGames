// ─────────────────────────────────────────────────────────────
// BACK view — backtest report (monolith renderBacktestView).
// ─────────────────────────────────────────────────────────────
import { t } from '../i18n'
import { useApp } from '../state/AppContext'
import { AccuracyCurve, BankrollChart } from '../components/charts'
import { EmptyState } from '../components/ui'

function MetricCard({ value, label, sub, color }: { value: string; label: string; sub: string; color?: string }) {
  return (
    <div className="panel px-4 py-3.5 text-center">
      <div className="stat-num text-3xl" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="font-display mt-1 text-[13px] font-bold tracking-wide uppercase">{label}</div>
      <div className="text-mute mt-0.5 text-[10px]">{sub}</div>
    </div>
  )
}

export default function BacktestView() {
  const { data } = useApp()
  const bd = data.backtest

  if (!bd) {
    return (
      <EmptyState
        icon="📈"
        title={t('bt_empty_title')}
        msg={t('bt_empty_msg')}
        code="python backtest.py --league PL --seasons 2024 2025"
      />
    )
  }

  const m = bd.metrics
  const acc = m.accuracy_1x2 !== undefined ? (m.accuracy_1x2 * 100).toFixed(1) + '%' : '—'
  const brier = m.brier_score !== undefined ? m.brier_score.toFixed(4) : '—'
  const roi =
    m.roi_flat !== undefined ? (m.roi_flat * 100 > 0 ? '+' : '') + (m.roi_flat * 100).toFixed(1) + '%' : '—'
  const accO = m.accuracy_over25 !== undefined ? (m.accuracy_over25 * 100).toFixed(1) + '%' : '—'

  const accColor = (m.accuracy_1x2 ?? 0) >= 0.5 ? 'var(--color-green)' : 'var(--color-red)'
  const roiColor =
    (m.roi_flat ?? 0) > 0 ? 'var(--color-green)' : (m.roi_flat ?? 0) < 0 ? 'var(--color-red)' : undefined
  const brierColor =
    (m.brier_score ?? 1) < 0.5 ? 'var(--color-green)' : (m.brier_score ?? 1) < 0.6 ? 'var(--color-gold)' : 'var(--color-red)'

  return (
    <div className="mx-auto max-w-260">
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <span className="bg-gold/10 border-gold/30 text-gold rounded-md border px-3 py-1 text-xs">
          {t('bt_league')}: {bd.league}
        </span>
        <span className="bg-raised border-line rounded-md border px-3 py-1 text-xs">
          {t('bt_seasons')}: {bd.seasons.join(', ')}
        </span>
        <span className="bg-raised border-line rounded-md border px-3 py-1 text-xs">
          {m.n_matches} {t('bt_matches_eval')}
        </span>
        <span className="bg-raised border-line text-mute rounded-md border px-3 py-1 text-xs">
          {t('bt_generated')}: {bd.generatedAt}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
        <MetricCard value={acc} label={t('trk_acc')} sub={t('trk_acc_sub')} color={accColor} />
        <MetricCard value={brier} label={t('trk_brier')} sub={t('trk_brier_sub')} color={brierColor} />
        <MetricCard value={roi} label={t('trk_roi')} sub={t('bt_roi_sub')} color={roiColor} />
        <MetricCard value={accO} label={t('bt_acc_o25')} sub={t('bt_acc_o25_sub')} color="var(--color-teal)" />
      </div>

      <div className="panel mt-4 px-4 py-4">
        <div className="micro-label mb-3 font-bold tracking-[0.1em]">{t('bt_curve')}</div>
        <BankrollChart history={bd.curve} />
      </div>

      {bd.curve && bd.curve.length >= 2 && (
        <div className="panel mt-4 px-4 py-4">
          <div className="micro-label mb-3 font-bold tracking-[0.1em]">{t('bt_acc_curve')}</div>
          <AccuracyCurve curve={bd.curve} />
        </div>
      )}
    </div>
  )
}

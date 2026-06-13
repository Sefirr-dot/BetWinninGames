// ─────────────────────────────────────────────────────────────
// SVG charts — 1:1 ports of the monolith's renderBankrollSVG,
// renderReliabilityDiagram and the backtest accuracy curve.
// ─────────────────────────────────────────────────────────────
import { t, tf } from '../i18n'
import type { BacktestCurvePoint, BankrollPoint, TrackerPick } from '../types'

// ── Bankroll curve (monolith renderBankrollSVG) ──────────────

export function BankrollChart({ history }: { history: BankrollPoint[] | null | undefined }) {
  if (!history || history.length < 2) {
    return <div className="text-mute px-4 py-8 text-center text-[13px]">{t('trk_curve_min')}</div>
  }
  const W = 700
  const H = 180
  const PAD = { top: 16, right: 16, bottom: 28, left: 48 }
  const iW = W - PAD.left - PAD.right
  const iH = H - PAD.top - PAD.bottom

  const vals = history.map((h) => h.bankroll)
  const minV = Math.min(...vals, 0.7)
  const maxV = Math.max(...vals, 1.3)
  const rng = maxV - minV || 1

  const xS = (i: number) => PAD.left + (i / (history.length - 1)) * iW
  const yS = (v: number) => PAD.top + iH - ((v - minV) / rng) * iH

  const pts = history.map((h, i) => `${xS(i).toFixed(1)},${yS(h.bankroll).toFixed(1)}`)
  const baseY = yS(1.0)
  const finalVal = vals[vals.length - 1]
  const lineColor = finalVal >= 1 ? 'var(--color-green)' : 'var(--color-red)'
  const fillColor = finalVal >= 1 ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)'

  const fillPath = `M ${xS(0).toFixed(1)},${baseY.toFixed(1)} L ${pts.join(' L ')} L ${xS(history.length - 1).toFixed(1)},${baseY.toFixed(1)} Z`

  const step = Math.max(1, Math.floor(history.length / 6))
  const labelIdx = history.map((_, i) => i).filter((i) => i % step === 0 || i === history.length - 1)
  const yRefs = [minV, 1.0, maxV]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full">
      {yRefs.map((v, k) => (
        <g key={k}>
          <line x1={PAD.left} y1={yS(v)} x2={W - PAD.right} y2={yS(v)} stroke="var(--color-line)" strokeWidth={0.5} />
          <text x={PAD.left - 4} y={yS(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-mute)">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      <line
        x1={PAD.left}
        y1={baseY}
        x2={W - PAD.right}
        y2={baseY}
        stroke="rgba(255,255,255,0.12)"
        strokeWidth={1}
        strokeDasharray="4,3"
      />
      <path d={fillPath} fill={fillColor} />
      <polyline points={pts.join(' ')} fill="none" stroke={lineColor} strokeWidth={2} />
      {history.map((h, i) => (
        <circle key={i} cx={xS(i)} cy={yS(h.bankroll)} r={3} fill={lineColor} />
      ))}
      {labelIdx.map((i) => {
        const d = history[i].date.split('-')
        return (
          <text key={i} x={xS(i)} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--color-mute)">
            {d[2]}/{d[1]}
          </text>
        )
      })}
    </svg>
  )
}

// ── Reliability diagram (monolith renderReliabilityDiagram) ──

export function ReliabilityDiagram({ picks }: { picks: TrackerPick[] }) {
  const resolved = picks.filter((p) => p.actualResult !== null)
  if (resolved.length < 20) {
    return (
      <div className="text-mute px-4 py-5 text-center text-[13px]">
        {tf('trk_reliability_min', resolved.length)}
      </div>
    )
  }
  const buckets = Array.from({ length: 10 }, (_, i) => ({
    min: i * 10,
    max: (i + 1) * 10,
    sumPred: 0,
    sumActual: 0,
    n: 0,
  }))
  resolved.forEach((p) => {
    const idx = Math.min(9, Math.floor(p.bestProb / 10))
    buckets[idx].sumPred += p.bestProb
    buckets[idx].sumActual += p.actualResult === p.bestOutcome ? 1 : 0
    buckets[idx].n++
  })

  const W = 480
  const H = 240
  const PAD = { top: 20, right: 20, bottom: 36, left: 44 }
  const iW = W - PAD.left - PAD.right
  const iH = H - PAD.top - PAD.bottom
  const xS = (pct: number) => PAD.left + (pct / 100) * iW
  const yS = (pct: number) => PAD.top + iH - (pct / 100) * iH
  const barW = iW / 10 - 2

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full max-w-130">
      {[0, 25, 50, 75, 100].map((v) => (
        <g key={v}>
          <text x={PAD.left - 4} y={yS(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-mute)">
            {v}%
          </text>
          <line x1={PAD.left} y1={yS(v)} x2={W - PAD.right} y2={yS(v)} stroke="var(--color-line)" strokeWidth={0.5} />
        </g>
      ))}
      <path
        d={`M ${xS(0)},${yS(0)} L ${xS(100)},${yS(100)}`}
        stroke="rgba(255,255,255,0.15)"
        strokeWidth={1}
        strokeDasharray="4,3"
        fill="none"
      />
      {buckets.map((b, i) => {
        if (!b.n) return null
        const actualRate = (b.sumActual / b.n) * 100
        const midX = xS(b.min + 5) - barW / 2
        const barH = (actualRate / 100) * iH
        const color =
          actualRate >= b.min && actualRate <= b.max + 10 ? 'var(--color-teal)' : 'var(--color-gold)'
        return (
          <g key={i}>
            <rect x={midX} y={PAD.top + iH - barH} width={barW} height={barH} fill={color} opacity={0.7} rx={2} />
            <text x={midX + barW / 2} y={H - 8} textAnchor="middle" fontSize={9} fill="var(--color-mute)">
              {b.min + 5}%
            </text>
          </g>
        )
      })}
      {buckets.map((b, i) => {
        if (!b.n) return null
        return (
          <circle
            key={i}
            cx={xS(b.min + 5)}
            cy={yS((b.sumActual / b.n) * 100)}
            r={4}
            fill="var(--color-gold)"
            stroke="var(--color-surface)"
            strokeWidth={1.5}
          />
        )
      })}
      <text x={W / 2} y={H} textAnchor="middle" fontSize={10} fill="var(--color-mute)">
        {t('trk_pred_prob')}
      </text>
      <text x={10} y={H / 2} textAnchor="middle" fontSize={10} fill="var(--color-mute)" transform={`rotate(-90,10,${H / 2})`}>
        {t('trk_real_rate')}
      </text>
    </svg>
  )
}

// ── Backtest cumulative-accuracy curve ───────────────────────

export function AccuracyCurve({ curve }: { curve: BacktestCurvePoint[] }) {
  if (!curve || curve.length < 2) return null
  const W = 700
  const H = 160
  const PAD = { top: 16, right: 16, bottom: 28, left: 44 }
  const iW = W - PAD.left - PAD.right
  const iH = H - PAD.top - PAD.bottom
  const xS = (i: number) => PAD.left + (i / (curve.length - 1)) * iW
  const yS = (v: number) => PAD.top + iH - (v / 100) * iH

  const pts = curve.map((c, i) => `${xS(i).toFixed(1)},${yS(c.acc).toFixed(1)}`).join(' ')
  const step = Math.max(1, Math.floor(curve.length / 6))
  const labelIdx = curve.map((_, i) => i).filter((i) => i % step === 0 || i === curve.length - 1)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full">
      <line x1={PAD.left} y1={yS(33)} x2={W - PAD.right} y2={yS(33)} stroke="rgba(239,68,68,0.35)" strokeWidth={1} strokeDasharray="4,3" />
      <text x={PAD.left - 4} y={yS(33) + 3} textAnchor="end" fontSize={9} fill="rgba(239,68,68,0.7)">
        33%
      </text>
      <line x1={PAD.left} y1={yS(50)} x2={W - PAD.right} y2={yS(50)} stroke="rgba(34,197,94,0.35)" strokeWidth={1} strokeDasharray="4,3" />
      <text x={PAD.left - 4} y={yS(50) + 3} textAnchor="end" fontSize={9} fill="rgba(34,197,94,0.7)">
        50%
      </text>
      <polyline points={pts} fill="none" stroke="var(--color-gold)" strokeWidth={2} />
      {labelIdx.map((i) => {
        const d = curve[i].date.split('-')
        return (
          <text key={i} x={xS(i)} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--color-mute)">
            {d[2]}/{d[1]}
          </text>
        )
      })}
    </svg>
  )
}

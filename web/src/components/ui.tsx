/* eslint-disable react-refresh/only-export-components */
import type { ReactNode } from 'react'
import { t } from '../i18n'
import { leagueColor, sharpnessMeta } from '../lib/format'

export function LeagueBadge({ league, className = '' }: { league: string; className?: string }) {
  const c = leagueColor(league)
  return (
    <span
      className={`inline-flex items-center rounded px-1.5 py-px font-display text-[11px] font-bold tracking-wide uppercase ${className}`}
      style={{ background: c + '22', color: c, border: `1px solid ${c}55` }}
    >
      {league}
    </span>
  )
}

const STAR_COLORS: Record<number, string> = {
  5: '#22c55e',
  4: '#a3e635',
  3: '#eab308',
  2: '#f97316',
  1: '#6b7e97',
}

export function Stars({ n, size = 13 }: { n: number; size?: number }) {
  const color = STAR_COLORS[n] || '#6b7e97'
  return (
    <span className="inline-flex gap-px leading-none" style={{ fontSize: size }}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} style={{ color: i <= n ? color : '#28354d' }}>
          ★
        </span>
      ))}
    </span>
  )
}

export function StarsInline({ n }: { n: number }) {
  return <span className="text-amber text-[11px]">{'★'.repeat(Math.max(0, n))}</span>
}

export function YesNo({ yes }: { yes: boolean }) {
  return (
    <span
      className="rounded px-1.5 py-px font-display text-[11px] font-bold"
      style={{
        background: yes ? 'rgba(34,197,94,0.14)' : 'rgba(239,68,68,0.12)',
        color: yes ? '#22c55e' : '#f87171',
      }}
    >
      {yes ? t('yn_yes') : t('yn_no')}
    </span>
  )
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <div className="micro-label mb-2.5 flex items-center gap-2 font-semibold">
      <span className="bg-teal/70 inline-block h-3 w-0.5 rounded-full" />
      {children}
    </div>
  )
}

export function PanelTitle({ children }: { children: ReactNode }) {
  return <div className="micro-label mb-3 font-bold tracking-[0.12em]">{children}</div>
}

/** Sharpness badge — same thresholds/colours as the monolith. */
export function SharpnessBadge({ score }: { score: number | null | undefined }) {
  const meta = sharpnessMeta(score)
  if (!meta) return null
  return (
    <span
      className="inline-flex items-center gap-1 rounded-md bg-black/30 px-2 py-0.5 text-[10px] font-bold"
      style={{ color: meta.color, border: `1px solid ${meta.color}44` }}
    >
      ⚡ {score} {meta.label}
    </span>
  )
}

export function EdgeBadge({ edge }: { edge: number }) {
  return (
    <span className="bg-green/10 border-green/30 text-green rounded-md border px-1.5 py-px text-[10px] font-bold">
      Edge +{edge}%
    </span>
  )
}

export function KellyBadge({ children }: { children: ReactNode }) {
  return (
    <span className="bg-teal/10 border-teal/30 text-teal rounded-md border px-1.5 py-px text-[10px] font-bold">
      {children}
    </span>
  )
}

export function EmptyState({
  icon,
  title,
  msg,
  code,
  children,
}: {
  icon: string
  title?: string
  msg?: ReactNode
  code?: string
  children?: ReactNode
}) {
  return (
    <div className="panel mx-auto my-10 max-w-130 px-8 py-14 text-center">
      <div className="text-4xl">{icon}</div>
      {title && <div className="font-display mt-4 text-2xl font-bold tracking-wide">{title}</div>}
      {msg && <div className="text-mute mt-2 text-sm leading-relaxed whitespace-pre-line">{msg}</div>}
      {code && (
        <code className="bg-raised border-line text-teal mt-5 inline-block rounded-md border px-4 py-2 text-xs">
          {code}
        </code>
      )}
      {children}
    </div>
  )
}

export function FormDots({ form }: { form: string | null }) {
  if (!form) return <span className="text-mute">—</span>
  return (
    <span className="inline-flex items-center gap-0.5">
      {[...form].map((c, i) => (
        <span
          key={i}
          className="inline-block h-2 w-2 rounded-full"
          style={{ background: c === '+' ? '#22c55e' : '#ef4444' }}
        />
      ))}
    </span>
  )
}

export function MiniBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="bg-raised h-2 w-full overflow-hidden rounded">
      <div
        className="h-full rounded transition-[width] duration-400"
        style={{ width: `${Math.min(100, Math.max(0, pct))}%`, background: color }}
      />
    </div>
  )
}

/** Accuracy → colour helper used across tables */
export function accColor(accPct: number | null): string {
  if (accPct === null) return 'var(--color-mute)'
  return accPct >= 60 ? 'var(--color-green)' : accPct >= 50 ? 'var(--color-amber)' : 'var(--color-red)'
}

export function roiColor(roiPct: number | null): string {
  if (roiPct === null) return 'var(--color-mute)'
  return roiPct > 0 ? 'var(--color-green)' : roiPct < 0 ? 'var(--color-red)' : 'var(--color-mute)'
}

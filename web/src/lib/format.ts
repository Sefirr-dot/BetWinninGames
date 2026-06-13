import { t } from '../i18n'
import type { Outcome } from '../types'

export function outcomeLabel(o: Outcome | string): string {
  const map: Record<string, string> = {
    home: t('out_home'),
    draw: t('out_draw'),
    away: t('out_away'),
    over25: t('out_over25'),
    btts: t('out_btts'),
  }
  return map[o] || o
}

/** Mirrors monolith _outcomeColor() */
export function outcomeColor(o: Outcome | string): string {
  const map: Record<string, string> = {
    home: 'var(--color-home)',
    draw: 'var(--color-draw)',
    away: 'var(--color-away)',
    over25: 'var(--color-teal)',
    btts: 'var(--color-gold)',
  }
  return map[o] || 'var(--color-mute)'
}

/** Same colour/label thresholds as the monolith _sharpnessBadge() */
export function sharpnessMeta(sc: number | null | undefined): { color: string; label: string } | null {
  if (!sc || sc <= 0) return null
  if (sc >= 80) return { color: '#22c55e', label: t('sharp_elite') }
  if (sc >= 65) return { color: '#3b82f6', label: t('sharp_strong') }
  if (sc >= 50) return { color: '#facc15', label: t('sharp_moderate') }
  if (sc >= 35) return { color: '#f97316', label: t('sharp_low') }
  return { color: '#6b7e97', label: t('sharp_min') }
}

export function fmtSignedPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return '—'
  const pct = v * 100
  return (pct > 0 ? '+' : '') + pct.toFixed(digits) + '%'
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return '—'
  return (v * 100).toFixed(digits) + '%'
}

export function fmtEur(v: number): string {
  return '€' + v.toFixed(2)
}

/** "2026-04-18" → "18/04" */
export function shortDate(d: string): string {
  const p = d.split('-')
  return p.length === 3 ? `${p[2]}/${p[1]}` : d
}

/** ISO timestamp → "2026-04-18 19:32" */
export function shortStamp(iso: string | undefined): string {
  return (iso || '').replace('T', ' ').slice(0, 16)
}

export function starsLabel(s: number): string {
  return t(`stars_${Math.min(5, Math.max(1, s))}`)
}

export const LEAGUE_COLORS: Record<string, string> = {
  PL: '#00d4aa',
  PD: '#ef4444',
  BL1: '#3b82f6',
  FL1: '#06b6d4',
}

export function leagueColor(lg: string): string {
  return LEAGUE_COLORS[lg] || '#475569'
}

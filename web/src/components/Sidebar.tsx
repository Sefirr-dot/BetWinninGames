import { useMemo } from 'react'
import { t } from '../i18n'
import { calcBestBets } from '../lib/betting'
import { useBankroll, useSlip } from '../lib/storage'
import { useApp } from '../state/AppContext'

interface Item {
  id: string
  icon: string
  label: string
  badge: string | number
  accent: string
}

export default function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { data, active, setActive } = useApp()
  const slip = useSlip()
  const brl = useBankroll()

  const items = useMemo<Item[]>(() => {
    const bestBets = calcBestBets(data.matches)
    const nVB = data.matches.filter((m) => m.valueBets && m.valueBets.length > 0).length
    const nBetNow =
      data.matches.filter((m) => m.valueBets && m.valueBets.length > 0 && m.stars >= 4).length +
      data.matches.filter((m) => m.valueBets && m.valueBets.length > 0 && m.stars === 3).length
    const nResolved = data.trackerPicks.filter((p) => p.actualResult !== null).length
    const nPending = data.trackerPicks.filter((p) => p.actualResult === null).length
    const nPD = data.matches.filter((m) => m.league === 'PD').length
    const slipBadge = slip.length > 0 ? slip.length : brl ? '€' + brl.current.toFixed(0) : '—'
    const trackBadge = nPending > 0 ? `${nResolved}/${nResolved + nPending}` : nResolved

    return [
      { id: 'AUTO', icon: '🤖', label: t('nav_auto'), badge: '✨', accent: 'var(--color-teal)' },
      { id: 'SLIP', icon: '🎯', label: t('nav_bankroll'), badge: slipBadge, accent: 'var(--color-teal)' },
      { id: 'BETNOW', icon: '💰', label: t('nav_betnow'), badge: nBetNow, accent: 'var(--color-green)' },
      { id: 'BEST', icon: '🏆', label: t('nav_top'), badge: bestBets.length, accent: 'var(--color-gold)' },
      { id: 'VALUE', icon: '💎', label: t('nav_value'), badge: nVB, accent: 'var(--color-green)' },
      { id: 'TRACK', icon: '📊', label: t('nav_track'), badge: trackBadge, accent: 'var(--color-teal)' },
      { id: 'BACK', icon: '📈', label: t('nav_back'), badge: data.backtest ? 'OK' : '—', accent: 'var(--color-gold)' },
      { id: 'WIN', icon: '🇪🇸', label: t('nav_win'), badge: nPD, accent: 'var(--color-red)' },
    ]
  }, [data, slip.length, brl])

  return (
    <nav className="flex h-full flex-col gap-0.5 px-3 py-4">
      <div className="font-display mb-4 px-2 text-[22px] font-extrabold tracking-wider uppercase">
        Bet<span className="text-gold">Win</span>G
      </div>
      <div className="micro-label mb-1.5 px-2 font-bold">{t('nav_views')}</div>
      {items.map((it) => {
        const isActive = active === it.id
        return (
          <button
            key={it.id}
            onClick={() => {
              setActive(it.id)
              onNavigate?.()
            }}
            className={`group flex w-full cursor-pointer items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left text-[13px] font-semibold transition-colors ${
              isActive
                ? 'bg-raised border-line text-ink'
                : 'text-mute hover:text-ink hover:bg-raised/60 border-transparent'
            }`}
            style={isActive ? { boxShadow: `inset 2px 0 0 ${it.accent}` } : undefined}
          >
            <span className="text-[15px]">{it.icon}</span>
            <span className="flex-1">{it.label}</span>
            <span
              className="stat-num rounded-md px-1.5 py-px text-[11px]"
              style={{
                color: isActive ? it.accent : 'var(--color-mute)',
                background: 'rgba(0,0,0,0.25)',
              }}
            >
              {it.badge}
            </span>
          </button>
        )
      })}
      <div className="bg-line my-3 h-px" />
      <div className="micro-label px-2 font-bold">{t('nav_dates')}</div>
      <div className="mt-1.5 flex flex-col gap-0.5">
        {data.dates.map((d) => {
          const isActive = active === d
          const n = data.matches.filter((m) => m.date === d).length
          return (
            <button
              key={d}
              onClick={() => {
                setActive(d)
                onNavigate?.()
              }}
              className={`flex w-full cursor-pointer items-center justify-between rounded-lg border px-2.5 py-1.5 text-left text-xs transition-colors ${
                isActive
                  ? 'bg-raised border-line text-gold'
                  : 'text-mute hover:text-ink border-transparent'
              }`}
            >
              <span className="font-semibold">{data.dateLabels[d] || d}</span>
              <span className="stat-num text-[11px]">{n}</span>
            </button>
          )
        })}
      </div>
      <div className="text-mute/60 mt-auto px-2 pt-6 text-[10px] leading-relaxed">
        DC · Elo · ML · Pinnacle CLV
        {data.meta.generatedAt && <div>gen {data.meta.generatedAt}</div>}
      </div>
    </nav>
  )
}

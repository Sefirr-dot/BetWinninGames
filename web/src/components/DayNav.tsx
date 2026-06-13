import { useMemo, useRef } from 'react'
import { t } from '../i18n'
import { bestDay } from '../lib/betting'
import { useApp } from '../state/AppContext'

export default function DayNav() {
  const { data, active, setActive } = useApp()
  const navRef = useRef<HTMLDivElement>(null)
  const best = useMemo(() => bestDay(data.matches, data.dates), [data])

  const scroll = (dir: number) => navRef.current?.scrollBy({ left: dir * 180, behavior: 'smooth' })

  return (
    <div className="flex items-center gap-2 py-3">
      <button
        className="border-line text-mute hover:text-ink hidden h-8 w-8 shrink-0 cursor-pointer rounded-md border md:block"
        onClick={() => scroll(-1)}
      >
        ◀
      </button>
      <div ref={navRef} className="thin-scroll flex flex-1 gap-2 overflow-x-auto pb-1">
        <DayTab
          active={active === 'ALL'}
          onClick={() => setActive('ALL')}
          line1={t('all_tab')}
          line2={`${data.matches.length} ${t('matches_word')}`}
        />
        {data.dates.map((d) => {
          const dayMatches = data.matches.filter((m) => m.date === d)
          const highConf = dayMatches.filter((m) => m.stars >= 4).length
          const p = d.split('-')
          return (
            <DayTab
              key={d}
              active={active === d}
              crown={d === best}
              onClick={() => setActive(d)}
              line1={`${p[2]}/${p[1]}`}
              sub={(data.dateLabels[d] || '').split(' ')[0]}
              line2={`${highConf}★★★★ · ${dayMatches.length} ${t('matches_short')}`}
            />
          )
        })}
      </div>
      <button
        className="border-line text-mute hover:text-ink hidden h-8 w-8 shrink-0 cursor-pointer rounded-md border md:block"
        onClick={() => scroll(1)}
      >
        ▶
      </button>
    </div>
  )
}

function DayTab({
  active,
  crown,
  onClick,
  line1,
  sub,
  line2,
}: {
  active: boolean
  crown?: boolean
  onClick: () => void
  line1: string
  sub?: string
  line2: string
}) {
  return (
    <button
      onClick={onClick}
      className={`min-w-27 shrink-0 cursor-pointer rounded-lg border px-3.5 py-2 text-center transition-all ${
        active
          ? 'border-gold/70 bg-raised shadow-[0_0_0_1px_rgba(240,180,41,0.2)]'
          : 'border-line bg-surface hover:border-gold/40'
      }`}
    >
      <div className="font-display text-[17px] leading-tight font-bold tracking-wide">
        {crown && <span className="mr-1">👑</span>}
        {line1}
      </div>
      {sub && <div className={`text-[11px] font-semibold ${crown ? 'text-gold' : 'text-mute'}`}>{sub}</div>}
      <div className="text-mute text-[10px]">{line2}</div>
    </button>
  )
}

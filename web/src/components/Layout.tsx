import { useState } from 'react'
import type { ReactNode } from 'react'
import Sidebar from './Sidebar'
import DayNav from './DayNav'

export default function Layout({ children }: { children: ReactNode }) {
  const [drawer, setDrawer] = useState(false)

  return (
    <div className="flex min-h-screen">
      {/* desktop sidebar */}
      <aside className="border-line bg-surface/60 sticky top-0 hidden h-screen w-56 shrink-0 border-r backdrop-blur-sm lg:block">
        <Sidebar />
      </aside>

      {/* mobile drawer */}
      {drawer && (
        <div className="fixed inset-0 z-50 lg:hidden" onClick={() => setDrawer(false)}>
          <div className="absolute inset-0 bg-black/70" />
          <aside
            className="bg-surface border-line absolute top-0 left-0 h-full w-64 overflow-y-auto border-r shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <Sidebar onNavigate={() => setDrawer(false)} />
          </aside>
        </div>
      )}

      <div className="min-w-0 flex-1">
        {/* header */}
        <header className="border-line bg-surface/85 sticky top-0 z-40 flex items-center gap-3 border-b px-4 py-3 backdrop-blur-md md:px-6">
          <button
            className="border-line text-ink hover:border-teal cursor-pointer rounded-md border px-2.5 py-1.5 text-sm lg:hidden"
            onClick={() => setDrawer(true)}
            aria-label="menu"
          >
            ☰
          </button>
          <div className="font-display text-xl font-extrabold tracking-wider uppercase">
            Bet<span className="text-gold">Winning</span>Games
          </div>
          <div className="border-line text-mute hidden rounded border px-2 py-0.5 text-[10px] tracking-widest uppercase sm:block">
            DC · Elo · Draw/O25 ML · Pinnacle CLV
          </div>
          <span className="bg-green ml-auto inline-block h-1.5 w-1.5 animate-pulse rounded-full" title="live" />
        </header>

        <div className="mx-auto max-w-350 px-3 pb-16 md:px-6">
          <DayNav />
          {children}
        </div>
      </div>
    </div>
  )
}

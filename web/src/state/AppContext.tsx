/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { AppData } from '../data/loadData'
import { loadBankroll, loadNavState, saveNavState } from '../lib/storage'
import type { MatchPred, SlipTab, SortMode } from '../types'
import { VIEW_IDS } from '../types'

interface AppContextValue {
  data: AppData
  active: string // view id or date string ("activeDate" in bwg_state)
  league: string
  sort: SortMode
  setActive: (v: string) => void
  setLeague: (v: string) => void
  setSort: (v: SortMode) => void
  // match detail modal
  modalMatch: MatchPred | null
  openMatch: (m: MatchPred) => void
  closeMatch: () => void
  // bankroll init modal
  bankrollModal: boolean
  setBankrollModal: (b: boolean) => void
  /** true when a bankroll exists; otherwise opens the init modal */
  requireBankroll: () => boolean
  // slip tab (shared so parlay loaders can switch to it)
  slipTab: SlipTab
  setSlipTab: (tab: SlipTab) => void
  // autopilot wizard
  wizardOpen: boolean
  setWizardOpen: (b: boolean) => void
}

const Ctx = createContext<AppContextValue | null>(null)

function isValidTarget(v: string | undefined, dates: string[]): v is string {
  return !!v && ((VIEW_IDS as readonly string[]).includes(v) || dates.includes(v))
}

function fromHash(): string {
  return decodeURIComponent(location.hash.replace(/^#\/?/, ''))
}

export function AppProvider({ data, children }: { data: AppData; children: ReactNode }) {
  const saved = loadNavState()

  const [active, setActiveRaw] = useState<string>(() => {
    const h = fromHash()
    if (isValidTarget(h, data.dates)) return h
    if (isValidTarget(saved.activeDate, data.dates)) return saved.activeDate
    return data.dates[0] || 'ALL'
  })
  const [league, setLeagueRaw] = useState<string>(saved.activeLeague || 'ALL')
  const [sort, setSortRaw] = useState<SortMode>(
    saved.sortMode === 'prob' || saved.sortMode === 'time' ? saved.sortMode : 'stars',
  )
  const [modalMatch, setModalMatch] = useState<MatchPred | null>(null)
  const [bankrollModal, setBankrollModal] = useState(false)
  const [slipTab, setSlipTab] = useState<SlipTab>('active')
  const [wizardOpen, setWizardOpen] = useState(false)

  // persist nav state (legacy bwg_state shape) + hash
  useEffect(() => {
    saveNavState({ activeDate: active, activeLeague: league, sortMode: sort })
    const target = '#/' + active
    if (location.hash !== target) {
      history.replaceState(null, '', target)
    }
  }, [active, league, sort])

  // back/forward + manual hash edits
  useEffect(() => {
    const onHash = () => {
      const h = fromHash()
      if (isValidTarget(h, data.dates)) setActiveRaw(h)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [data.dates])

  const setActive = useCallback((v: string) => {
    setActiveRaw(v)
    window.scrollTo({ top: 0 })
  }, [])

  const requireBankroll = useCallback((): boolean => {
    if (loadBankroll()) return true
    setBankrollModal(true)
    return false
  }, [])

  const openMatch = useCallback((m: MatchPred) => setModalMatch(m), [])
  const closeMatch = useCallback(() => setModalMatch(null), [])

  const value = useMemo<AppContextValue>(
    () => ({
      data,
      active,
      league,
      sort,
      setActive,
      setLeague: setLeagueRaw,
      setSort: setSortRaw,
      modalMatch,
      openMatch,
      closeMatch,
      bankrollModal,
      setBankrollModal,
      requireBankroll,
      slipTab,
      setSlipTab,
      wizardOpen,
      setWizardOpen,
    }),
    [data, active, league, sort, modalMatch, bankrollModal, slipTab, wizardOpen, setActive, openMatch, closeMatch, requireBankroll],
  )

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp(): AppContextValue {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp outside AppProvider')
  return v
}

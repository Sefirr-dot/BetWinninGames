import { useEffect, useState } from 'react'
import { loadData, startAutoReload } from './data/loadData'
import type { AppData } from './data/loadData'
import { t } from './i18n'
import { runSettlement } from './lib/actions'
import { loadBankroll } from './lib/storage'
import { AppProvider, useApp } from './state/AppContext'
import Layout from './components/Layout'
import BankrollModal from './components/BankrollModal'
import MatchModal from './components/MatchModal'
import Toast from './components/Toast'
import { EmptyState } from './components/ui'
import AutopilotView from './views/AutopilotView'
import BacktestView from './views/BacktestView'
import BestBetsView from './views/BestBetsView'
import BetNowView from './views/BetNowView'
import MatchesView from './views/MatchesView'
import SlipView from './views/SlipView'
import TrackView from './views/TrackView'
import ValueBetsView from './views/ValueBetsView'
import WinielaView from './views/WinielaView'

function ViewRouter() {
  const { active } = useApp()
  switch (active) {
    case 'AUTO':
      return <AutopilotView />
    case 'SLIP':
      return <SlipView />
    case 'BETNOW':
      return <BetNowView />
    case 'BEST':
      return <BestBetsView />
    case 'VALUE':
      return <ValueBetsView />
    case 'TRACK':
      return <TrackView />
    case 'BACK':
      return <BacktestView />
    case 'WIN':
      return <WinielaView />
    default:
      return <MatchesView />
  }
}

/** First-visit bankroll prompt + auto-settlement (monolith initBankrollSystem). */
function InitEffects() {
  const { data, setBankrollModal } = useApp()
  useEffect(() => {
    if (!loadBankroll()) setBankrollModal(true)
    else runSettlement(data.results)
  }, [data, setBankrollModal])
  return null
}

export default function App() {
  const [data, setData] = useState<AppData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    loadData()
      .then((d) => {
        if (alive) setData(d)
      })
      .catch(() => {
        if (alive) setError('missing')
      })
    const stop = startAutoReload()
    return () => {
      alive = false
      stop()
    }
  }, [])

  if (error) {
    return (
      <EmptyState icon="⚽" title={t('data_missing')} msg={t('data_missing_sub')} code="python main.py" />
    )
  }
  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-mute animate-pulse text-sm tracking-widest uppercase">{t('loading')}</div>
      </div>
    )
  }

  return (
    <AppProvider data={data}>
      <InitEffects />
      <Layout>
        <ViewRouter />
      </Layout>
      <MatchModal />
      <BankrollModal />
      <Toast />
    </AppProvider>
  )
}

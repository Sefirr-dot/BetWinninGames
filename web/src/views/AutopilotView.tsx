// ─────────────────────────────────────────────────────────────
// AUTO view — Autopilot wizard + weekend plan
// (monolith renderAutopilotView + autopilotAnalyzeAI).
// ─────────────────────────────────────────────────────────────
import { useState } from 'react'
import { LANG, t, tf } from '../i18n'
import { selectAutopilotBets } from '../lib/betting'
import { autopilotAddAll } from '../lib/actions'
import { ollamaChat } from '../lib/ollama'
import { outcomeColor, outcomeLabel, leagueColor } from '../lib/format'
import { saveProfile, useBankroll, useProfile } from '../lib/storage'
import { useApp } from '../state/AppContext'
import { LeagueBadge, StarsInline } from '../components/ui'
import type { AutopilotProfile, RiskLevel } from '../types'

const DAY_NAMES: Record<string, string[]> = {
  es: ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'],
  en: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
}

const RISKS: { value: RiskLevel; icon: string; labelKey: string }[] = [
  { value: 'conservative', icon: '🛡️', labelKey: 'auto_risk_con' },
  { value: 'balanced', icon: '⚖️', labelKey: 'auto_risk_bal' },
  { value: 'aggressive', icon: '🚀', labelKey: 'auto_risk_agg' },
]

const ALL_LEAGUES = ['PL', 'PD', 'BL1', 'FL1']

function Wizard({ onDone }: { onDone: () => void }) {
  const { data } = useApp()
  const existing = useProfile()
  const [risk, setRisk] = useState<RiskLevel>(existing?.riskLevel || 'balanced')
  const [maxBets, setMaxBets] = useState<number>(existing?.maxBetsPerDay || 3)
  const [leagues, setLeagues] = useState<string[]>(existing?.leagues || [...ALL_LEAGUES])
  const [dates, setDates] = useState<string[]>(
    existing?.activeDates?.filter((d) => data.dates.includes(d)) ?? [...data.dates],
  )

  const toggle = (arr: string[], v: string, set: (a: string[]) => void) =>
    set(arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v])

  const save = () => {
    if (!leagues.length) return
    const profile: AutopilotProfile = {
      riskLevel: risk,
      maxBetsPerDay: maxBets,
      leagues,
      activeDates: dates.length ? dates : [...data.dates],
      markets: ['1x2', 'over25', 'btts'],
      minOdds: 1.4,
      minStars: 5,
      maxExposurePct: 10,
    }
    saveProfile(profile)
    onDone()
  }

  const names = DAY_NAMES[LANG] || DAY_NAMES.en

  return (
    <div className="panel modal-pop mx-auto max-w-130 px-6 py-7">
      <div className="text-center">
        <div className="text-3xl">🤖</div>
        <div className="font-display mt-2 text-2xl font-extrabold tracking-wide">{t('auto_wiz_title')}</div>
        <div className="text-mute mt-1 text-[13px]">{t('auto_wiz_sub')}</div>
      </div>

      <div className="micro-label mt-6 mb-2 font-bold">{t('auto_wiz_risk')}</div>
      <div className="grid grid-cols-3 gap-2">
        {RISKS.map((r) => (
          <button
            key={r.value}
            onClick={() => setRisk(r.value)}
            className={`cursor-pointer rounded-lg border px-2 py-3 text-center transition-colors ${
              risk === r.value ? 'border-teal bg-teal/12 text-teal' : 'border-line text-mute hover:text-ink'
            }`}
          >
            <div className="text-base">{r.icon}</div>
            <div className="mt-1 text-[11px] font-semibold">{t(r.labelKey)}</div>
          </button>
        ))}
      </div>

      <div className="micro-label mt-5 mb-2 font-bold">{t('auto_wiz_max')}</div>
      <div className="grid grid-cols-5 gap-2">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            onClick={() => setMaxBets(n)}
            className={`stat-num cursor-pointer rounded-lg border py-2.5 text-[22px] transition-colors ${
              maxBets === n ? 'border-teal bg-teal/12 text-teal' : 'border-line text-mute hover:text-ink'
            }`}
          >
            {n}
          </button>
        ))}
      </div>

      <div className="micro-label mt-5 mb-2 font-bold">{t('auto_wiz_lg')}</div>
      <div className="flex flex-wrap gap-2">
        {ALL_LEAGUES.map((lg) => {
          const on = leagues.includes(lg)
          const c = leagueColor(lg)
          return (
            <button
              key={lg}
              onClick={() => toggle(leagues, lg, setLeagues)}
              className="font-display cursor-pointer rounded-lg border px-4 py-1.5 text-sm font-bold tracking-wide uppercase transition-colors"
              style={
                on
                  ? { background: c + '22', color: c, borderColor: c + '88' }
                  : { borderColor: 'var(--color-line)', color: 'var(--color-mute)', opacity: 0.6 }
              }
            >
              {lg}
            </button>
          )
        })}
      </div>

      <div className="micro-label mt-5 mb-2 font-bold">{t('auto_wiz_dates')}</div>
      <div className="flex flex-wrap gap-2">
        {data.dates.map((d) => {
          const [y, mo, day] = d.split('-').map(Number)
          const dow = new Date(y, mo - 1, day).getDay()
          const on = dates.includes(d)
          return (
            <button
              key={d}
              onClick={() => toggle(dates, d, setDates)}
              className={`cursor-pointer rounded-lg border px-3.5 py-2 text-center text-[11px] font-semibold transition-colors ${
                on ? 'border-teal bg-teal/15 text-teal' : 'border-line text-mute hover:text-ink'
              }`}
            >
              {names[dow]}
              <div className="stat-num text-lg">
                {String(day).padStart(2, '0')}/{String(mo).padStart(2, '0')}
              </div>
            </button>
          )
        })}
      </div>

      <button
        onClick={save}
        disabled={!leagues.length}
        className="bg-teal font-display mt-7 w-full cursor-pointer rounded-xl py-3.5 text-lg font-bold text-black disabled:opacity-50"
      >
        {t('auto_wiz_btn')}
      </button>
    </div>
  )
}

export default function AutopilotView() {
  const { data, setActive, setSlipTab, setBankrollModal, openMatch, wizardOpen, setWizardOpen } = useApp()
  const profile = useProfile()
  const bankroll = useBankroll()
  const [aiBusy, setAiBusy] = useState(false)
  const [aiText, setAiText] = useState<string | null>(null)
  const [aiErr, setAiErr] = useState(false)

  if (!profile || wizardOpen) {
    return (
      <div className="py-6">
        <Wizard onDone={() => setWizardOpen(false)} />
      </div>
    )
  }
  if (!bankroll) {
    return (
      <div className="px-6 py-14 text-center">
        <div className="text-4xl">💰</div>
        <button
          onClick={() => setBankrollModal(true)}
          className="bg-green mt-5 cursor-pointer rounded-lg px-5 py-2.5 text-sm font-bold text-black"
        >
          {t('btn_config_bkrl')}
        </button>
      </div>
    )
  }

  const { picks, relaxedStars } = selectAutopilotBets(data.matches, profile, bankroll, data.trackerMetrics)
  const totalStake = picks.reduce((s, p) => s + p.stake, 0)
  const totalEV = picks.reduce((s, p) => s + p.ev, 0)
  const exposurePct = bankroll.current > 0 ? (totalStake / bankroll.current) * 100 : 0
  const hasCorr = picks.some((p) => p.correlated)
  const riskLabel = t(
    profile.riskLevel === 'conservative' ? 'auto_risk_con' : profile.riskLevel === 'aggressive' ? 'auto_risk_agg' : 'auto_risk_bal',
  )

  const addAll = () => {
    const res = autopilotAddAll(picks)
    if (res === 'NO_BANKROLL') {
      setBankrollModal(true)
      return
    }
    if (res !== 'OK') return
    setSlipTab('active')
    setActive('SLIP')
  }

  const analyzeAI = async () => {
    if (!picks.length) {
      setAiErr(false)
      setAiText(t('auto_no_analyze'))
      return
    }
    setAiBusy(true)
    setAiErr(false)
    setAiText(t('ollama_querying'))

    const riskEn = { conservative: 'Conservative', balanced: 'Balanced', aggressive: 'Aggressive' }[profile.riskLevel] || 'Balanced'
    const picksText = picks
      .map((p, i) => {
        const outcome =
          { home: 'Home Win', draw: 'Draw', away: 'Away Win', over25: 'Over 2.5', btts: 'BTTS' }[p.vb.outcome] ||
          p.vb.outcome
        return `${i + 1}. ${p.m.homeShort} vs ${p.m.awayShort} [${p.m.league}] ${p.m.stars}★ | ${outcome} @${p.vb.bkOdds} | Edge ${p.vb.edge}% | Stake €${p.stake} | EV +€${p.ev}${p.correlated ? ' [CORRELATED]' : ''}`
      })
      .join('\n')

    const prompt = `You are a professional sports betting analyst. Analyze this automated betting plan in ${LANG === 'es' ? 'Spanish' : 'English'}.

PROFILE: ${riskEn} risk | Max ${profile.maxBetsPerDay || 3} bets/day
BANKROLL: €${bankroll.current.toFixed(2)} (initial €${bankroll.initial.toFixed(2)})
TOTAL STAKE: €${totalStake.toFixed(2)} (${exposurePct.toFixed(1)}% of bankroll)
TOTAL EV: +€${totalEV.toFixed(2)}

PICKS:
${picksText}

Respond ONLY with:
OVERALL RISK: (Low/Medium/High)
PLAN QUALITY: (1-10 score with brief reason)
PER-PICK: (one line per pick: pick number, brief comment on value/concern)
RECOMMENDATION: (1-2 sentences of concrete advice)`

    try {
      const txt = await ollamaChat(prompt, 90000)
      setAiText(txt || '—')
    } catch {
      setAiErr(true)
      setAiText(t('ollama_unavail'))
    } finally {
      setAiBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-210">
      {/* header */}
      <div className="border-line flex flex-wrap items-center justify-between gap-3 border-b pb-4">
        <div>
          <div className="font-display text-2xl font-extrabold tracking-wide">🤖 Autopilot</div>
          <div className="text-mute mt-0.5 text-xs">
            {picks.length} {t('auto_sub')}
          </div>
        </div>
        <button
          onClick={() => setWizardOpen(true)}
          className="bg-raised border-line hover:border-teal cursor-pointer rounded-lg border px-3.5 py-2 text-[13px] transition-colors"
        >
          {t('auto_settings')} · {riskLabel}
        </button>
      </div>

      {/* summary bar */}
      {picks.length > 0 && (
        <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
          {(
            [
              [String(picks.length), t('auto_bets'), 'var(--color-teal)', null],
              [`€${totalStake.toFixed(2)}`, t('auto_exposure'), 'var(--color-amber)', `${exposurePct.toFixed(1)}%`],
              [`+€${totalEV.toFixed(2)}`, t('auto_ev_total'), totalEV >= 0 ? 'var(--color-green)' : 'var(--color-orange)', null],
              [`€${bankroll.current.toFixed(0)}`, t('lbl_bankroll'), 'var(--color-ink)', null],
            ] as [string, string, string, string | null][]
          ).map(([v, l, c, sub]) => (
            <div key={l} className="panel px-3 py-2.5 text-center">
              <div className="stat-num text-xl" style={{ color: c }}>
                {v}
                {sub && <span className="text-mute ml-1 text-[10px] font-normal">{sub}</span>}
              </div>
              <div className="micro-label mt-0.5">{l}</div>
            </div>
          ))}
        </div>
      )}

      {/* warnings */}
      {relaxedStars && (
        <div className="bg-gold/10 border-gold/30 text-gold mt-3 rounded-lg border px-3.5 py-2 text-xs">
          ⚡ {tf('auto_relaxed', profile.minStars || 5, relaxedStars)}
        </div>
      )}
      {hasCorr && (
        <div className="bg-orange/10 border-orange/30 text-orange mt-3 rounded-lg border px-3.5 py-2 text-xs">
          {t('auto_corr')}
        </div>
      )}

      {/* picks */}
      <div className="mt-4 flex flex-col gap-2.5">
        {picks.length === 0 ? (
          <div className="px-6 py-14 text-center">
            <div className="text-4xl">🤖</div>
            <div className="text-mute mt-3 text-sm whitespace-pre-line">{t('auto_no_picks')}</div>
          </div>
        ) : (
          picks.map((p, i) => (
            <div
              key={`${p.date}|${p.m.home}|${p.m.away}`}
              className="panel hover:border-teal/50 flex cursor-pointer items-center gap-3.5 p-3.5 transition-colors"
              onClick={() => openMatch(p.m)}
            >
              <div className="stat-num bg-teal/12 text-teal flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[15px]">
                {i + 1}
              </div>
              <div className="min-w-0 flex-1">
                <div className="font-display text-base font-bold tracking-wide">
                  {p.m.homeShort} vs {p.m.awayShort}
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[12px]">
                  <LeagueBadge league={p.m.league} />
                  <span className="font-semibold" style={{ color: outcomeColor(p.vb.outcome) }}>
                    {outcomeLabel(p.vb.outcome)}
                  </span>
                  <span>@{p.vb.bkOdds}</span>
                  <span className="text-mute">
                    {t('lbl_edge')} {p.vb.edge}%
                  </span>
                  {p.correlated && <span className="text-orange text-[10px]">⚡ corr</span>}
                </div>
                <div className="bg-raised mt-1.5 h-1 overflow-hidden rounded">
                  <div
                    className="bg-teal h-full rounded"
                    style={{ width: `${Math.min(p.vb.edge * 4, 100)}%` }}
                  />
                </div>
              </div>
              <div className="shrink-0 text-right">
                <div className="stat-num text-green text-xl">€{p.stake}</div>
                <div className="text-teal text-[11px]">
                  {t('lbl_ev')} +€{p.ev}
                </div>
                <div className="mt-0.5">
                  <StarsInline n={p.m.stars} />
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* footer */}
      {picks.length > 0 && (
        <div className="border-line mt-5 flex flex-wrap gap-2.5 border-t pt-4">
          <button
            disabled={aiBusy}
            onClick={analyzeAI}
            className="bg-surface border-line cursor-pointer rounded-lg border px-4 py-3 text-sm font-semibold disabled:opacity-60"
          >
            {aiBusy ? t('btn_analyzing') : `🤖 ${t('auto_analyze_ai')}`}
          </button>
          <button
            onClick={addAll}
            className="bg-teal font-display flex-1 cursor-pointer rounded-lg px-4 py-3 text-base font-bold text-black"
          >
            {t('auto_accept')}
          </button>
        </div>
      )}
      {aiText && (
        <div
          className="bg-raised/60 border-line mt-3 rounded-lg border px-4 py-3 text-[13px] leading-relaxed whitespace-pre-line"
          style={aiErr ? { color: 'var(--color-orange)' } : undefined}
        >
          {aiText}
        </div>
      )}
    </div>
  )
}

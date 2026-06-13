// ─────────────────────────────────────────────────────────────
// localStorage store — byte-compatible with the legacy page.
// Keys: bwg_state, bwg_bankroll, bwg_slip, bwg_history, bwg_profile.
// Reactive via useSyncExternalStore.
// ─────────────────────────────────────────────────────────────
import { useSyncExternalStore } from 'react'
import type { AutopilotProfile, Bankroll, HistoryBet, NavStateStored, SlipItem } from '../types'

export const KEY_STATE = 'bwg_state'
export const KEY_BANKROLL = 'bwg_bankroll'
export const KEY_SLIP = 'bwg_slip'
export const KEY_HISTORY = 'bwg_history'
export const KEY_PROFILE = 'bwg_profile'

type Listener = () => void
const listeners = new Set<Listener>()

function emit() {
  listeners.forEach((l) => l())
}

export function subscribe(l: Listener): () => void {
  listeners.add(l)
  return () => {
    listeners.delete(l)
  }
}

// Parsed-value cache so getSnapshot returns stable references.
const cache = new Map<string, unknown>()

function read<T>(key: string, fallback: T): T {
  if (cache.has(key)) return cache.get(key) as T
  let v: T = fallback
  try {
    const raw = localStorage.getItem(key)
    if (raw !== null) {
      const parsed = JSON.parse(raw) as T | null
      // legacy `|| []` semantics: stored null falls back too
      v = (parsed ?? fallback) as T
    }
  } catch {
    v = fallback
  }
  cache.set(key, v)
  return v
}

function write<T>(key: string, value: T): void {
  cache.set(key, value)
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* quota / private mode — keep in-memory value */
  }
  emit()
}

function remove(key: string): void {
  cache.delete(key)
  try {
    localStorage.removeItem(key)
  } catch {
    /* ignore */
  }
  emit()
}

// Cross-tab sync (e.g. legacy page open alongside)
window.addEventListener('storage', (e) => {
  if (e.key === null || e.key.startsWith('bwg_')) {
    cache.clear()
    emit()
  }
})

// ── typed accessors ──────────────────────────────────────────

export const loadBankroll = (): Bankroll | null => read<Bankroll | null>(KEY_BANKROLL, null)
export const saveBankroll = (b: Bankroll): void => write(KEY_BANKROLL, b)
export const removeBankroll = (): void => remove(KEY_BANKROLL)

export const loadSlip = (): SlipItem[] => read<SlipItem[]>(KEY_SLIP, [])
export const saveSlip = (s: SlipItem[]): void => write(KEY_SLIP, s)

export const loadHistory = (): HistoryBet[] => read<HistoryBet[]>(KEY_HISTORY, [])
export const saveHistory = (h: HistoryBet[]): void => write(KEY_HISTORY, h)
export const removeHistory = (): void => remove(KEY_HISTORY)

export const loadProfile = (): AutopilotProfile | null => read<AutopilotProfile | null>(KEY_PROFILE, null)
export const saveProfile = (p: AutopilotProfile): void => write(KEY_PROFILE, p)

export const loadNavState = (): NavStateStored => read<NavStateStored>(KEY_STATE, {})
export const saveNavState = (s: NavStateStored): void => write(KEY_STATE, s)

// ── React hooks ──────────────────────────────────────────────

export function useBankroll(): Bankroll | null {
  return useSyncExternalStore(subscribe, () => read<Bankroll | null>(KEY_BANKROLL, null))
}

export function useSlip(): SlipItem[] {
  return useSyncExternalStore(subscribe, () => read<SlipItem[]>(KEY_SLIP, []))
}

export function useHistory(): HistoryBet[] {
  return useSyncExternalStore(subscribe, () => read<HistoryBet[]>(KEY_HISTORY, []))
}

export function useProfile(): AutopilotProfile | null {
  return useSyncExternalStore(subscribe, () => read<AutopilotProfile | null>(KEY_PROFILE, null))
}

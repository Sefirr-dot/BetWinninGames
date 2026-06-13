import { useEffect, useRef, useState } from 'react'
import { onToast } from '../lib/toast'

export default function Toast() {
  const [msg, setMsg] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(
    () =>
      onToast((m) => {
        setMsg(m)
        if (timer.current) clearTimeout(timer.current)
        timer.current = setTimeout(() => setMsg(null), 5000)
      }),
    [],
  )

  if (!msg) return null
  return (
    <div className="toast-in border-teal bg-surface fixed bottom-6 left-1/2 z-[99999] max-w-[90vw] -translate-x-1/2 rounded-xl border px-6 py-3 text-center text-[13px] font-semibold whitespace-nowrap shadow-[0_4px_24px_rgba(0,0,0,0.5)]">
      {msg}
    </div>
  )
}

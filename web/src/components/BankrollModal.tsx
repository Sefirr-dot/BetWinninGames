import { useState } from 'react'
import { t } from '../i18n'
import { confirmBankroll, runSettlement } from '../lib/actions'
import { useApp } from '../state/AppContext'

export default function BankrollModal() {
  const { data, bankrollModal, setBankrollModal } = useApp()
  const [val, setVal] = useState('1000')

  if (!bankrollModal) return null

  const confirm = () => {
    const v = parseFloat(val)
    if (confirmBankroll(v)) {
      setBankrollModal(false)
      runSettlement(data.results)
    }
  }

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/75 p-4">
      <div className="modal-pop panel w-full max-w-95 px-9 py-9 text-center">
        <div className="text-4xl">💰</div>
        <div className="font-display mt-3 text-2xl font-extrabold tracking-wide">{t('modal_title')}</div>
        <div className="text-mute mt-2 mb-6 text-[13px]">{t('modal_sub')}</div>
        <div className="mb-6 flex items-center justify-center gap-2">
          <span className="text-mute text-2xl">€</span>
          <input
            type="number"
            min={1}
            max={1000000}
            step={50}
            value={val}
            onChange={(e) => setVal(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') confirm()
            }}
            className="bg-raised border-line stat-num w-40 rounded-lg border px-4 py-2.5 text-center text-[26px] outline-none"
          />
        </div>
        <button
          onClick={confirm}
          className="bg-green font-display w-full cursor-pointer rounded-xl py-3.5 text-lg font-bold text-black"
        >
          {t('btn_start')}
        </button>
      </div>
    </div>
  )
}

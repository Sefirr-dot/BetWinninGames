import { useCallback } from 'react'
import { addToSlip } from '../lib/actions'
import type { AddToSlipArgs } from '../lib/actions'
import { useApp } from './AppContext'

/**
 * addToSlip with the monolith's bankroll guard: when no bankroll
 * exists, the init modal opens instead (pick is not added).
 */
export function useAddToSlip(): (args: AddToSlipArgs) => void {
  const { setBankrollModal } = useApp()
  return useCallback(
    (args: AddToSlipArgs) => {
      if (addToSlip(args) === 'NO_BANKROLL') setBankrollModal(true)
    },
    [setBankrollModal],
  )
}

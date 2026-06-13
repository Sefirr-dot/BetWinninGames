// Tiny global toast bus
type ToastListener = (msg: string) => void
const listeners = new Set<ToastListener>()

export function onToast(l: ToastListener): () => void {
  listeners.add(l)
  return () => {
    listeners.delete(l)
  }
}

export function toast(msg: string): void {
  listeners.forEach((l) => l(msg))
}

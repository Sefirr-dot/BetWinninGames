// ── Ollama — browser → localhost:11434, same call shape as the
// monolith: model qwen3.5:9b, stream:false, think:false, temp 0.1.
const OLLAMA_URL = 'http://localhost:11434/api/chat'
const OLLAMA_MODEL = 'qwen3.5:9b'

interface OllamaResponse {
  message?: { content?: string }
}

/**
 * Sends one user prompt; resolves with the model's text.
 * Throws on network failure / timeout — caller shows the
 * "Ollama unavailable" message.
 */
export async function ollamaChat(prompt: string, timeoutMs = 60000): Promise<string> {
  const r = await fetch(OLLAMA_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(timeoutMs),
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      messages: [{ role: 'user', content: prompt }],
      stream: false,
      think: false,
      options: { temperature: 0.1 },
    }),
  })
  const d = (await r.json()) as OllamaResponse
  return d.message?.content || ''
}

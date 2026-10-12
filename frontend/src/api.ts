export type Mode = 'hybrid' | 'dense' | 'sparse'

export interface Book {
  book_id: string
  title: string
  author: string
  language: 'en' | 'de'
  year: number
  era: string
  subjects: string[]
  description: string
  gutenberg_url: string
  num_passages: number
  score: number
}

export interface Source {
  n: number
  book_id: string
  title: string
  author: string
  chapter_idx: number
  chapter_title: string
  passage_idx: number
  text: string
  score: number
  dense_score: number | null
}

export interface Facets {
  eras: { name: string; count: number }[]
  languages: { name: string; count: number }[]
}

export interface Health {
  index: { books: number; passages: number }
  llm: { reachable: boolean; model: string; model_ready: boolean }
}

export interface DoneEvent {
  mode: 'generative' | 'extractive'
  text: string
  citations: number[]
  invalid_citations: number[]
  uncited?: boolean
  latency_s: number
  ttft_s?: number | null
  tokens_per_s?: number | null
  model?: string
}

export type ChatEvent =
  | { event: 'sources'; sources: Source[] }
  | { event: 'token'; text: string }
  | { event: 'refusal'; reason: string; text: string }
  | { event: 'fallback'; reason: string; text: string }
  | ({ event: 'done' } & DoneEvent)

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json() as Promise<T>
}

export const api = {
  health: () => getJson<Health>('/api/health'),
  facets: () => getJson<Facets>('/api/facets'),
  search: (q: string, mode: Mode, language: string, era: string) => {
    const p = new URLSearchParams({ q, mode, limit: '12' })
    if (language) p.set('language', language)
    if (era) p.set('era', era)
    return getJson<{ results: Book[] }>(`/api/search?${p}`)
  },
}

/** POST /api/chat and parse the server-sent event stream. */
export async function streamChat(
  body: { question: string; book_id?: string | null; history: { role: string; content: string }[] },
  onEvent: (ev: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!r.ok || !r.body) throw new Error(`${r.status} ${r.statusText}`)
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const data = frame.split('\n').find((l) => l.startsWith('data: '))
      if (data) onEvent(JSON.parse(data.slice(6)) as ChatEvent)
    }
  }
}

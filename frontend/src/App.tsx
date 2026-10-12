import { useEffect, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { api, streamChat } from './api'
import type { Book, DoneEvent, Facets, Health, Mode, Source } from './api'

const LANG_LABEL: Record<string, string> = { en: 'English', de: 'Deutsch' }

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  done?: DoneEvent
  refusal?: string
  notice?: string
  streaming?: boolean
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [selected, setSelected] = useState<Book | null>(null)
  const [passage, setPassage] = useState<Source | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/favicon.svg" alt="" width={28} height={28} />
          <div>
            <h1>SemanticBookAI</h1>
            <p>Hybrid search and grounded chat over public-domain books</p>
          </div>
        </div>
        <StatusBadge health={health} />
      </header>
      <main className="layout">
        <CatalogPanel selected={selected} onSelect={setSelected} />
        <ChatPanel book={selected} onClearBook={() => setSelected(null)} onOpenSource={setPassage} />
      </main>
      {passage && <PassageDrawer source={passage} onClose={() => setPassage(null)} />}
    </div>
  )
}

function StatusBadge({ health }: { health: Health | null }) {
  if (!health) return <span className="badge warn">API offline</span>
  const llm = health.llm
  const llmText = !llm.reachable ? 'LLM offline' : llm.model_ready ? llm.model : `${llm.model} not pulled`
  return (
    <div className="status">
      <span className="badge">{health.index.books} books · {health.index.passages.toLocaleString('en')} passages</span>
      <span className={`badge ${llm.reachable && llm.model_ready ? 'ok' : 'warn'}`}>Local LLM: {llmText}</span>
    </div>
  )
}

function CatalogPanel({ selected, onSelect }: { selected: Book | null; onSelect: (b: Book | null) => void }) {
  const [q, setQ] = useState('a whaling voyage and obsession')
  const [mode, setMode] = useState<Mode>('hybrid')
  const [language, setLanguage] = useState('')
  const [era, setEra] = useState('')
  const [facets, setFacets] = useState<Facets | null>(null)
  const [results, setResults] = useState<Book[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.facets().then(setFacets).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!q.trim()) return
    const t = setTimeout(() => {
      setLoading(true)
      setError('')
      api
        .search(q, mode, language, era)
        .then((r) => setResults(r.results))
        .catch((e: Error) => setError(e.message))
        .finally(() => setLoading(false))
    }, 250)
    return () => clearTimeout(t)
  }, [q, mode, language, era])

  return (
    <section className="panel catalog">
      <h2>Catalog search</h2>
      <input
        className="search"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search by title, author, theme…"
        aria-label="Search the catalog"
      />
      <div className="filters">
        <div className="segmented" role="group" aria-label="Retrieval mode">
          {(['hybrid', 'dense', 'sparse'] as Mode[]).map((m) => (
            <button key={m} className={mode === m ? 'active' : ''} onClick={() => setMode(m)}
              title={m === 'hybrid' ? 'Dense + BM25 fused with Reciprocal Rank Fusion' : m === 'dense' ? 'Semantic embeddings only' : 'BM25 keywords only'}>
              {m === 'sparse' ? 'BM25' : m}
            </button>
          ))}
        </div>
        <select value={language} onChange={(e) => setLanguage(e.target.value)} aria-label="Language">
          <option value="">All languages</option>
          {facets?.languages.map((l) => (
            <option key={l.name} value={l.name}>{LANG_LABEL[l.name] ?? l.name} ({l.count})</option>
          ))}
        </select>
        <select value={era} onChange={(e) => setEra(e.target.value)} aria-label="Era">
          <option value="">All eras</option>
          {facets?.eras.map((x) => (
            <option key={x.name} value={x.name}>{x.name} ({x.count})</option>
          ))}
        </select>
      </div>
      {error && <p className="error">{error}</p>}
      <ol className={`results ${loading ? 'loading' : ''}`}>
        {results.map((b, i) => (
          <li key={b.book_id} className={selected?.book_id === b.book_id ? 'selected' : ''}>
            <button className="result" onClick={() => onSelect(selected?.book_id === b.book_id ? null : b)}>
              <span className="rank">{i + 1}</span>
              <span className="result-body">
                <span className="result-title">{b.title}</span>
                <span className="meta">
                  {b.author} · {b.year < 0 ? `${-b.year} BC` : b.year} · {LANG_LABEL[b.language]}
                </span>
                <span className="desc">{b.description}</span>
              </span>
            </button>
          </li>
        ))}
        {!loading && results.length === 0 && q && <li className="empty">No results.</li>}
      </ol>
    </section>
  )
}

function ChatPanel({ book, onClearBook, onOpenSource }: {
  book: Book | null
  onClearBook: () => void
  onOpenSource: (s: Source) => void
}) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }), [messages])

  const update = (fn: (m: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), fn(ms[ms.length - 1])])

  async function ask(question: string) {
    if (!question.trim() || busy) return
    const history = messages.filter((m) => !m.refusal).map((m) => ({ role: m.role, content: m.content }))
    setMessages((ms) => [...ms, { role: 'user', content: question }, { role: 'assistant', content: '', streaming: true }])
    setInput('')
    setBusy(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      await streamChat({ question, book_id: book?.book_id, history }, (ev) => {
        if (ev.event === 'sources') update((m) => ({ ...m, sources: ev.sources }))
        else if (ev.event === 'token') update((m) => ({ ...m, content: m.content + ev.text }))
        else if (ev.event === 'refusal') update((m) => ({ ...m, content: ev.text, refusal: ev.reason, streaming: false }))
        else if (ev.event === 'fallback') update((m) => ({ ...m, notice: ev.text }))
        else if (ev.event === 'done') update((m) => ({ ...m, content: ev.text || m.content, done: ev, streaming: false }))
      }, ctrl.signal)
    } catch (e) {
      if (!ctrl.signal.aborted) update((m) => ({ ...m, notice: `Request failed: ${(e as Error).message}` }))
    } finally {
      update((m) => ({ ...m, streaming: false }))
      setBusy(false)
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault()
    void ask(input)
  }

  const examples = book
    ? [`What happens at the beginning of ${book.title}?`, `Who are the main characters in ${book.title}?`]
    : ['Why does Ahab hunt the white whale?', 'Wie verwandelt sich Gregor Samsa?', 'What does Frankenstein’s creature ask of Victor?']

  return (
    <section className="panel chat">
      <div className="chat-head">
        <h2>Ask the books</h2>
        <span className="scope">
          {book ? (
            <>Scope: <strong>{book.title}</strong> <button className="link" onClick={onClearBook}>all books</button></>
          ) : 'Scope: all books · select a book to narrow down'}
        </span>
      </div>
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <p>Answers come only from the indexed full texts and cite book, chapter and passage. Questions in German or English.</p>
            <div className="examples">
              {examples.map((x) => <button key={x} onClick={() => void ask(x)}>{x}</button>)}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <MessageView key={i} m={m} onOpenSource={onOpenSource} />
        ))}
        <div ref={endRef} />
      </div>
      <form className="composer" onSubmit={submit}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a question about the books…" aria-label="Question" />
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()}>Stop</button>
        ) : (
          <button type="submit" disabled={!input.trim()}>Ask</button>
        )}
      </form>
    </section>
  )
}

function MessageView({ m, onOpenSource }: { m: Message; onOpenSource: (s: Source) => void }) {
  if (m.role === 'user') return <div className="msg user">{m.content}</div>
  const sources = m.sources ?? []
  const cited = new Set(m.done?.citations ?? [])
  const shown = m.done?.mode === 'extractive' ? sources.slice(0, 3) : sources.filter((s) => cited.has(s.n))
  return (
    <div className={`msg assistant ${m.refusal ? 'refusal' : ''}`}>
      {m.notice && <p className="notice">{m.notice}</p>}
      {m.content ? (
        <div className="answer">{renderWithCitations(m.content, sources, onOpenSource)}{m.streaming && <span className="caret" />}</div>
      ) : (
        m.streaming && <div className="thinking">{sources.length ? `Reading ${sources.length} passages…` : 'Searching…'}</div>
      )}
      {shown.length > 0 && (
        <div className="sources">
          {shown.map((s) => (
            <button key={s.n} className="source" onClick={() => onOpenSource(s)}>
              <span className="cite">{s.n}</span>
              <span><strong>{s.title}</strong> · {s.chapter_title} · ¶{s.passage_idx + 1}</span>
            </button>
          ))}
        </div>
      )}
      {m.done && m.done.mode === 'generative' && (
        <div className="foot">
          {m.done.model} · {m.done.latency_s}s{m.done.tokens_per_s ? ` · ${m.done.tokens_per_s} tok/s` : ''}
          {m.done.invalid_citations.length > 0 && ` · removed ${m.done.invalid_citations.length} invalid citation(s)`}
          {m.done.uncited && ' · ⚠ answer without citation'}
        </div>
      )}
    </div>
  )
}

function renderWithCitations(text: string, sources: Source[], onOpen: (s: Source) => void): ReactNode[] {
  const parts = text.split(/(\[\d{1,2}\])/g)
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d{1,2})\]$/)
    const src = m && sources.find((s) => s.n === Number(m[1]))
    if (src) {
      return (
        <button key={i} className="cite inline" title={`${src.title} – ${src.chapter_title}`} onClick={() => onOpen(src)}>
          {src.n}
        </button>
      )
    }
    return <span key={i}>{part}</span>
  })
}

function PassageDrawer({ source, onClose }: { source: Source; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label="Source passage">
        <button className="close" onClick={onClose} aria-label="Close">×</button>
        <p className="eyebrow">Source [{source.n}]</p>
        <h3>{source.title}</h3>
        <p className="meta">{source.author} · {source.chapter_title} · passage {source.passage_idx + 1}</p>
        <blockquote>{source.text}</blockquote>
        <p className="meta">
          RRF score {source.score}{source.dense_score != null ? ` · cosine ${source.dense_score}` : ''} ·{' '}
          <a href={`https://www.gutenberg.org/ebooks/${source.book_id.replace('pg', '')}`} target="_blank" rel="noreferrer">Project Gutenberg</a>
        </p>
      </aside>
    </div>
  )
}

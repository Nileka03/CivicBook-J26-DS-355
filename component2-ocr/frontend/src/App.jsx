import { useEffect, useState } from 'react'
import { health, listComplaints } from './api'
import ComplaintForm from './ComplaintForm'
import ComplaintResult from './ComplaintResult'

function History({ onOpen }) {
  const [rows, setRows] = useState(null)
  useEffect(() => {
    listComplaints().then(setRows).catch(() => setRows([]))
  }, [])
  if (!rows) return null
  if (!rows.length) return <p className="muted">No complaints yet.</p>
  return (
    <ul className="history">
      {rows.map((r) => (
        <li key={r.complaint_id}>
          <button onClick={() => onOpen(r.complaint_id)}>
            <span className="mono">{r.complaint_id}</span>
            <span>{r.complaint_date}</span>
            <span>{r.n_pages} page{r.n_pages === 1 ? '' : 's'}</span>
            <span className={`decision ${r.decision || ''}`}>{r.decision || r.status}</span>
          </button>
        </li>
      ))}
    </ul>
  )
}

export default function App() {
  const [view, setView] = useState({ name: 'form' })
  const [h, setH] = useState(null)

  useEffect(() => {
    health().then(setH).catch(() => setH({ status: 'down' }))
  }, [])

  return (
    <div className="app">
      <header>
        <div>
          <div className="brand">CivicBook</div>
          <div className="sub">Written complaints · Sinhala & English handwriting</div>
        </div>
        <nav>
          <button className={view.name === 'form' ? 'on' : ''} onClick={() => setView({ name: 'form' })}>New complaint</button>
          <button className={view.name === 'history' ? 'on' : ''} onClick={() => setView({ name: 'history' })}>Submitted</button>
        </nav>
      </header>

      {h && h.status !== 'ok' && <div className="err">Backend not reachable. Start it with: uvicorn app.main:app --reload</div>}
      {h && h.status === 'ok' && (
        <div className="svc muted small">
          OCR {h.ocr_engine} (fallback {h.ocr_fallback}) · LLM {h.llm_model} {h.llm_key_present ? '' : '(no API key – corrections disabled)'} ·
          {' '}{h.past_complaints} past complaints in memory
        </div>
      )}

      <main>
        {view.name === 'form' && <ComplaintForm onSubmitted={(id) => setView({ name: 'result', id })} />}
        {view.name === 'result' && <ComplaintResult id={view.id} onNew={() => setView({ name: 'form' })} />}
        {view.name === 'history' && (
          <div className="card">
            <h2>Submitted complaints</h2>
            <History onOpen={(id) => setView({ name: 'result', id })} />
          </div>
        )}
      </main>
    </div>
  )
}

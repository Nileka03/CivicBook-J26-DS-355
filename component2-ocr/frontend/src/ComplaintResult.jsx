import { useEffect, useState } from 'react'
import { fileUrl, forwardComplaint, getComplaint, reprocessComplaint } from './api'

const DECISION_TEXT = {
  ACCEPT: 'All corrections verified',
  PARTIAL: 'Some corrections reverted by the verifier',
  NO_CHANGES: 'No corrections needed',
  REVIEW: 'Needs human review',
  FALLBACK_RAW: 'LLM unavailable – raw OCR passed on',
  NOT_PROCESSED: 'Not processed',
}

function PageView({ page, showBoxes }) {
  return (
    <div className="page">
      <div className="page-img">
        <img src={fileUrl(page.image_uri)} alt={`Page ${page.page_no}`} />
        {showBoxes &&
          page.words.map((w) => {
            const [x0, y0, x1, y1] = w.bbox
            const changed = w.final_text !== w.text
            const cls = changed ? 'box fixed' : w.uncertain ? 'box unsure' : 'box ok'
            return (
              <span key={w.id} className={cls} title={`${w.text}${changed ? ' → ' + w.final_text : ''} (${Math.round(w.confidence * 100)}%)`}
                style={{
                  left: `${(x0 / page.width) * 100}%`, top: `${(y0 / page.height) * 100}%`,
                  width: `${((x1 - x0) / page.width) * 100}%`, height: `${((y1 - y0) / page.height) * 100}%`,
                }} />
            )
          })}
      </div>
      <div className="page-meta">
        <span className="chip">Page {page.page_no}</span>
        <span className="chip">{page.layout}</span>
        <span className="chip">skew {page.skew_deg}°</span>
        <span className="chip">{page.n_words} words · {page.n_uncertain} unsure</span>
        {page.mean_confidence != null && <span className="chip">conf {Math.round(page.mean_confidence * 100)}%</span>}
      </div>
    </div>
  )
}

function Transcript({ page }) {
  // highlight words that changed, on the final text
  const changed = page.words.filter((w) => w.final_text !== w.text).map((w) => w.final_text)
  let html = page.final_text.replace(/&/g, '&amp;').replace(/</g, '&lt;')
  for (const c of new Set(changed)) {
    const safe = c.replace(/&/g, '&amp;').replace(/</g, '&lt;')
    html = html.split(safe).join(`<mark>${safe}</mark>`)
  }
  return (
    <div className="transcripts">
      <div>
        <h4>OCR transcript</h4>
        <pre className="tx">{page.raw_text || '—'}</pre>
      </div>
      <div>
        <h4>Final transcript</h4>
        <pre className="tx" dangerouslySetInnerHTML={{ __html: html || '—' }} />
      </div>
    </div>
  )
}

export default function ComplaintResult({ id, onNew }) {
  const [pkg, setPkg] = useState(null)
  const [err, setErr] = useState('')
  const [boxes, setBoxes] = useState(true)
  const [fwd, setFwd] = useState(null)
  const [showJson, setShowJson] = useState(false)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let stop = false
    let timer
    async function poll() {
      try {
        const p = await getComplaint(id)
        if (stop) return
        setPkg(p)
        if (p.status === 'queued' || p.status === 'processing') timer = setTimeout(poll, 1200)
      } catch (e) {
        if (!stop) setErr(e.message)
      }
    }
    poll()
    return () => {
      stop = true
      clearTimeout(timer)
    }
  }, [id, tick])

  if (err) return <div className="card err">{err}</div>
  if (!pkg) return <div className="card">Loading…</div>

  const hw = pkg.handwritten
  const running = pkg.status === 'queued' || pkg.status === 'processing'
  const loc = pkg.location

  async function forward() {
    try {
      setFwd(await forwardComplaint(id))
    } catch (e) {
      setFwd({ error: e.message })
    }
  }

  async function rerun() {
    await reprocessComplaint(id)
    setFwd(null)
    setTick((t) => t + 1)
  }

  return (
    <div className="result">
      <div className="card head">
        <div>
          <div className="muted mono">{pkg.complaint_id}</div>
          <h2>Complaint received</h2>
          <div className="facts">
            <span><b>Date</b> {pkg.complaint_date}</span>
            <span>
              <b>Location</b>{' '}
              {loc.mode === 'gps'
                ? <a href={`https://maps.google.com/?q=${loc.latitude},${loc.longitude}`} target="_blank" rel="noreferrer">
                    {loc.latitude.toFixed(5)}, {loc.longitude.toFixed(5)}</a>
                : loc.address}
            </span>
          </div>
          {pkg.description && <p className="desc">{pkg.description}</p>}
        </div>
        <div className="head-actions">
          <span className={`status ${pkg.status}`}>{running ? 'Reading handwriting…' : pkg.status}</span>
          {hw && !running && (
            <span className={`decision ${hw.verifier_decision}`}>{DECISION_TEXT[hw.verifier_decision]}</span>
          )}
        </div>
      </div>

      {pkg.error && <div className="card err">Processing failed: {pkg.error}</div>}

      {pkg.evidence_images.length > 0 && (
        <div className="card">
          <h3>Evidence photos</h3>
          <div className="thumbs">
            {pkg.evidence_images.map((f) => (
              <a key={f.file_id} href={fileUrl(f.uri)} target="_blank" rel="noreferrer">
                <img src={fileUrl(f.uri)} alt={f.filename} />
              </a>
            ))}
          </div>
        </div>
      )}

      {hw && !running && (
        <>
          {hw.review_reasons.length > 0 && (
            <div className="card warn">
              <b>Notes:</b>
              <ul>{hw.review_reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
            </div>
          )}

          <div className="card">
            <div className="row-between">
              <h3>Handwritten complaint · {hw.pages.length} page{hw.pages.length === 1 ? '' : 's'}</h3>
              <label className="toggle">
                <input type="checkbox" checked={boxes} onChange={(e) => setBoxes(e.target.checked)} /> Word boxes
                <span className="legend"><i className="ok" /> sure <i className="unsure" /> unsure <i className="fixed" /> corrected</span>
              </label>
            </div>
            {hw.pages.map((p) => (
              <div key={p.page_no} className="page-row">
                <PageView page={p} showBoxes={boxes} />
                <Transcript page={p} />
              </div>
            ))}
          </div>

          <div className="card">
            <h3>Corrections checked by the verifier ({hw.edits.length})</h3>
            {hw.edits.length === 0 ? (
              <p className="muted">No corrections were proposed.</p>
            ) : (
              <table className="edits">
                <thead>
                  <tr><th>Word</th><th>OCR</th><th>Proposed</th><th>Evidence</th><th>Round</th><th>Result</th></tr>
                </thead>
                <tbody>
                  {hw.edits.map((e, i) => (
                    <tr key={i} className={e.accepted ? 'acc' : 'rej'}>
                      <td className="mono">{e.word_id}</td>
                      <td>{e.from_text}</td>
                      <td>{e.to_text}</td>
                      <td>{e.evidence}</td>
                      <td>{e.round}</td>
                      <td>{e.accepted ? 'kept' : `reverted · ${e.reasons.join(', ')}`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {hw.retrieved_context.length > 0 && (
              <>
                <h4>Similar past complaints used as reference</h4>
                <ul className="refs">
                  {hw.retrieved_context.map((r) => <li key={r.id}><span className="mono">{r.score}</span> {r.text}</li>)}
                </ul>
              </>
            )}
            <p className="muted small">
              OCR: {hw.ocr_engine || '—'} · LLM: {hw.llm_model || '—'} · rounds: {hw.llm_rounds} ·{' '}
              {Object.entries(hw.timings_ms).map(([k, v]) => `${k} ${v} ms`).join(' · ')}
            </p>
          </div>
        </>
      )}

      {!running && (
        <div className="card">
          <div className="row-between">
            <h3>Output package</h3>
            <div className="btns">
              {hw && <button className="btn ghost" onClick={rerun}>Re-process</button>}
              <button className="btn ghost" onClick={() => setShowJson((s) => !s)}>{showJson ? 'Hide' : 'Show'} JSON</button>
              <button className="btn primary" onClick={forward} disabled={pkg.status !== 'done'}>Send to verification (C3)</button>
            </div>
          </div>
          {fwd && (
            <div className={fwd.error ? 'err' : 'ok-box'}>
              {fwd.error ? fwd.error : `Sent to ${fwd.target} → HTTP ${fwd.status_code}: ${JSON.stringify(fwd.response)}`}
            </div>
          )}
          {showJson && <pre className="json">{JSON.stringify(pkg, null, 2)}</pre>}
        </div>
      )}

      <button className="btn ghost" onClick={onNew}>← New complaint</button>
    </div>
  )
}

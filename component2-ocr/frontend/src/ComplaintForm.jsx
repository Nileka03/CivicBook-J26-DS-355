import { useEffect, useMemo, useState } from 'react'
import { submitComplaint } from './api'

const today = () => new Date().toISOString().slice(0, 10)
const HW_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf']
const IMG_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const MAX_MB = 15

function useObjectUrls(files) {
  const urls = useMemo(() => files.map((f) => (f.type.startsWith('image/') ? URL.createObjectURL(f) : null)), [files])
  useEffect(() => () => urls.forEach((u) => u && URL.revokeObjectURL(u)), [urls])
  return urls
}

function checkFiles(list, allowed) {
  for (const f of list) {
    if (!allowed.includes(f.type)) return `${f.name}: file type not allowed`
    if (f.size > MAX_MB * 1024 * 1024) return `${f.name}: larger than ${MAX_MB} MB`
    if (f.size === 0) return `${f.name}: file is empty`
  }
  return null
}

export default function ComplaintForm({ onSubmitted }) {
  const [date, setDate] = useState(today())
  const [locMode, setLocMode] = useState('gps')
  const [gps, setGps] = useState(null)
  const [gpsState, setGpsState] = useState('idle')
  const [address, setAddress] = useState('')
  const [description, setDescription] = useState('')
  const [pages, setPages] = useState([])
  const [evidence, setEvidence] = useState([])
  const [layoutHint, setLayoutHint] = useState('auto')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const pageUrls = useObjectUrls(pages)
  const evidenceUrls = useObjectUrls(evidence)

  function locate() {
    if (!navigator.geolocation) {
      setGpsState('unsupported')
      return
    }
    setGpsState('locating')
    navigator.geolocation.getCurrentPosition(
      (p) => {
        setGps({ lat: p.coords.latitude, lng: p.coords.longitude, acc: p.coords.accuracy })
        setGpsState('ok')
      },
      (e) => setGpsState(`error: ${e.message}`),
      { enableHighAccuracy: true, timeout: 15000 },
    )
  }

  function addFiles(e, setter, allowed) {
    const list = [...e.target.files]
    e.target.value = ''
    const problem = checkFiles(list, allowed)
    if (problem) return setError(problem)
    setError('')
    setter((prev) => [...prev, ...list])
  }

  const move = (i, d) =>
    setPages((p) => {
      const n = [...p]
      const j = i + d
      if (j < 0 || j >= n.length) return p
      ;[n[i], n[j]] = [n[j], n[i]]
      return n
    })

  async function submit(e) {
    e.preventDefault()
    setError('')
    if (locMode === 'gps' && !gps) return setError('Get your live location, or switch to typing the address.')
    if (locMode === 'typed' && !address.trim()) return setError('Type the address of the problem.')
    if (!description.trim() && pages.length === 0)
      return setError('Add the handwritten complaint pages, or type a description.')

    const fd = new FormData()
    fd.append('complaint_date', date)
    fd.append('location_mode', locMode)
    if (locMode === 'gps') {
      fd.append('latitude', gps.lat)
      fd.append('longitude', gps.lng)
      fd.append('accuracy_m', gps.acc)
    } else fd.append('address', address.trim())
    if (description.trim()) fd.append('description', description.trim())
    fd.append('layout_hint', layoutHint)
    pages.forEach((f) => fd.append('handwritten_pages', f))
    evidence.forEach((f) => fd.append('evidence_images', f))

    setBusy(true)
    try {
      const res = await submitComplaint(fd)
      onSubmitted(res.complaint_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="card form" onSubmit={submit}>
      <h2>Submit a written complaint</h2>
      <p className="muted">One complaint per submission. If your letter has several pages, add them all below.</p>

      <div className="grid2">
        <label className="field">
          <span>Complaint date</span>
          <input type="date" value={date} max={today()} onChange={(e) => setDate(e.target.value)} required />
        </label>
        <div className="field">
          <span>Location of the problem</span>
          <div className="seg">
            <button type="button" className={locMode === 'gps' ? 'on' : ''} onClick={() => setLocMode('gps')}>
              Live location
            </button>
            <button type="button" className={locMode === 'typed' ? 'on' : ''} onClick={() => setLocMode('typed')}>
              Type address
            </button>
          </div>
        </div>
      </div>

      {locMode === 'gps' ? (
        <div className="locbox">
          <button type="button" className="btn ghost" onClick={locate} disabled={gpsState === 'locating'}>
            {gpsState === 'locating' ? 'Locating…' : gps ? 'Update location' : 'Use my current location'}
          </button>
          {gps && (
            <span className="mono">
              {gps.lat.toFixed(6)}, {gps.lng.toFixed(6)} · ±{Math.round(gps.acc)} m
            </span>
          )}
          {gpsState.startsWith('error') && <span className="err-inline">{gpsState}</span>}
        </div>
      ) : (
        <label className="field">
          <span>Address</span>
          <textarea rows={2} value={address} onChange={(e) => setAddress(e.target.value)}
            placeholder="e.g. 57/2, Malabe Road, Kaduwela" />
        </label>
      )}

      <label className="field">
        <span>Anything else about the complaint <em>(optional)</em></span>
        <textarea rows={3} value={description} onChange={(e) => setDescription(e.target.value)}
          placeholder="Type details in Sinhala or English. This also helps us read your handwriting." />
      </label>

      <section className="drop">
        <div className="drop-head">
          <div>
            <h3>Handwritten complaint</h3>
            <p className="muted">Photos or scans (JPG, PNG, PDF). Pages are read in this order.</p>
          </div>
          <label className="btn">
            Add pages
            <input type="file" multiple accept={HW_TYPES.join(',')} hidden
              onChange={(e) => addFiles(e, setPages, HW_TYPES)} />
          </label>
        </div>
        {pages.length > 0 && (
          <ol className="pagelist">
            {pages.map((f, i) => (
              <li key={`${f.name}-${i}`}>
                {pageUrls[i] ? <img src={pageUrls[i]} alt="" /> : <div className="pdf">PDF</div>}
                <span className="name">Page {i + 1} · {f.name}</span>
                <span className="actions">
                  <button type="button" onClick={() => move(i, -1)} disabled={i === 0} aria-label="Move up">↑</button>
                  <button type="button" onClick={() => move(i, 1)} disabled={i === pages.length - 1} aria-label="Move down">↓</button>
                  <button type="button" onClick={() => setPages((p) => p.filter((_, k) => k !== i))} aria-label="Remove">✕</button>
                </span>
              </li>
            ))}
          </ol>
        )}
        <label className="field inline">
          <span>Page layout</span>
          <select value={layoutHint} onChange={(e) => setLayoutHint(e.target.value)}>
            <option value="auto">Detect automatically</option>
            <option value="letter">Letter (read line by line)</option>
            <option value="table">Table / form (bracket each cell)</option>
          </select>
        </label>
      </section>

      <section className="drop">
        <div className="drop-head">
          <div>
            <h3>Evidence photos <em>(optional)</em></h3>
            <p className="muted">Pictures of the problem itself, not the letter.</p>
          </div>
          <label className="btn ghost">
            Add photos
            <input type="file" multiple accept={IMG_TYPES.join(',')} hidden
              onChange={(e) => addFiles(e, setEvidence, IMG_TYPES)} />
          </label>
        </div>
        {evidence.length > 0 && (
          <div className="thumbs">
            {evidence.map((f, i) => (
              <figure key={`${f.name}-${i}`}>
                <img src={evidenceUrls[i]} alt={f.name} />
                <button type="button" onClick={() => setEvidence((p) => p.filter((_, k) => k !== i))}
                  aria-label="Remove">✕</button>
              </figure>
            ))}
          </div>
        )}
      </section>

      {error && <div className="err">{error}</div>}
      <button className="btn primary big" disabled={busy}>
        {busy ? 'Uploading…' : 'Submit complaint'}
      </button>
    </form>
  )
}

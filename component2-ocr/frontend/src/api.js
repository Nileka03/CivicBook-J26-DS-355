const BASE = import.meta.env.VITE_API_BASE || ''

async function handle(res) {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* not JSON */ }
    throw new Error(msg)
  }
  return res.json()
}

export const fileUrl = (uri) => `${BASE}${uri}`

export function submitComplaint(formData) {
  return fetch(`${BASE}/api/v1/complaints`, { method: 'POST', body: formData }).then(handle)
}

export function getComplaint(id) {
  return fetch(`${BASE}/api/v1/complaints/${id}`).then(handle)
}

export function listComplaints() {
  return fetch(`${BASE}/api/v1/complaints`).then(handle)
}

export function forwardComplaint(id) {
  return fetch(`${BASE}/api/v1/complaints/${id}/forward`, { method: 'POST' }).then(handle)
}

export function reprocessComplaint(id) {
  return fetch(`${BASE}/api/v1/complaints/${id}/reprocess`, { method: 'POST' }).then(handle)
}

export function health() {
  return fetch(`${BASE}/api/v1/health`).then(handle)
}

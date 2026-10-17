// CivicBook Component 4 – routing demo front end (no build step, plain JavaScript)

const API = "/api/v1";
let CONFIG = null;
let COMPLAINTS = [];

const EXAMPLES = [
  "මගේ විදුලි බිල මේ මාසේ සාමාන්‍යයට වඩා ගොඩක් වැඩියි. ඒ වගේම අපේ පාරේ කානු අවහිර වෙලා වැස්සට වතුර ගෙවල් ඇතුළට එනවා.",
  "My electricity bill for this month is much higher than usual. Also the name on my account is still the previous owner's name, please change it.",
  "There was no power in our area since this morning. In addition, garbage has not been collected for two weeks and mosquitoes are breeding near the drain. Furthermore my bill was not issued after the flood.",
  "අපේ ගෙදර ළඟ අනවසර ඉදිකිරීමක් කරනවා. ඊට අමතරව කසළ එකතු නොකරන නිසා මදුරුවන් බෝවෙනවා.",
];

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (v == null ? "–" : (v * 100).toFixed(1) + "%");
const partClass = (i) => "p" + ((i % 4) + 1);

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" }, ...opts,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText);
  return body;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2200);
}

function statusBadge(s) {
  if (s.officer_status === "discarded as fake") return `<span class="badge bad">Discarded as fake</span>`;
  if (s.officer_status === "forwarded by officer") return `<span class="badge ok">Forwarded by officer</span>`;
  if (s.status === "routed to department") return `<span class="badge ok">Routed to department</span>`;
  if (s.status === "routed to department (flagged)") return `<span class="badge warn">Routed · department needs verification</span>`;
  return `<span class="badge warn">Held at institution for officer review</span>`;
}

function flagBadges(s) {
  let b = "";
  if (s.is_fake) b += `<span class="badge bad">is_fake = true</span>`;
  if (s.needs_verification) b += `<span class="badge warn">needs_verification: ${esc(s.needs_verification)}</span>`;
  return b;
}

// Complaint text with every sub-complaint highlighted (uses the processed text offsets).
function highlightAll(text, subs) {
  const sorted = subs.map((s, i) => ({ ...s, idx: i })).sort((a, b) => a.char_start - b.char_start);
  let out = "", pos = 0;
  for (const s of sorted) {
    if (s.char_start > pos) out += `<span class="connector">${esc(text.slice(pos, s.char_start))}</span>`;
    out += `<span class="seg ${partClass(s.idx)}"><sup>${s.idx + 1}</sup>${esc(text.slice(s.char_start, s.char_end))}</span>`;
    pos = s.char_end;
  }
  if (pos < text.length) out += `<span class="connector">${esc(text.slice(pos))}</span>`;
  return out;
}

// Complaint text with only one part highlighted (officer view).
function highlightOne(text, s) {
  return `<span class="dim">${esc(text.slice(0, s.char_start))}</span>` +
    `<span class="hl">${esc(text.slice(s.char_start, s.char_end))}</span>` +
    `<span class="dim">${esc(text.slice(s.char_end))}</span>`;
}

function destination(s, useCurrent = false) {
  const inst = useCurrent ? s.current_institution : s.institution;
  const dept = useCurrent ? s.current_department : s.department;
  return `<span class="inst ${esc(inst)}">${esc(inst)}</span>` +
    (dept ? ` <span class="arrow">›</span> <span class="dest">${esc(dept)}</span>`
          : ` <span class="arrow">›</span> <span class="muted">institution review</span>`);
}

// ------------------------------------------------------------------ tabs
function showView(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  if (name === "LECO" || name === "MC") loadInbox(name);
  if (name === "pipeline") renderPipeline();
}

// ------------------------------------------------------------------ citizen
function renderCitizenResult(rec) {
  const box = $("#citizen-result");
  const routes = rec.sub_complaints.map((s, i) => {
    let explain = "";
    if (s.status === "stopped at institution" && s.is_fake && s.needs_verification) {
      explain = "Marked as possibly fake and the institution is uncertain: an officer will check it.";
    } else if (s.status === "stopped at institution" && s.is_fake) {
      explain = "Marked as possibly fake: an officer at the institution verifies it before it goes to a department.";
    } else if (s.status === "stopped at institution") {
      explain = "The system is not confident about the institution: an officer will confirm it.";
    } else if (s.needs_verification === "department") {
      explain = "Sent to the most likely department; an officer will confirm the department.";
    }
    return `<div class="route">
      <div class="route-head"><span class="part ${partClass(i)}">Part ${i + 1}</span>${destination(s)}${statusBadge(s)}</div>
      <div class="route-text">${esc(s.sub_complaint_text)}</div>
      ${explain ? `<div class="meta">${explain}</div>` : ""}
    </div>`;
  }).join("");
  box.innerHTML = `
    <div class="panel-head"><h2>Where your complaint goes</h2>
      <span class="mono muted">${esc(rec.complaint_id)}</span></div>
    <p class="muted small">Split into <b>${rec.num_sub_complaints}</b> part${rec.num_sub_complaints === 1 ? "" : "s"}
      in ${rec.processing_ms} ms ${rec.is_fake ? '· <span class="badge bad">is_fake = true</span>' : ""}</p>
    <div class="complaint-text">${highlightAll(rec.processed_text, rec.sub_complaints)}</div>
    ${rec.segmentation_fallback ? '<p class="error">The model found no clear sub-complaint; the whole text was routed as one.</p>' : ""}
    <div class="routes">${routes}</div>
    <p style="margin-top:12px"><button class="ghost small" data-open-pipeline="${esc(rec.complaint_id)}">See the pipeline steps</button></p>`;
}

function renderHistory() {
  const el = $("#history");
  if (!COMPLAINTS.length) { el.innerHTML = `<div class="empty">No complaints yet.</div>`; return; }
  el.innerHTML = `<div class="hist">` + COMPLAINTS.map((c) => `
    <div class="hist-row">
      <span class="mono">${esc(c.complaint_id)}</span>
      <span class="hist-text">${esc(c.raw_complaint_text)}</span>
      <span>${c.sub_complaints.map((s) => `<span class="inst ${esc(s.current_institution)}">${esc(s.current_institution)}</span>`).join(" ")}
        <button class="ghost small" data-open-pipeline="${esc(c.complaint_id)}">Pipeline</button></span>
    </div>`).join("") + `</div>`;
}

async function submitComplaint() {
  const text = $("#complaint").value.trim();
  $("#submit-error").textContent = "";
  if (!text) { $("#submit-error").textContent = "Please type a complaint."; return; }
  const btn = $("#submit");
  btn.disabled = true; btn.textContent = "Routing…";
  try {
    const body = {
      complaint_text: text,
      is_fake: $("#is-fake").checked,
      threshold_institution: parseFloat($("#t-inst").value),
      threshold_department: parseFloat($("#t-dept").value),
    };
    const rec = await api("/route", { method: "POST", body: JSON.stringify(body) });
    await refreshComplaints();
    renderCitizenResult(COMPLAINTS.find((c) => c.complaint_id === rec.complaint_id) || rec);
    $("#pipeline-select").value = rec.complaint_id;
  } catch (e) {
    $("#submit-error").textContent = "Could not route the complaint: " + e.message;
  } finally {
    btn.disabled = false; btn.textContent = "Route complaint";
  }
}

async function refreshComplaints() {
  COMPLAINTS = await api("/complaints");
  renderHistory();
  const sel = $("#pipeline-select");
  const current = sel.value;
  sel.innerHTML = COMPLAINTS.map((c) =>
    `<option value="${esc(c.complaint_id)}">${esc(c.complaint_id)} – ${esc(c.raw_complaint_text.slice(0, 50))}</option>`).join("");
  if (current && COMPLAINTS.some((c) => c.complaint_id === current)) sel.value = current;
  for (const inst of CONFIG.institutions) {
    const n = COMPLAINTS.flatMap((c) => c.sub_complaints)
      .filter((s) => s.current_institution === inst && s.officer_status === "awaiting institution review").length;
    $("#count-" + inst).textContent = n || "";
  }
}

// ------------------------------------------------------------------ pipeline
function bars(scores, t, chosen) {
  return `<div class="bars">` + Object.entries(scores).map(([k, v]) => `
    <div class="bar-row"><span>${esc(k)}</span>
      <div class="bar"><div class="bar-fill ${k === chosen ? "chosen" : ""}" style="width:${(v * 100).toFixed(1)}%"></div>
        <div class="bar-t" style="left:${(t * 100).toFixed(1)}%" title="threshold t = ${t}"></div></div>
      <span class="mono">${v.toFixed(3)}</span></div>`).join("") + `</div>`;
}

function caseTable(hit) {
  const cell = (n) => `<td class="${n === hit ? "hit" : ""}">Case ${n}</td>`;
  return `<table class="cases"><tr><th>Score condition</th><th>is_fake = false</th><th>is_fake = true</th></tr>
    <tr><th>All ≥ t</th>${cell(1)}${cell(2)}</tr>
    <tr><th>One ≥ t</th>${cell(3)}${cell(4)}</tr>
    <tr><th>All &lt; t</th>${cell(5)}${cell(6)}</tr></table>`;
}

function renderPipeline() {
  const id = $("#pipeline-select").value;
  const rec = COMPLAINTS.find((c) => c.complaint_id === id);
  const el = $("#pipeline");
  if (!rec) { el.innerHTML = `<div class="empty">Submit a complaint first.</div>`; return; }

  // which sub-complaint each token belongs to
  const owner = new Array(rec.tokens.length).fill(-1);
  rec.sub_complaints.forEach((s, i) => { for (let k = s.token_start; k < s.token_end; k++) owner[k] = i; });
  const tokens = rec.tokens.map((t, k) => `<span class="tok ${rec.bio_tags[k]} ${owner[k] >= 0 && rec.bio_tags[k] !== "O" ? "" : ""}">
      ${esc(t)}<b>${rec.bio_tags[k]}</b></span>`).join("");

  const gates = rec.sub_complaints.map((s, i) => `
    <div class="gate">
      <div class="route-head"><span class="part ${partClass(i)}">Part ${i + 1}</span>
        <span class="route-text">${esc(s.sub_complaint_text)}</span></div>
      <h3 style="margin-top:10px">Gate 1 · institution (t = ${rec.thresholds.institution})</h3>
      ${bars(s.institution_scores, rec.thresholds.institution, s.institution)}
      ${caseTable(s.gate1_case)}
      <p class="meta">${esc(s.gate1_explanation)}</p>
      ${s.department_scores ? `
        <h3>Gate 2 · department of ${esc(s.institution)} (t = ${rec.thresholds.department})</h3>
        ${bars(s.department_scores, rec.thresholds.department, s.department)}
        <p class="meta">${esc(s.gate2_explanation)}</p>`
      : `<p class="meta"><b>Gate 2 not reached</b> — the sub-complaint stops at ${esc(s.institution)} and is not forwarded to the department head.</p>`}
      <div class="route-head">Result: ${destination(s)} ${statusBadge(s)} ${flagBadges(s)}</div>
    </div>`).join("");

  const output = {
    complaint_id: rec.complaint_id, raw_complaint_text: rec.raw_complaint_text, is_fake: rec.is_fake,
    num_sub_complaints: rec.num_sub_complaints,
    sub_complaints: rec.sub_complaints.map((s) => ({
      sub_complaint_id: s.sub_complaint_id, sub_complaint_text: s.sub_complaint_text,
      institution: s.institution, institution_scores: s.institution_scores,
      department: s.department, department_scores: s.department_scores,
      needs_verification: s.needs_verification, status: s.status })),
  };

  el.innerHTML = `
    <div class="step"><div class="step-num">1</div><div>
      <h3>Input from Component 3</h3>
      <div class="complaint-text">${esc(rec.raw_complaint_text)}</div>
      <p class="meta">is_fake = <b>${rec.is_fake}</b></p></div></div>
    <div class="step"><div class="step-num">2</div><div>
      <h3>Light pre-processing</h3>
      <p class="meta">Unicode normalisation; phone/account numbers → &lt;NUMBER&gt;, emails → &lt;EMAIL&gt;. Word order, punctuation and code-mixing are kept.</p>
      <div class="complaint-text">${esc(rec.processed_text)}</div></div></div>
    <div class="step"><div class="step-num">3</div><div>
      <h3>Tokens and BIO tags (XLM-R)</h3>
      <div class="legend">B = start of a sub-complaint · I = inside · O = connector, not part of any sub-complaint</div>
      <div class="tokens">${tokens}</div></div></div>
    <div class="step"><div class="step-num">4</div><div>
      <h3>Segmentation into ${rec.num_sub_complaints} sub-complaint${rec.num_sub_complaints === 1 ? "" : "s"}</h3>
      <div class="complaint-text">${highlightAll(rec.processed_text, rec.sub_complaints)}</div></div></div>
    <div class="step"><div class="step-num">5</div><div>
      <h3>Chained classification with confidence gates</h3>${gates}</div></div>
    <div class="step"><div class="step-num">6</div><div>
      <h3>Output record (sent to the CivicBook backend) · ${rec.processing_ms} ms</h3>
      <pre class="json">${esc(JSON.stringify(output, null, 2))}</pre></div></div>`;
}

// ------------------------------------------------------------------ officer inbox
function actionsFor(s) {
  const depts = CONFIG.departments[s.current_institution] || [];
  const other = CONFIG.institutions.filter((i) => i !== s.current_institution);
  const deptSelect = `<select data-dept>${depts.map((d) => `<option ${d === s.current_department ? "selected" : ""}>${esc(d)}</option>`).join("")}</select>`;
  const transfer = other.map((o) => `<button class="btn-plain" data-act="transfer" data-inst="${esc(o)}">Transfer to ${esc(o)}</button>`).join("");
  const id = esc(s.sub_complaint_id);

  if (s.officer_status === "discarded as fake") return "";
  if (s.officer_status === "awaiting institution review") {
    if (s.is_fake) {
      return `<div class="actions" data-id="${id}">
        <span class="small">Verified real →</span>${deptSelect}
        <button class="btn-ok" data-act="forward">Forward</button>
        <button class="btn-bad" data-act="discard">Discard as fake</button>${transfer}</div>`;
    }
    return `<div class="actions" data-id="${id}">
      <span class="small">Confirm institution →</span>${deptSelect}
      <button class="btn-ok" data-act="forward">Forward</button>${transfer}</div>`;
  }
  if (s.officer_status === "in department (needs verification)") {
    return `<div class="actions" data-id="${id}">
      <button class="btn-ok" data-act="confirm_department">Confirm ${esc(s.current_department)}</button>
      <span class="small">or move to</span>${deptSelect}
      <button class="btn-plain" data-act="reassign_department">Move</button></div>`;
  }
  return `<div class="actions" data-id="${id}"><span class="small">Reassign →</span>${deptSelect}
    <button class="btn-plain" data-act="reassign_department">Move</button></div>`;
}

function card(s) {
  const reason = s.officer_status === "awaiting institution review"
    ? (s.is_fake ? "Possibly fake (Component 3). Verify before forwarding." :
       s.current_institution !== s.institution ? `Transferred from ${s.institution}.` :
       "Institution confidence below threshold. Confirm or transfer.")
    : "";
  return `<div class="card">
    <div class="badges"><span class="mono">${esc(s.sub_complaint_id)}</span>${statusBadge(s)}${flagBadges(s)}</div>
    <div><div class="small muted">Sub-complaint highlighted in the full complaint</div>
      <div class="complaint-text">${highlightOne(s.processed_text, s)}</div></div>
    ${reason ? `<div class="meta"><b>${esc(reason)}</b></div>` : ""}
    <div class="meta">Institution scores: ${Object.entries(s.institution_scores).map(([k, v]) => `${esc(k)} ${pct(v)}`).join(" · ")}
      ${s.department_scores && s.department ? `<br>Department score (${esc(s.department)}): ${pct(s.department_scores[s.department])}` : ""}</div>
    ${s.other_parts.length ? `<div class="meta">Other parts of this complaint: ${s.other_parts.map((o) =>
      `${destination({ current_institution: o.institution, current_department: o.department }, true)}`).join(", ")}</div>` : ""}
    <details class="more"><summary>Original text as submitted · history</summary>
      <div class="complaint-text" style="margin-top:6px">${esc(s.raw_complaint_text)}</div>
      <ul class="history">${s.history.map((h) => `<li>${esc(h.time)} — ${esc(h.event)}</li>`).join("")}</ul>
    </details>
    ${actionsFor(s)}
  </div>`;
}

async function loadInbox(inst) {
  const el = $("#inbox-" + inst);
  let items;
  try { items = await api("/inbox/" + inst); }
  catch (e) { el.innerHTML = `<div class="error">${esc(e.message)}</div>`; return; }

  const review = items.filter((s) => s.officer_status === "awaiting institution review");
  const discarded = items.filter((s) => s.officer_status === "discarded as fake");
  const inDept = (d) => items.filter((s) => s.current_department === d &&
    !["awaiting institution review", "discarded as fake"].includes(s.officer_status));
  const depts = CONFIG.departments[inst];

  const col = (title, list, cls = "") => `<div class="col ${cls}">
    <h3><span>${esc(title)}</span><span class="badge grey">${list.length}</span></h3>
    ${list.length ? list.map(card).join("") : `<div class="muted small">Nothing here.</div>`}</div>`;

  el.innerHTML = `
    <div class="stats">
      <div class="stat"><b>${items.length}</b><span class="small muted">sub-complaints</span></div>
      <div class="stat"><b>${review.length}</b><span class="small muted">awaiting institution review</span></div>
      <div class="stat"><b>${items.filter((s) => s.is_fake).length}</b><span class="small muted">marked is_fake</span></div>
      <div class="stat"><b>${items.filter((s) => s.needs_verification).length}</b><span class="small muted">needs_verification</span></div>
    </div>
    <div class="board">
      ${col("Institution review", review, "review")}
      ${depts.map((d) => col(d, inDept(d))).join("")}
      ${col("Discarded", discarded)}
    </div>`;
}

async function doAction(btn) {
  const box = btn.closest(".actions");
  const body = { action: btn.dataset.act };
  const sel = box.querySelector("[data-dept]");
  if (sel) body.department = sel.value;
  if (btn.dataset.inst) body.institution = btn.dataset.inst;
  try {
    await api(`/subcomplaints/${encodeURIComponent(box.dataset.id)}/action`, { method: "POST", body: JSON.stringify(body) });
    toast("Updated");
    await refreshComplaints();
    const view = $(".tab.active").dataset.view;
    if (view === "LECO" || view === "MC") loadInbox(view);
  } catch (e) { toast("Error: " + e.message); }
}

// ------------------------------------------------------------------ start
document.addEventListener("click", (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  if (t.classList.contains("tab")) showView(t.dataset.view);
  else if (t.dataset.example) $("#complaint").value = EXAMPLES[+t.dataset.example];
  else if (t.dataset.openPipeline) { $("#pipeline-select").value = t.dataset.openPipeline; showView("pipeline"); }
  else if (t.dataset.act) doAction(t);
  else if (t.hasAttribute("data-refresh")) { refreshComplaints().then(() => loadInbox($(".tab.active").dataset.view)); }
});

$("#submit").addEventListener("click", submitComplaint);
$("#is-fake").addEventListener("change", (e) => { $("#fake-text").textContent = "is_fake = " + e.target.checked; });
$("#pipeline-select").addEventListener("change", renderPipeline);
$("#reset").addEventListener("click", async () => {
  if (!confirm("Delete all demo complaints?")) return;
  await api("/complaints", { method: "DELETE" });
  await refreshComplaints();
  $("#citizen-result").innerHTML = `<h2>Where your complaint goes</h2><div class="empty">Submit a complaint to see how it is split and routed.</div>`;
});

(async function init() {
  try {
    CONFIG = await api("/config");
    $("#t-inst").value = CONFIG.thresholds.institution;
    $("#t-dept").value = CONFIG.thresholds.department;
    await refreshComplaints();
  } catch (e) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="panel error">Cannot reach the routing service: ${esc(e.message)}</div>`);
  }
})();

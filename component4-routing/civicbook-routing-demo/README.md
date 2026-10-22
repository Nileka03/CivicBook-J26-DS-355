# CivicBook – Component 4 Routing Demo

A local web demo of the Multi-Department Complaint Routing Engine:
citizen complaint → XLM-R BIO segmentation → institution gate (with `is_fake`) → department gate → LECO / MC officer inboxes.

## File structure

```
civicbook-routing-demo/
├── app/
│   ├── __init__.py
│   ├── model.py        # RoutingModel (same layers as the Kaggle notebook) + loader
│   ├── pipeline.py     # pre-processing, tokenising, BIO → spans, Gate 1 / Gate 2, RoutingEngine
│   ├── store.py        # saves complaints to data/complaints.json, officer actions
│   └── main.py         # FastAPI: API endpoints + serves the web page
├── static/
│   ├── index.html      # 4 views: Citizen, Pipeline, LECO officer, MC officer
│   ├── styles.css
│   └── app.js
├── tests/
│   └── test_gates.py   # checks the 6 gate cases
├── model/              # ← put your Kaggle model files here
├── data/               # created automatically (demo complaints)
├── requirements.txt
└── README.md
```

## Setup in VS Code (Windows)

1. **Install Python 3.11** (3.10–3.12 work). Tick "Add Python to PATH".
2. Open the `civicbook-routing-demo` folder in VS Code (*File → Open Folder*).
3. Open a terminal (*Terminal → New Terminal*) and run:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

   If PowerShell blocks `activate`, run once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
   (macOS / Linux: `source .venv/bin/activate`.)

   In VS Code press *Ctrl+Shift+P → Python: Select Interpreter* and choose `.venv`.

4. **Copy the model.** From Kaggle, Version #1 (xlm-roberta-base) → *Output* → download `outputs/model/`.
   Put these files directly in `model/`:
   - `best_model.pt`
   - `routing_config.json`
   - `tokenizer.json`, `tokenizer_config.json` (plus `sentencepiece.bpe.model`, `special_tokens_map.json` if present)

5. **Test the gate logic** (optional):

   ```powershell
   python -m pytest tests
   ```

6. **Run:**

   ```powershell
   uvicorn app.main:app --port 8000
   ```

   Wait for `Application startup complete`, then open <http://127.0.0.1:8000>.
   The first start needs internet once (downloads the small xlm-roberta-base config; there is a built-in fallback if offline).
   Stop with *Ctrl+C*.

API docs (Swagger) are at <http://127.0.0.1:8000/docs>.

## Views

| Tab | Who | What it shows |
|---|---|---|
| **Citizen** | citizen | Type a complaint, set the simulated Component 3 `is_fake` flag, see each part highlighted and where it goes |
| **Pipeline** | examiner | Step by step: input → pre-processing → tokens + BIO tags → sub-complaints → Gate 1 bars + 6-case table → Gate 2 bars → output JSON |
| **LECO officer** | LECO staff | Board: *Institution review* column, one column per department, *Discarded*. Each card shows the sub-complaint highlighted inside the full complaint |
| **MC officer** | MC staff | Same for Engineering / Health |

## PP1 demo script (about 5 minutes)

1. **Normal multi-topic complaint (Case 1/3).** Citizen tab → example *Sinhala · LECO + MC*, `is_fake = false` → *Route complaint*.
   Show the parts going to LECO › department and MC › department.
2. **Pipeline.** Click *See the pipeline steps*. Walk through BIO tags, the split, Gate 1 bars against the threshold line, the highlighted case in the table, then Gate 2.
3. **Officer side.** Open *LECO officer* and *MC officer*: each sees only its own part, with the full complaint for context and where the other parts went.
4. **Fake complaint (Case 2/4).** Back to Citizen, same text, switch `is_fake = true` → route.
   Each part stops at the institution (no department). In the officer tab it sits in *Institution review* with **Verified real → Forward** or **Discard as fake**.
5. **Low institution confidence (Case 5/6).** Open *Confidence thresholds*, set institution t = 0.99 → route.
   Parts stop with `needs_verification: institution`; the officer can **Confirm institution → Forward** or **Transfer** to the other institution.
6. **Low department confidence.** Set department t = 0.85 (suggested operating point) → route.
   Parts still go to the top department but are flagged `needs_verification: department`; the officer can **Confirm** or **Move**.
7. *Clear demo data* resets everything.

### Gate rules used

| Institution scores | `is_fake = false` | `is_fake = true` |
|---|---|---|
| all ≥ t | Case 1 → highest institution, continue to department | Case 2 → stop at institution |
| one ≥ t | Case 3 → that institution, continue | Case 4 → stop at institution |
| all < t | Case 5 → highest institution, stop, needs_verification = institution | Case 6 → same + fake |

Department gate: highest department ≥ t; if none, highest anyway with `needs_verification = department`.
Thresholds default to the values in `routing_config.json`.

## API (for the CivicBook backend)

`POST /api/v1/route`

```json
{ "complaint_id": "CB-001", "complaint_text": "...", "is_fake": false }
```

Returns `complaint_id, raw_complaint_text, is_fake, num_sub_complaints, sub_complaints[...]` with
`sub_complaint_id, sub_complaint_text, institution, institution_scores, department, department_scores, needs_verification, status`.
Scores are independent sigmoids (they do not add up to 1).

Other endpoints: `GET /api/v1/complaints`, `GET /api/v1/inbox/{LECO|MC}`, `POST /api/v1/subcomplaints/{id}/action`, `DELETE /api/v1/complaints`.

## Data privacy

`data/complaints.json` stores whatever you type. Use invented examples for demos and screenshots; do not paste real LECO complaints into a shared screen or commit `data/` or `model/` to GitHub (they are in `.gitignore`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `best_model.pt not found` | Files must be directly inside `model/`, not `model/model/` |
| `Missing keys` / `Unexpected keys` | Wrong weights file — use Version #1 (base). For the large model, internet is needed for its config |
| `ModuleNotFoundError: app` | Run uvicorn from the project folder, not from inside `app/` |
| Port in use | `uvicorn app.main:app --port 8001` |
| Slow first request | Normal; the model warms up at start-up. CPU routing takes ~100–300 ms per complaint |

"""
CivicBook Component 4 - routing demo service.

Run from the project folder:
    uvicorn app.main:app --port 8000
then open http://127.0.0.1:8000
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import model as model_loader
from .pipeline import RoutingEngine
from .store import Store

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("C4_MODEL_DIR", ROOT / "model"))
DATA_FILE = Path(os.environ.get("C4_DATA_FILE", ROOT / "data" / "complaints.json"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

state = {}


@asynccontextmanager
async def lifespan(app):
    model, tokenizer, cfg = model_loader.load(MODEL_DIR)
    state["engine"] = RoutingEngine(model, tokenizer, cfg)
    state["store"] = Store(DATA_FILE)
    state["engine"].route("warm up", False)            # first call is slow; do it now
    yield


app = FastAPI(title="CivicBook Component 4 - Complaint Routing", lifespan=lifespan)


# ------------------------------------------------------------------ request models
class RouteRequest(BaseModel):
    complaint_id: Optional[str] = None
    complaint_text: str = Field(min_length=1)
    is_fake: bool
    # demo only: override the thresholds from routing_config.json
    threshold_institution: Optional[float] = Field(None, ge=0, le=1)
    threshold_department: Optional[float] = Field(None, ge=0, le=1)


class ActionRequest(BaseModel):
    action: str          # forward | reassign_department | confirm_department | transfer | discard
    department: Optional[str] = None
    institution: Optional[str] = None


# ------------------------------------------------------------------ API
@app.get("/api/v1/config")
def config():
    e = state["engine"]
    return {"model": e.cfg["model_name"], "institutions": e.institutions,
            "departments": e.departments,
            "thresholds": {"institution": e.t_inst, "department": e.t_dept}}


@app.post("/api/v1/route")
def route(req: RouteRequest):
    if not req.complaint_text.strip():
        raise HTTPException(422, "complaint_text is empty")
    if req.complaint_id and state["store"].get(req.complaint_id):
        raise HTTPException(409, f"complaint_id {req.complaint_id} already exists")
    try:
        record = state["engine"].route(req.complaint_text, req.is_fake, req.complaint_id,
                                       req.threshold_institution, req.threshold_department)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return state["store"].add(record)


@app.get("/api/v1/complaints")
def complaints():
    return state["store"].all()


@app.get("/api/v1/complaints/{complaint_id}")
def complaint(complaint_id: str):
    c = state["store"].get(complaint_id)
    if not c:
        raise HTTPException(404, "not found")
    return c


@app.get("/api/v1/inbox/{institution}")
def inbox(institution: str):
    if institution not in state["engine"].institutions:
        raise HTTPException(404, f"Unknown institution {institution}")
    return state["store"].inbox(institution)


@app.post("/api/v1/subcomplaints/{sub_id}/action")
def action(sub_id: str, req: ActionRequest):
    try:
        return state["store"].act(sub_id, req.action, state["engine"].departments,
                                  req.department, req.institution)
    except KeyError:
        raise HTTPException(404, "sub-complaint not found")
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.delete("/api/v1/complaints")
def reset():
    state["store"].reset()
    return {"ok": True}


# ------------------------------------------------------------------ web page
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

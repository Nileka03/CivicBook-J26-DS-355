"""
Demo storage: keeps routed complaints in data/complaints.json so the officer inboxes
survive a restart. (In the full CivicBook system this is the backend's database.)
"""

import json
import threading
import time
from pathlib import Path

# officer_status values
AWAITING_REVIEW = "awaiting institution review"   # stopped at institution (fake or low confidence)
IN_DEPARTMENT = "in department"                    # routed by the model
FLAGGED = "in department (needs verification)"     # routed, department score below t
FORWARDED = "forwarded by officer"                 # officer confirmed and sent to a department
DISCARDED = "discarded as fake"


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.complaints = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.complaints, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, record: dict):
        for s in record["sub_complaints"]:
            if s["status"] == "stopped at institution":
                s["officer_status"] = AWAITING_REVIEW
            elif s["needs_verification"] == "department":
                s["officer_status"] = FLAGGED
            else:
                s["officer_status"] = IN_DEPARTMENT
            s["current_institution"] = s["institution"]
            s["current_department"] = s["department"]
            s["history"] = [{"time": time.strftime("%H:%M:%S"),
                             "event": f"Model: {s['status']} ({s['institution']}"
                                      f"{' / ' + s['department'] if s['department'] else ''})"}]
        with self.lock:
            self.complaints.insert(0, record)
            self._save()
        return record

    def all(self):
        return self.complaints

    def get(self, complaint_id):
        return next((c for c in self.complaints if c["complaint_id"] == complaint_id), None)

    def find_sub(self, sub_id):
        for c in self.complaints:
            for s in c["sub_complaints"]:
                if s["sub_complaint_id"] == sub_id:
                    return c, s
        return None, None

    def inbox(self, institution):
        """Every sub-complaint currently at this institution, with its full complaint."""
        out = []
        for c in self.complaints:
            for s in c["sub_complaints"]:
                if s["current_institution"] == institution:
                    out.append({"complaint_id": c["complaint_id"],
                                "raw_complaint_text": c["raw_complaint_text"],
                                "processed_text": c["processed_text"],
                                "submitted_at": c["submitted_at"],
                                "other_parts": [
                                    {"sub_complaint_id": o["sub_complaint_id"],
                                     "institution": o["current_institution"],
                                     "department": o["current_department"]}
                                    for o in c["sub_complaints"] if o is not s],
                                **s})
        return out

    def act(self, sub_id, action, departments, department=None, institution=None):
        with self.lock:
            c, s = self.find_sub(sub_id)
            if s is None:
                raise KeyError(sub_id)
            inst = s["current_institution"]
            if action in ("forward", "reassign_department"):
                if department not in departments.get(inst, []):
                    raise ValueError(f"Choose a department of {inst}")
                s["current_department"] = department
                s["officer_status"] = FORWARDED
                event = f"Officer forwarded to {inst} / {department}"
            elif action == "confirm_department":
                s["officer_status"] = FORWARDED
                event = f"Officer confirmed {inst} / {s['current_department']}"
            elif action == "transfer":
                if institution not in departments or institution == inst:
                    raise ValueError("Choose the other institution")
                s["current_institution"] = institution
                s["current_department"] = None
                s["officer_status"] = AWAITING_REVIEW
                event = f"Officer transferred from {inst} to {institution}"
            elif action == "discard":
                s["officer_status"] = DISCARDED
                event = "Officer discarded the sub-complaint as fake"
            else:
                raise ValueError(f"Unknown action {action}")
            s["history"].append({"time": time.strftime("%H:%M:%S"), "event": event})
            self._save()
            return s

    def reset(self):
        with self.lock:
            self.complaints = []
            self._save()

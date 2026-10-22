"""Very small persistence layer: files on disk + one SQLite table of JSON packages.

For PP1 this keeps setup at zero. The package JSON maps 1:1 onto a PostgreSQL
JSONB column (and files onto cloud object storage) when the team integrates.
"""
import json
import sqlite3
import threading
import uuid
from pathlib import Path

from .config import settings
from .schemas import ComplaintPackage, StoredFile

_lock = threading.Lock()


def _db_path() -> Path:
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return settings.STORAGE_DIR / "complaints.sqlite3"


def _files_dir() -> Path:
    d = settings.STORAGE_DIR / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _connect():
    con = sqlite3.connect(_db_path(), check_same_thread=False)
    con.execute(
        "CREATE TABLE IF NOT EXISTS complaints ("
        "id TEXT PRIMARY KEY, submitted_at TEXT, status TEXT, package TEXT)"
    )
    return con


EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "application/pdf": ".pdf"}


def save_file(data: bytes, filename: str, content_type: str) -> StoredFile:
    file_id = uuid.uuid4().hex
    path = _files_dir() / f"{file_id}{EXT.get(content_type, '.bin')}"
    path.write_bytes(data)
    return StoredFile(
        file_id=file_id,
        filename=filename or path.name,
        content_type=content_type,
        size_bytes=len(data),
        uri=f"/api/v1/files/{file_id}",
    )


def file_path(file_id: str) -> Path | None:
    if not file_id.isalnum():
        return None
    matches = list(_files_dir().glob(f"{file_id}.*"))
    return matches[0] if matches else None


def save_package(pkg: ComplaintPackage) -> None:
    with _lock, _connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO complaints (id, submitted_at, status, package) VALUES (?,?,?,?)",
            (pkg.complaint_id, pkg.submitted_at.isoformat(), pkg.status, pkg.model_dump_json()),
        )


def get_package(complaint_id: str) -> ComplaintPackage | None:
    with _connect() as con:
        row = con.execute("SELECT package FROM complaints WHERE id=?", (complaint_id,)).fetchone()
    return ComplaintPackage.model_validate(json.loads(row[0])) if row else None


def list_packages(limit: int = 50) -> list[ComplaintPackage]:
    with _connect() as con:
        rows = con.execute(
            "SELECT package FROM complaints ORDER BY submitted_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [ComplaintPackage.model_validate(json.loads(r[0])) for r in rows]

"""Immutable raw snapshot: fetch pinned metadata, hash every raw file, verify against manifest.

The manifest (reports/raw_manifest.csv) is committed. Every later stage calls `verify` first,
so a changed or corrupted raw file stops the pipeline instead of silently changing results.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .config import METADATA_FILES, MLRAN_COMMIT, MLRAN_RAW_BASE, TABLES, Paths
from .report import write_stage

log = logging.getLogger(__name__)

ZENODO_RECORD = "https://zenodo.org/records/23403681"
FIELDS = ["path", "bytes", "sha256", "source", "retrieved_at"]


class SnapshotError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def expected_raw_files(paths: Paths) -> list[Path]:
    files: list[Path] = []
    for t in TABLES:
        files.append(paths.raw_dataset / f"{t}_dataset.parquet")
        files.append(paths.raw_dataset / f"{t}_feature_names_dic.json")
    files.extend(paths.metadata_dir / name for name in METADATA_FILES)
    return files


def fetch_metadata(paths: Paths) -> None:
    """Download the pinned metadata CSVs if they are not already present."""
    paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    for name, rel in METADATA_FILES.items():
        dest = paths.metadata_dir / name
        if dest.exists():
            continue
        url = f"{MLRAN_RAW_BASE}/{rel}"
        log.info("downloading %s", url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed URL
                    expected = resp.headers.get("Content-Length")
                    body = resp.read()
                if expected is None or len(body) == int(expected):
                    break
                log.warning("short read (%d/%s bytes), retry %d", len(body), expected, attempt)
            except (OSError, http.client.HTTPException) as e:
                log.warning("download failed (%s), retry %d", e, attempt)
        else:
            raise SnapshotError(f"could not download {url}")
        tmp.write_bytes(body)
        tmp.rename(dest)


def _source(path: Path, paths: Paths) -> str:
    if path.parent == paths.metadata_dir:
        return f"github.com/faithfulco/mlran@{MLRAN_COMMIT}"
    return f"{ZENODO_RECORD} (MLRan_full_dataset.zip)"


def build_manifest(paths: Paths) -> list[dict[str, str]]:
    """Hash all raw files and write the manifest. Refuses to overwrite a different manifest."""
    fetch_metadata(paths)
    missing = [p for p in expected_raw_files(paths) if not p.exists()]
    if missing:
        raise SnapshotError(f"missing raw files: {[str(p) for p in missing]}")

    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for p in expected_raw_files(paths):
        log.info("hashing %s", p.name)
        rows.append(
            {
                "path": str(p.relative_to(paths.root)) if p.is_relative_to(paths.root) else str(p),
                "bytes": str(p.stat().st_size),
                "sha256": sha256_file(p),
                "source": _source(p, paths),
                "retrieved_at": now,
            }
        )

    if paths.raw_manifest.exists():
        old = {r["path"]: r["sha256"] for r in _read(paths.raw_manifest)}
        new = {r["path"]: r["sha256"] for r in rows}
        if old != new:
            raise SnapshotError(
                "raw files differ from the committed manifest; the snapshot is immutable. "
                "Delete reports/raw_manifest.csv only if the dataset version changed on purpose."
            )
        log.info("manifest unchanged; keeping original retrieved_at timestamps")
        return _read(paths.raw_manifest)

    paths.raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    with paths.raw_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_stage(
        paths.stage_report,
        "snapshot",
        {"files": len(rows), "total_bytes": sum(int(r["bytes"]) for r in rows)},
    )
    return rows


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verify(paths: Paths, full_hash: bool = False) -> None:
    """Check every manifest entry exists with the recorded size (and hash when full_hash).

    Size-only is the fast default for each stage; `mlran verify --full` rehashes everything.
    """
    if not paths.raw_manifest.exists():
        raise SnapshotError("no raw manifest; run `mlran snapshot` first")
    problems = []
    for r in _read(paths.raw_manifest):
        p = Path(r["path"])
        p = p if p.is_absolute() else paths.root / p
        if not p.exists():
            problems.append(f"missing: {p}")
            continue
        if p.stat().st_size != int(r["bytes"]):
            problems.append(f"size changed: {p}")
            continue
        if full_hash and sha256_file(p) != r["sha256"]:
            problems.append(f"hash changed: {p}")
    if problems:
        raise SnapshotError("raw snapshot verification failed:\n" + "\n".join(problems))
    log.info("raw snapshot verified (%s)", "sha256" if full_hash else "size")

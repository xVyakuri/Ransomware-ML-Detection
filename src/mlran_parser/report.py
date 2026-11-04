"""Stage report: per-stage counts of records in/out, drops, malformed and duplicate rates.

Each stage writes its own section into reports/stage_report.json so a rerun of one stage
updates only that section.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_stage(path: Path, stage: str, payload: dict[str, Any]) -> None:
    """Replace one stage's section in the report file."""
    report: dict[str, Any] = {}
    if path.exists():
        report = json.loads(path.read_text())
    payload = {"completed_at": datetime.now(UTC).isoformat(timespec="seconds"), **payload}
    report[stage] = _jsonable(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    log.info("stage report updated: %s", stage)


def read_stage(path: Path, stage: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get(stage, {})

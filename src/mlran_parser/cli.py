"""Command line: `mlran <stage>` or `mlran all`.

Stages run in order: snapshot -> convert -> aggregates -> split -> leakcheck -> audit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import aggregates, audit, convert, labels, leakage, snapshot, split
from .config import Paths
from .report import write_stage

log = logging.getLogger("mlran")


def _paths(args: argparse.Namespace) -> Paths:
    return Paths.default(Path(args.root) if args.root else None)


def cmd_snapshot(p: Paths, args) -> int:
    snapshot.build_manifest(p)
    return 0


def cmd_verify(p: Paths, args) -> int:
    snapshot.verify(p, full_hash=args.full)
    return 0


def cmd_labels(p: Paths, args) -> int:
    snapshot.verify(p)
    p.ensure()
    frame, stats = labels.build_labels(p)
    frame.to_csv(p.labels, index=False)
    write_stage(p.stage_report, "labels", stats)
    return 0


def cmd_convert(p: Paths, args) -> int:
    convert.run(p)
    return 0


def cmd_aggregates(p: Paths, args) -> int:
    aggregates.run(p)
    return 0


def cmd_split(p: Paths, args) -> int:
    split.run(p)
    return 0


def cmd_leakcheck(p: Paths, args) -> int:
    return 0 if all(c.passed for c in leakage.run_all(p)) else 1


def cmd_audit(p: Paths, args) -> int:
    return 0 if audit.run(p) else 1


STAGES = {
    "snapshot": cmd_snapshot,
    "convert": cmd_convert,
    "aggregates": cmd_aggregates,
    "split": cmd_split,
    "leakcheck": cmd_leakcheck,
    "audit": cmd_audit,
}


def cmd_all(p: Paths, args) -> int:
    for name, fn in STAGES.items():
        t0 = time.time()
        log.info("=== %s ===", name)
        rc = fn(p, args)
        log.info("=== %s done (%.0fs) ===", name, time.time() - t0)
        if rc:
            log.error("stage %s failed; stopping", name)
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlran", description=__doc__)
    ap.add_argument("--root", help="repository root (default: current directory)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in [*STAGES, "all"]:
        sub.add_parser(name)
    sub.add_parser("labels", help="rebuild labels.csv only (convert also does this)")
    v = sub.add_parser("verify", help="check raw files against the manifest")
    v.add_argument("--full", action="store_true", help="rehash every file (slow)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    fn = {**STAGES, "all": cmd_all, "verify": cmd_verify, "labels": cmd_labels}[args.cmd]
    try:
        return fn(_paths(args), args)
    except (snapshot.SnapshotError, convert.ConvertError, split.SplitError, labels.LabelError) as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())

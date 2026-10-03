"""Pure helpers for reproducible, machine-readable parkour evaluation."""

import json
import os
import re
from pathlib import Path


SCHEMA_VERSION = 1
RANKING_RULES = (
    "successful full campaign first",
    "more cleared terrain stages",
    "fewer failures",
    "higher best course progress",
    "fewer resets",
    "shorter simulation time",
)


def checkpoint_number(policy_path):
    """Extract a checkpoint iteration from policy_4200.pt-like names."""
    match = re.search(
        r"(?:policy|model)[_-](\d+)(?:[^0-9].*)?\.pt$",
        Path(policy_path).name)
    return int(match.group(1)) if match else None


def evaluation_record(
        policy_path, result=None, error=None, wall_time_seconds=None):
    """Build one JSON-compatible candidate record."""
    record = {
        "policy_path": str(policy_path),
        "checkpoint": checkpoint_number(policy_path),
        "status": "ok" if error is None else "error",
    }
    if wall_time_seconds is not None:
        record["wall_time_seconds"] = float(wall_time_seconds)
    if error is not None:
        record.update({
            "success": False,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        return record

    result = dict(result or {})
    record.update(result)
    record["success"] = bool(
        result.get("campaign_finished", result.get("completed", False)))
    return record


def evaluation_rank_key(record):
    """Return a stability-first key; larger tuples rank better."""
    if record.get("status") != "ok":
        return (0, 0, float("-inf"), float("-inf"),
                float("-inf"), float("-inf"))
    return (
        int(bool(record.get("success", False))),
        int(record.get("stage_clears") or 0),
        -int(record.get("failures") or 0),
        float(record.get("best_progress_fraction") or 0.0),
        -int(record.get("resets") or 0),
        -float(record.get("sim_time") or 0.0),
    )


def best_evaluation(records):
    """Return the best valid record, or None when every run errored."""
    valid = [record for record in records
             if record.get("status") == "ok"]
    return max(valid, key=evaluation_rank_key) if valid else None


def evaluation_report(
        config_file, duration_seconds, records, command_x=None,
        control_overrides=None):
    """Build the versioned top-level report."""
    best = best_evaluation(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "config_file": str(config_file),
        "duration_limit_seconds": float(duration_seconds),
        "command_x": (
            None if command_x is None else float(command_x)),
        "control_overrides": dict(control_overrides or {}),
        "ranking_rules": list(RANKING_RULES),
        "candidate_count": len(records),
        "successful_count": sum(
            bool(record.get("success")) for record in records),
        "best_policy_path": (
            best.get("policy_path") if best is not None else None),
        "best_checkpoint": (
            best.get("checkpoint") if best is not None else None),
        "candidates": list(records),
    }


def write_evaluation_report(path, report):
    """Atomically write an indented UTF-8 JSON report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(str(temporary), str(output))

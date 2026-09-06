#!/usr/bin/env python3
"""Validate a spatially sampled reproduction and compare committed results.

The comparison is intentionally informational: spatial key sampling and fresh
per-trace models change workload rates. Structural and simulator-accounting failures
are hard errors; numerical deltas from the full committed run are reported but
are not treated as regressions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence


COUNTERS = (
    "all", "hit_fresh", "hit_reval", "hit_stale", "miss_mandatory",
    "miss_expired", "miss_evicted", "fetches", "revals", "reval_wasted",
)
STATE_COUNTERS = COUNTERS


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no data rows in {path}")
    missing = set(COUNTERS) - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def _threshold(row: dict[str, str]) -> float:
    return float(row["mlthres"])


def _key(row: dict[str, str]) -> tuple[str, str, str, float, str]:
    return (
        row["in"], row["cache_type"], row["mode"],
        round(_threshold(row), 6), row["evict_expired"].lower(),
    )


def _rates(row: dict[str, str]) -> dict[str, float]:
    total = float(row["all"])
    return {
        "hit_rate": sum(float(row[name]) for name in ("hit_fresh", "hit_reval", "hit_stale")) / total,
        "origin_fetch_rate": float(row["fetches"]) / total,
        "revalidation_rate": float(row["revals"]) / total,
        "optional_miss_rate": sum(float(row[name]) for name in ("miss_expired", "miss_evicted")) / total,
    }


def _expected_keys(
    traces: Sequence[str], caches: Sequence[str], ml_thresholds: Sequence[float],
    oracle_thresholds: Sequence[float],
) -> dict[str, set[tuple[str, str, str, float, str]]]:
    return {
        "baseline.csv": {
            (trace, cache, "never", 1.0, "true") for trace in traces for cache in caches
        },
        "baseline_keep.csv": {
            (trace, cache, "never", 1.0, "false") for trace in traces for cache in caches
        },
        "oracle.csv": {
            (trace, cache, "oracle", threshold, "true")
            for trace in traces for cache in caches for threshold in oracle_thresholds
        },
        **{
            f"ml_{dataset}_rfc.csv": {
                (trace, cache, "ml", threshold, "true")
                for trace in traces if trace.startswith(dataset + "_")
                for cache in caches for threshold in ml_thresholds
            }
            for dataset in ("cf", "fb", "wm")
        },
    }


def validate(
    candidate_dir: Path,
    reference_dir: Path,
    traces: Sequence[str],
    caches: Sequence[str],
    ml_thresholds: Sequence[float],
    oracle_thresholds: Sequence[float],
    capacity_gib: int = 2048,
    key_sample_ratio: int = 256,
) -> dict[str, object]:
    expected = _expected_keys(traces, caches, ml_thresholds, oracle_thresholds)
    candidates: dict[str, list[dict[str, str]]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for filename, expected_rows in expected.items():
        rows = _read(candidate_dir / filename)
        candidates[filename] = rows
        actual = {_key(row) for row in rows}
        missing = sorted(expected_rows - actual)
        unexpected = sorted(actual - expected_rows)
        if missing:
            errors.append(f"{filename}: missing {len(missing)} expected configuration(s): {missing}")
        if unexpected:
            errors.append(f"{filename}: found {len(unexpected)} unexpected configuration(s): {unexpected}")
        if len(actual) != len(rows):
            errors.append(f"{filename}: duplicate configuration rows")

        for row in rows:
            label = f"{filename}:{_key(row)}"
            if int(row["gib"]) != capacity_gib:
                errors.append(f"{label}: gib={row['gib']}, expected {capacity_gib}")
            # wm_u has a simulator-specific multiplier, but bounded validation
            # deliberately selects wm_t, so every selected trace has one KSR.
            if int(row["ksr"]) != key_sample_ratio:
                errors.append(f"{label}: ksr={row['ksr']}, expected {key_sample_ratio}")
            try:
                values = {name: int(row[name]) for name in COUNTERS}
            except ValueError:
                errors.append(f"{label}: non-integer counter")
                continue
            accounted = sum(values[name] for name in (
                "hit_fresh", "hit_reval", "hit_stale", "miss_mandatory",
                "miss_expired", "miss_evicted",
            ))
            if values["all"] != accounted:
                errors.append(f"{label}: request counters sum to {accounted}, all={values['all']}")
            if values["all"] <= 0:
                errors.append(f"{label}: all must be positive")
            if any(value < 0 for value in values.values()):
                errors.append(f"{label}: negative counter")
            if values["reval_wasted"] > values["revals"]:
                errors.append(f"{label}: reval_wasted exceeds revals")
            if row["mode"] == "never" and any(
                values[name] for name in ("hit_reval", "revals", "reval_wasted")
            ):
                errors.append(f"{label}: never policy recorded revalidation activity")

    baseline_by_pair = {
        (row["in"], row["cache_type"]): row for row in candidates["baseline.csv"]
    }
    for dataset in ("cf", "fb", "wm"):
        for row in candidates[f"ml_{dataset}_rfc.csv"]:
            if not math.isclose(_threshold(row), 1.0):
                continue
            baseline = baseline_by_pair.get((row["in"], row["cache_type"]))
            if baseline is None:
                continue
            differences = [name for name in STATE_COUNTERS if row[name] != baseline[name]]
            if differences:
                errors.append(
                    f"ML threshold 1 differs from never for {row['in']}/{row['cache_type']}: "
                    + ", ".join(differences)
                )

    # Merely requesting two cache policies is weak coverage if the sampled
    # working set never creates an eviction decision. Require at least one
    # baseline trace to distinguish the policy implementations.
    if len(caches) > 1:
        baseline_by_trace: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates["baseline.csv"]:
            baseline_by_trace[row["in"]].append(row)
        policy_difference = any(
            len({tuple(row[name] for name in STATE_COUNTERS) for row in rows}) > 1
            for rows in baseline_by_trace.values()
        )
        if not policy_difference:
            errors.append(
                "cache-policy coverage is degenerate: all baseline counters are identical"
            )

    # Threshold response is useful as a smoke signal but not a universal
    # correctness law because revalidation changes later cache state.
    for filename in ("ml_cf_rfc.csv", "ml_fb_rfc.csv", "ml_wm_rfc.csv", "oracle.csv"):
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in candidates[filename]:
            groups[(row["in"], row["cache_type"])].append(row)
        for pair, rows in groups.items():
            # Lower ML thresholds and higher oracle lookahead multipliers are
            # both progressively more permissive.
            ordered = sorted(
                rows, key=_threshold, reverse=filename.startswith("ml_")
            )
            revals = [int(row["revals"]) for row in ordered]
            if filename.startswith("ml_") and all(value == 0 for value in revals):
                errors.append(f"{filename}:{pair}: no threshold exercises revalidation")
            if any(after < before for before, after in zip(revals, revals[1:])):
                warnings.append(f"{filename}:{pair}: revalidation count is not monotonic by threshold")

    reference_files = {
        "baseline.csv": "r10i1_baseline.csv",
        "baseline_keep.csv": "r10i1_baseline_keep.csv",
        "oracle.csv": "r10i1_oracle.csv",
        "ml_cf_rfc.csv": "r10i1_ml_cf_rfc.csv",
        "ml_fb_rfc.csv": "r10i1_ml_fb_rfc.csv",
        "ml_wm_rfc.csv": "r10i1_ml_wm_rfc.csv",
    }
    comparisons: list[dict[str, object]] = []
    for candidate_name, reference_name in reference_files.items():
        references = {_key(row): row for row in _read(reference_dir / reference_name)}
        for row in candidates[candidate_name]:
            reference = references.get(_key(row))
            if reference is None:
                warnings.append(f"no committed match for {candidate_name}:{_key(row)}")
                continue
            candidate_rates = _rates(row)
            reference_rates = _rates(reference)
            comparisons.append({
                "file": candidate_name,
                "trace": row["in"],
                "cache_type": row["cache_type"],
                "mode": row["mode"],
                "threshold": _threshold(row),
                "candidate": candidate_rates,
                "committed": reference_rates,
                "delta": {
                    name: candidate_rates[name] - reference_rates[name]
                    for name in candidate_rates
                },
            })

    if not comparisons:
        errors.append("no candidate rows matched committed results")

    return {
        "status": "pass" if not errors else "fail",
        "scope": {
            "traces": list(traces),
            "cache_types": list(caches),
            "ml_thresholds": list(ml_thresholds),
            "oracle_thresholds": list(oracle_thresholds),
            "capacity_gib": capacity_gib,
            "key_sample_ratio": key_sample_ratio,
        },
        "candidate_rows": sum(len(rows) for rows in candidates.values()),
        "committed_matches": len(comparisons),
        "errors": errors,
        "warnings": warnings,
        "comparisons": comparisons,
        "comparison_note": (
            "Candidates preserve each selected trace's complete timeline while spatially "
            "sampling keys and training fresh per-trace models. Rate deltas against the full "
            "committed workload runs are informational and are not pass/fail tolerances."
        ),
    }


def _percent(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def write_reports(report: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation.json"
    markdown_path = output_dir / "validation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    errors = report["errors"]
    warnings = report["warnings"]
    comparisons = report["comparisons"]
    lines = [
        "# Sampled full-timeline validation",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        f"Validated {report['candidate_rows']} candidate rows; "
        f"{report['committed_matches']} matched committed full-run rows.",
        "",
        str(report["comparison_note"]),
        "",
        "## Structural checks",
        "",
    ]
    lines.extend([f"- ERROR: {item}" for item in errors] or ["- All hard checks passed."])
    lines.extend([f"- Warning: {item}" for item in warnings])
    lines.extend([
        "",
        "## Comparison with committed results",
        "",
        "| Trace | Cache | Mode | Threshold | Hit-rate delta | Fetch-rate delta | Revalidation-rate delta | Optional-miss delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in comparisons:
        delta = item["delta"]
        lines.append(
            f"| {item['trace']} | {item['cache_type']} | {item['mode']} | "
            f"{item['threshold']:.2f} | {_percent(delta['hit_rate'])} | "
            f"{_percent(delta['origin_fetch_rate'])} | {_percent(delta['revalidation_rate'])} | "
            f"{_percent(delta['optional_miss_rate'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--cache-types", nargs="+", required=True)
    parser.add_argument("--ml-thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--oracle-thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--capacity-gib", type=int, default=2048)
    parser.add_argument("--key-sample-ratio", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate(
            args.candidate_dir, args.reference_dir, args.traces, args.cache_types,
            args.ml_thresholds, args.oracle_thresholds,
            args.capacity_gib, args.key_sample_ratio,
        )
        write_reports(report, args.output_dir)
    except (FileNotFoundError, ValueError) as error:
        print(f"validation error: {error}")
        return 2
    print(f"validation {report['status']}: {report['candidate_rows']} rows, "
          f"{report['committed_matches']} committed matches")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

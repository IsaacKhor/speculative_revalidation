#!/usr/bin/env python3
"""Regenerate the data-derived figures used by the paper.

This is a direct, path-parameterized extraction of the plotting code in the
analysis notebooks.  Some calculations may look unusual (notably the
positional alignment in the oracle plot and the CDN feature-importance order),
but are intentionally retained so this script reproduces the published plots.
"""

from __future__ import annotations

import argparse
import itertools
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

# The reproducer is normally run without a display (CI, containers, or SSH).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RENAME = {
    "cf_a": "BOS 1",
    "cf_b": "BOS 2",
    "cf_c": "DXB 1",
    "cf_d": "DXB 2",
    "cf_e": "SIN 1",
    "cf_f": "SIN 2",
    "cf_g": "FRA 1",
    "cf_h": "FRA 2",
    "fb_a": "Meta 1",
    "fb_b": "Meta 2",
    "fb_c": "Meta 3",
    "wm_t": "WM 1",
    "wm_u": "WM 2",
    "106m105": "BOS 1",
    "106m106": "BOS 2",
    "243m12": "DXB 1",
    "243m13": "DXB 2",
    "411m264": "SIN 1",
    "411m325": "SIN 2",
    "472m378": "FRA 1",
    "472m379": "FRA 2",
    "fb_reag": "Meta 1",
    "fb_rhna": "Meta 2",
    "fb_rprn": "Meta 3",
}

# This is deliberately the notebook's source list: cf_h was loaded by some
# other analyses, but was not included in the transferability/model notebook.
MODEL_SOURCES = [
    "cf_a", "cf_b", "cf_c", "cf_d", "cf_e", "cf_f", "cf_g",
    "fb_a", "fb_b", "fb_c", "wm_t", "wm_u",
]
ALL_SOURCES = [*MODEL_SOURCES[:7], "cf_h", *MODEL_SOURCES[7:]]
CF_SERVER = dict(zip(
    (f"cf_{letter}" for letter in "abcdefgh"),
    ("106m105", "106m106", "243m12", "243m13", "411m264", "411m325",
     "472m378", "472m379"),
))
FB_STEM = {"fb_a": "reag", "fb_b": "rhna", "fb_c": "rprn"}
EXPIRY_COLUMNS = [
    "ts", "next_ts", "ttl", "create_ts", "freq", "last_ts",
    "update_ts", "revals", "good_rv", "bad_rv", "mime", "size",
]

ACTIVE_FIGURES = (
    "ttl_cdf_by_key",
    "miss_types_by_trace",
    "expiry_population_ttl_dist",
    "oracle_eligible",
    "ml_tradeoff_combined",
    "all_models_feature_importance",
    "heuristic_ml_prc_combined",
    "ml_tradeoff_by_eviction_avg",
    "cross_dataset_auc_roc_heatmap",
    "miss_over_time",
)


def _linestyle(source: str) -> str:
    if source.startswith("cf"):
        return "solid"
    if source.startswith("fb"):
        return "dashed"
    if source.startswith("wm"):
        return "dotted"
    return "solid"


def _sample_rows(fraction: float) -> Callable[[int], bool]:
    # This intentionally uses the process-global, unseeded RNG, as did the
    # notebooks that produced the paper figures.
    return lambda _row: random.random() > fraction


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required reproduction input not found: {path}")
    return path


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "required reproduction input not found; tried: "
        + ", ".join(str(path) for path in paths)
    )


def _expiry_path(inputs: "Inputs", source: str) -> Path:
    """Resolve both the archive layout and freshly generated trace layout."""
    filename = f"r10{source}.csv.zst"
    fresh_zst = inputs.expiry_dir / filename
    fresh_csv = inputs.expiry_dir / f"{source}.csv"
    archive = inputs.trace_root / "cdn_cf21_expiry_events" / filename
    return _first_existing(fresh_zst, fresh_csv, archive)


@dataclass
class Inputs:
    trace_root: Path
    results_dir: Path
    models_dir: Path
    output_dir: Path
    expiry_dir: Path
    sources: tuple[str, ...] = tuple(ALL_SOURCES)
    cache: dict[str, Any] = field(default_factory=dict)

    def result(self, name: str) -> Path:
        return _required(self.results_dir / name)

    def result_one_of(self, *names: str) -> Path:
        for name in names:
            path = self.results_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            "required reproduction input not found; tried: "
            + ", ".join(str(self.results_dir / name) for name in names)
        )

    def model(self, name: str) -> Path:
        return _required(self.models_dir / name)

    def model_one_of(self, *names: str) -> Path:
        for name in names:
            path = self.models_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            "required reproduction model not found; tried: "
            + ", ".join(str(self.models_dir / name) for name in names)
        )

    def trace(self, *parts: str) -> Path:
        return _required(self.trace_root.joinpath(*parts))

    def save(self, name: str, **kwargs: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output_dir / f"{name}.png", **kwargs)
        plt.close()


def _configure_plots() -> None:
    plt.rcParams.update({
        "legend.fontsize": "large",
        "figure.figsize": (6, 4),
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
    })


def _baseline_keep(inputs: Inputs) -> pd.DataFrame:
    key = "baseline_keep"
    if key not in inputs.cache:
        df = pd.read_csv(inputs.result_one_of(
            "r10i1_baseline_keep.csv", "baseline_keep.csv"
        ))
        df["miss_all"] = df.miss_mandatory + df.miss_expired + df.miss_evicted
        df["miss_mandatory_pct"] = df.miss_mandatory / df.miss_all
        df["miss_expired_pct"] = df.miss_expired / df.miss_all
        df["miss_evicted_pct"] = df.miss_evicted / df.miss_all
        df["in_lbl"] = df["in"].map(RENAME).fillna(df["in"])
        inputs.cache[key] = df
    return inputs.cache[key]


def _simulation_results(inputs: Inputs) -> pd.DataFrame:
    key = "simulation_results"
    if key in inputs.cache:
        return inputs.cache[key]
    baseline = inputs.result_one_of("r10i1_baseline.csv", "baseline.csv")
    oracle = inputs.result_one_of("r10i1_oracle.csv", "oracle.csv")
    fresh_names = ["ml_cf_rfc.csv", "ml_fb_rfc.csv", "ml_wm_rfc.csv"]
    if all((inputs.results_dir / name).exists() for name in fresh_names):
        ml_paths = [inputs.results_dir / name for name in fresh_names]
        ml_paths.extend(
            inputs.results_dir / name
            for name in (
                "ml_wm_rfc_extended.csv",
                "ml_fb_rfc_extended2.csv",
                "ml_wm_rfc_extended2.csv",
            )
            if (inputs.results_dir / name).exists()
        )
    else:
        legacy_names = [
            "r10i1_ml_cf_rfc.csv",
            "r10i1_ml_fb_rfc.csv",
            "r10i1_ml_wm_rfc.csv",
            "r10i1_ml_wm_rfc_extended.csv",
            "r10i1_ml_fb_rfc_extended2.csv",
            "r10i1_ml_wm_rfc_extended2.csv",
        ]
        ml_paths = [inputs.result(name) for name in legacy_names]
    df = pd.concat([pd.read_csv(path) for path in [baseline, oracle, *ml_paths]])
    df = df.drop(columns=["rv_min_ttl", "rv_min_freq", "rv_max_za"])
    df = df.drop_duplicates().sort_values(
        by=["mode", "in", "cache_type", "mlthres"]
    )
    df["trace"] = df["in"].map(RENAME).fillna(df["in"])
    df["hit_all"] = df.hit_fresh + df.hit_reval + df.hit_stale
    df["hit_pc"] = df.hit_all / df["all"] * 100
    df["miss_all"] = df.miss_mandatory + df.miss_expired + df.miss_evicted
    df["rv_pending"] = df.revals - df.hit_reval - df.reval_wasted
    df["optmiss"] = (df.miss_all - df.miss_mandatory) / df["all"] * 100
    df["amp_lower"] = df.reval_wasted / (df.miss_all + df.hit_reval) + 1
    df["amp_upper"] = (
        (df.reval_wasted + df.rv_pending) / (df.miss_all + df.hit_reval) + 1
    )
    df["amp_avg"] = (df.amp_lower + df.amp_upper) / 2
    inputs.cache[key] = df
    return df


def _calc_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["should_reval"] = (
        (df.ts <= df.next_ts)
        & (df.next_ts <= (df.ts + df.ttl))
        & (df.ttl < 7 * 86400)
    )
    df["generations"] = ((df.ts - df.create_ts - 1) / df.ttl).round()
    df["t_since_last"] = df.ts - df.last_ts
    df["t_since_last_frac"] = (df.t_since_last - 1) / df.ttl
    return df


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return (
        df[[
            "ttl", "freq", "generations", "t_since_last",
            "t_since_last_frac", "mime", "size",
        ]],
        df["should_reval"],
    )


def _expiry_ml(inputs: Inputs) -> pd.DataFrame:
    key = "expiry_ml"
    if key in inputs.cache:
        return inputs.cache[key]
    frames = []
    # Partial-source runs are the sampled validation path.  Their expiry files
    # are already small, and sampling them again can remove an entire class.
    partial_sources = not set(MODEL_SOURCES).issubset(inputs.sources)
    for source in (source for source in MODEL_SOURCES if source in inputs.sources):
        fraction = 1.0 if partial_sources else (
            0.03 if source.startswith("cf") else (
                0.10 if source.startswith("fb") else 0.05
            )
        )
        path = _expiry_path(inputs, source)
        frame = pd.read_csv(
            path,
            names=EXPIRY_COLUMNS,
            skiprows=None if fraction == 1.0 else _sample_rows(fraction),
        )
        frame["in"] = source
        frame["src"] = RENAME.get(source, source)
        frames.append(frame)
    df = _calc_features(pd.concat(frames))
    inputs.cache[key] = df
    return df


def _load_aggregate_models(inputs: Inputs) -> tuple[Any, Any, Any]:
    key = "aggregate_models"
    if key not in inputs.cache:
        import pickle

        models = []
        for source in ("cf", "fb", "wm"):
            path = inputs.model_one_of(
                f"v10_{source}_rfc.pkl", f"fresh_{source}_rfc.pkl"
            )
            with path.open("rb") as handle:
                models.append(pickle.load(handle))
        inputs.cache[key] = tuple(models)
    return inputs.cache[key]


def _collect_model_stats(df: pd.DataFrame, model: Any) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
    )
    from sklearn.model_selection import train_test_split

    x, y = _xy(df)
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )
    probability = model.predict_proba(x_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probability)
    return {
        "precision": precision,
        "recall": recall,
        "prc_auc": average_precision_score(y_test, probability),
        "feature_importance": pd.DataFrame({
            "feature": x.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False),
    }


def _aggregate_model_stats(inputs: Inputs) -> tuple[dict[str, Any], ...]:
    key = "aggregate_model_stats"
    if key not in inputs.cache:
        df = _expiry_ml(inputs)
        models = _load_aggregate_models(inputs)
        groups = (
            df[df["in"].str.startswith("cf_")],
            df[df["in"].str.startswith("fb_")],
            df[df["in"].str.startswith("wm_")],
        )
        inputs.cache[key] = tuple(
            _collect_model_stats(group, model)
            for group, model in zip(groups, models)
        )
    return inputs.cache[key]


def ttl_cdf_by_key(inputs: Inputs) -> None:
    import duckdb

    connection = duckdb.connect()
    connection.execute("SET memory_limit = '48GB'")
    selected_servers = [CF_SERVER[source] for source in inputs.sources if source in CF_SERVER]
    if not selected_servers:
        df_cf = pd.DataFrame(columns=["trace", "ttl_min", "key_count"])
    elif (inputs.trace_root / "cdn_cf25_parquet").is_dir():
        cf_glob = str(inputs.trace_root / "cdn_cf25_parquet" / "**" / "*.parquet")
        quoted_cf = cf_glob.replace("'", "''")
        server_sql = ", ".join(
            "'" + server.replace("'", "''") + "'" for server in selected_servers
        )
        df_cf = connection.execute(f"""
            SELECT server AS trace, FLOOR(expiry_time / 60) AS ttl_min,
                   COUNT(DISTINCT key) AS key_count
            FROM '{quoted_cf}' WHERE server IN ({server_sql}) GROUP BY server, ttl_min
        """).df()
    else:
        cf_csv_dir = _first_existing(
            inputs.trace_root / "cdn_cf25_csv", inputs.trace_root / "cf" / "csv"
        )
        cf_frames = []
        for server in selected_servers:
            path = _required(cf_csv_dir / f"{server}.csv.zst")
            quoted_path = str(path).replace("'", "''")
            quoted_server = server.replace("'", "''")
            cf_frames.append(connection.execute(f"""
                SELECT '{quoted_server}' AS trace,
                       FLOOR(CAST(SPLIT_PART(line, ',', 5) AS BIGINT) / 60) AS ttl_min,
                       COUNT(DISTINCT SPLIT_PART(line, ',', 2)) AS key_count
                FROM read_csv(
                    '{quoted_path}', columns = {{'line': 'VARCHAR'}}, delim = '\x1f',
                    header = false, auto_detect = false, quote = ''
                )
                WHERE NOT STARTS_WITH(line, 'timestamp,')
                GROUP BY ttl_min
            """).df())
        df_cf = pd.concat(cf_frames, ignore_index=True)

    if (inputs.trace_root / "cdn_fb23_parquet").is_dir():
        fb_dir = inputs.trace_root / "cdn_fb23_parquet"
        fb_is_parquet = True
        fb_paths = {
            source: _required(fb_dir / f"{stem}.parquet")
            for source, stem in FB_STEM.items() if source in inputs.sources
        }
    else:
        fb_dir = _first_existing(
            inputs.trace_root / "cdn_fb23_csv", inputs.trace_root / "fb23"
        )
        fb_is_parquet = False
        fb_paths = {
            source: _required(fb_dir / f"{stem}.csv.zst")
            for source, stem in FB_STEM.items() if source in inputs.sources
        }

    frames = [df_cf]
    for trace, path in fb_paths.items():
        quoted_path = str(path).replace("'", "''")
        reader = (
            f"'{quoted_path}'" if fb_is_parquet
            else (
                "read_csv("
                f"'{quoted_path}', columns = {{'line': 'VARCHAR'}}, delim = '\\x1f', "
                "header = false, auto_detect = false, quote = '')"
            )
        )
        if fb_is_parquet:
            query = f"""
                SELECT '{trace}' AS trace, FLOOR(ttl / 60) AS ttl_min,
                       COUNT(DISTINCT cacheKey) AS key_count
                FROM {reader} GROUP BY ttl_min
            """
        else:
            query = f"""
                SELECT '{trace}' AS trace,
                       FLOOR(CAST(SPLIT_PART(line, ',', 9) AS DOUBLE) / 60) AS ttl_min,
                       COUNT(DISTINCT SPLIT_PART(line, ',', 2)) AS key_count
                FROM {reader} WHERE NOT STARTS_WITH(line, 'timestamp,')
                GROUP BY ttl_min
            """
        frames.append(connection.execute(query).df())
    connection.close()
    df = pd.concat(frames)

    plt.figure(figsize=(6, 4))
    xmax = 525600 * 2
    for trace in sorted(df_cf.trace.unique()):
        subset = df[df.trace == trace].sort_values("ttl_min").copy()
        subset["cum_keys"] = subset.key_count.cumsum()
        subset["cdf"] = subset.cum_keys / subset.key_count.sum()
        plt.step(subset.ttl_min, subset.cdf, label=RENAME[trace], where="post")
    for trace in sorted(fb_paths):
        subset = df[df.trace == trace].sort_values("ttl_min").copy()
        subset["cum_keys"] = subset.key_count.cumsum()
        subset["cdf"] = subset.cum_keys / subset.key_count.sum()
        if (
            not subset.empty
            and subset.cdf.iloc[-1] >= 1.0
            and subset.ttl_min.iloc[-1] < xmax
        ):
            subset = pd.concat([
                subset, pd.DataFrame({"ttl_min": [xmax], "cdf": [1.0]})
            ], ignore_index=True)
        plt.step(
            subset.ttl_min, subset.cdf, "--", label=RENAME[trace], where="post"
        )
    plt.xscale("log")
    plt.xlabel("Time-to-Live")
    plt.ylabel("CDF (Fraction of Unique Keys)")
    plt.title("Key-Weighted CDF of TTLs")
    plt.xticks(
        [1, 15, 60, 240, 1440, 10080, 43200, 525600],
        ["1m", "15m", "1h", "4h", "1d", "7d", "30d", "1y"],
    )
    plt.xlim(-10, xmax)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    inputs.save("ttl_cdf_by_key")


def miss_types_by_trace(inputs: Inputs) -> None:
    df = _baseline_keep(inputs)
    miss_types = df[df.cache_type == "lru"].sort_values(by="in")
    plt.figure(figsize=(6, 4))
    plt.bar(
        miss_types.in_lbl, miss_types.miss_evicted_pct,
        label="Evicted", color="skyblue",
    )
    plt.bar(
        miss_types.in_lbl, miss_types.miss_expired_pct,
        bottom=miss_types.miss_evicted_pct, label="Expired", color="gold",
    )
    plt.bar(
        miss_types.in_lbl, miss_types.miss_mandatory_pct,
        bottom=miss_types.miss_evicted_pct + miss_types.miss_expired_pct,
        label="Compulsory", color="salmon",
    )
    plt.xlabel("Trace")
    plt.ylabel("Fraction of Misses")
    plt.title("Miss Types Breakdown")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.xticks(rotation=60)
    plt.tight_layout()
    inputs.save("miss_types_by_trace")


def expiry_population_ttl_dist(inputs: Inputs) -> None:
    # The source notebook used its older 11-column schema for this plot.
    columns = EXPIRY_COLUMNS[:-1]
    sources = [
        source for source in inputs.sources
        if source.startswith("cf_") or source.startswith("fb_")
    ]
    frames = []
    for source in sources:
        frame = pd.read_csv(
            _expiry_path(inputs, source),
            names=columns,
            skiprows=_sample_rows(0.01),
        )
        frame["in"] = source
        frame["trace"] = RENAME.get(source, source)
        frames.append(frame)
    df = pd.concat(frames)
    plt.figure(figsize=(6, 4))
    for source, group in df.groupby("in"):
        x_sorted = np.sort(group.ttl)
        y_values = np.linspace(0, 1, len(x_sorted))
        plt.plot(
            x_sorted, y_values, label=RENAME[source],
            linestyle=_linestyle(source), drawstyle="steps-post",
        )
    plt.axvline(86400, color="grey", linestyle=":", label="Wiki")
    plt.xticks(
        [0, 3600 * 24 * 7, 3600 * 24 * 14, 3600 * 24 * 21, 3600 * 24 * 30],
        ["0", "7d", "14d", "21d", "30d"],
    )
    plt.title("CDF: TTL of All Expiration Events")
    plt.xlabel("Time-to-live (TTL)")
    plt.ylabel("CDF")
    plt.ylim(0, 1)
    plt.xlim(0, 86400 * 25)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    inputs.save("expiry_population_ttl_dist", bbox_inches="tight")


def oracle_eligible(inputs: Inputs) -> None:
    df = _simulation_results(inputs)
    miss_types = _baseline_keep(inputs)
    miss_types = miss_types[miss_types.cache_type == "lru"].sort_values(by="in")
    oracle = df[
        (df.cache_type == "lru") & df["mode"].isin(["oracle", "never"])
    ]
    never = oracle[oracle["mode"] == "never"].sort_values("in").reset_index()
    oracle1 = oracle[
        (oracle["mode"] == "oracle") & (oracle.mlthres == 1)
    ].sort_values("in").reset_index()

    # Positional Series alignment is retained verbatim from the notebook.
    expiry_max = miss_types.miss_expired / (
        miss_types.miss_expired + miss_types.miss_evicted
    )
    eligible = 100 - (oracle1.optmiss / never.optmiss * 100)
    x = np.arange(len(oracle1.trace))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(
        x - width / 2, expiry_max * 100, width,
        color="gold", label="All expiry misses",
    )
    plt.bar(
        x + width / 2, eligible, width,
        color="lightgreen", label="Time to next access < TTL",
    )
    plt.xticks(x, oracle1.trace, rotation=45)
    plt.title("Time to next access of expiry misses")
    plt.ylabel("Fraction of non-compulsory misses (%)")
    plt.legend()
    plt.tight_layout()
    inputs.save("oracle_eligible")


def ml_tradeoff_combined(inputs: Inputs) -> None:
    df = _simulation_results(inputs)
    mldf = df[
        df["mode"].isin(["ml", "never"]) & (df.cache_type == "lru")
    ][["in", "trace", "mlthres", "optmiss", "amp_avg", "mode"]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for source in sorted(mldf["in"].unique()):
        nevermax = df[
            (df["mode"] == "never")
            & (df["in"] == source)
            & (df.cache_type == "lru")
        ].optmiss.values[0]
        subset = mldf[mldf["in"] == source].sort_values("amp_avg")
        axes[0].plot(
            (subset.amp_avg - 1) * 100, nevermax - subset.optmiss,
            marker="", ms=4, linewidth=1, label=RENAME.get(source, source),
            linestyle=_linestyle(source),
        )
        axes[1].plot(
            (subset.amp_avg - 1) * 100, subset.optmiss / nevermax * 100,
            marker="", ms=4, linewidth=1, label=RENAME.get(source, source),
            linestyle=_linestyle(source),
        )
    axes[0].set_xlim(-0.1, 4.5)
    axes[0].set_title("Hit rate improvement")
    axes[0].set_xlabel("Overhead %")
    axes[0].set_ylabel("Absolute hit rate improvement (% points)")
    axes[1].set_xlim(-0.1, 4.5)
    axes[1].set_title("Normalised miss rate")
    axes[1].set_xlabel("Overhead %")
    axes[1].set_ylabel("Non-compulsory Miss Rate (%)")
    dflru = df[df.cache_type == "lru"]
    for source in sorted(dflru["in"].unique()):
        subset = dflru[
            (dflru["in"] == source) & (dflru["mode"] == "ml")
        ].sort_values("mlthres")
        never = dflru[
            (dflru["in"] == source) & (dflru["mode"] == "never")
        ].optmiss.values[0]
        oracle = dflru[
            (dflru["in"] == source) & (dflru["mode"] == "oracle")
        ].optmiss.max()
        optimal = oracle - never
        axes[2].plot(
            (subset.amp_avg - 1) * 100,
            (subset.optmiss - never) / optimal * 100,
            marker="", label=RENAME.get(source, source),
            linestyle=_linestyle(source),
        )
    axes[2].axhline(y=100, color="gray", linestyle="--")
    axes[2].set_xlabel("Overhead %")
    axes[2].set_ylabel("Miss rate relative to oracle (%)")
    axes[2].set_xlim(-0.1, 4.5)
    axes[2].set_title("Performance relative to oracle")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, bbox_to_anchor=(1.01, 0.95), loc="upper left",
    )
    plt.tight_layout()
    inputs.save("ml_tradeoff_combined", bbox_inches="tight")


def all_models_feature_importance(inputs: Inputs) -> None:
    models = _load_aggregate_models(inputs)
    feature_names = [
        "ttl", "freq", "generations", "t_since_last",
        "t_since_last_frac", "mime", "size",
    ]
    stats = []
    for model in models:
        stats.append(pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False))
    cf_stats, fb_stats, wm_stats = stats
    features = sorted(cf_stats.feature.tolist())
    # Intentional notebook behavior: only the latter two vectors are reindexed
    # to `features`; CDN1 remains in descending-importance order.
    cf_importances = cf_stats.importance.tolist()
    fb_importances = fb_stats.set_index("feature").loc[features, "importance"].tolist()
    wm_importances = wm_stats.set_index("feature").loc[features, "importance"].tolist()
    y = np.arange(len(features))
    height = 0.25
    fig, ax = plt.subplots()
    ax.barh(y - height, cf_importances, height=height, label="CDN1", color="crimson")
    ax.barh(y, fb_importances, height=height, label="Meta", color="skyblue")
    ax.barh(y + height, wm_importances, height=height, label="Wikimedia", color="limegreen")
    feature_rename = {
        "ttl": "TTL", "freq": "Frequency", "generations": "Generations",
        "t_since_last": "Recency (secs)",
        "t_since_last_frac": "Recency (%)", "mime": "Content Type",
        "size": "Size",
    }
    ax.set_yticks(y, [feature_rename.get(feature, feature) for feature in features])
    ax.set_xlabel("Gini Importance")
    ax.set_title("Feature Importance Comparison - All Models")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    inputs.save("all_models_feature_importance", dpi=300, bbox_inches="tight")


def heuristic_ml_prc_combined(inputs: Inputs) -> None:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, precision_recall_curve

    df = _expiry_ml(inputs)
    models = _load_aggregate_models(inputs)
    groups = [
        df[df["in"].str.startswith("cf_")],
        df[df["in"].str.startswith("fb_")],
        df[df["in"].str.startswith("wm_")],
    ]
    curve_stats = []
    for group, model in zip(groups, models):
        x, y = _xy(group)
        _, x_test, _, y_test = train_test_split(
            x, y, test_size=0.33, random_state=42
        )
        probability = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, probability)
        curve_stats.append((
            precision, recall, average_precision_score(y_test, probability)
        ))

    samples = []
    for group in groups:
        sampled = group.sample(frac=0.05, random_state=42)
        x, y = _xy(sampled)
        samples.append((x, y.to_numpy(dtype=bool)))

    ttls = [0, 5, 10, 30, 60, 900, 3600, 21600, 86400,
            7 * 86400, 14 * 86400, 30 * 86400]
    freqs = list(range(0, 7))
    recencies = list(np.linspace(0.0, 1.0, num=20))
    generations = list(range(0, 10))
    labels = ("CDN1", "Meta", "Wiki")
    results: list[dict[str, Any]] = []
    for heuristic_id, (ttl, freq, recency, generation) in enumerate(
        itertools.product(ttls, freqs, recencies, generations)
    ):
        for label, (x, truth) in zip(labels, samples):
            truth_count = int(truth.sum())
            if truth_count == 0:
                continue
            predicted = (
                (x.generations.to_numpy() >= generation)
                | (
                    (x.ttl.to_numpy() <= ttl)
                    & (x.freq.to_numpy() <= freq)
                    & (x.t_since_last_frac.to_numpy() <= recency)
                )
            )
            predicted_count = int(predicted.sum())
            if predicted_count > 0:
                true_positive = int((predicted & truth).sum())
                results.append({
                    "heuristic_id": heuristic_id,
                    "precision": true_positive / predicted_count,
                    "recall": true_positive / truth_count,
                    "src": label,
                })
    heuristic_df = pd.DataFrame(results)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=heuristic_df, x="recall", y="precision", hue="src",
        alpha=0.66, s=4,
        palette={"CDN1": "blue", "Meta": "orange", "Wiki": "green"},
        legend=False,
    )
    for label, color, (precision, recall, score) in zip(
        labels, ("blue", "orange", "green"), curve_stats
    ):
        plt.plot(
            recall, precision, lw=2, color=color,
            label=f"{label} (AP = {score:.3f})",
        )
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall for ML and Heuristic Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    inputs.save("heuristic_ml_prc_combined", dpi=300, bbox_inches="tight")


def ml_tradeoff_by_eviction_avg(inputs: Inputs) -> None:
    df = _simulation_results(inputs)
    rows = []
    available_traces = set(df.loc[df["mode"] == "ml", "in"])
    for trace in [source for source in inputs.sources
                  if source.startswith("cf_") and source in available_traces]:
        mldf = df[
            (df["in"] == trace) & df["mode"].isin(["ml", "never"])
        ][["in", "mlthres", "optmiss", "amp_avg", "cache_type"]]
        for cache_type in df.cache_type.unique():
            has_never = (
                (df["mode"] == "never") & (df.cache_type == cache_type)
                & (df["in"] == trace)
            ).any()
            has_oracle = (
                (df["mode"] == "oracle") & (df.cache_type == cache_type)
                & (df["in"] == trace) & (df.mlthres == 1)
            ).any()
            if not has_never or not has_oracle:
                continue
            nevermax = df[
                (df["mode"] == "never")
                & (df.cache_type == cache_type)
                & (df["in"] == trace)
            ].optmiss.values[0]
            oraclemin = df[
                (df["mode"] == "oracle")
                & (df.cache_type == cache_type)
                & (df["in"] == trace)
                & (df.mlthres == 1)
            ].optmiss.values[0]
            subset = mldf[mldf.cache_type == cache_type].sort_values("amp_avg")
            pc_oracle = (
                (nevermax - subset.optmiss) / (nevermax - oraclemin) * 100
            )
            rows.append(pd.DataFrame({
                "cache_type": cache_type,
                "mlthres": subset.mlthres.values,
                "amp_avg": subset.amp_avg.values,
                "pc_oracle": pc_oracle.values,
            }))
    all_df = pd.concat(rows, ignore_index=True)
    aggregate = all_df.groupby(["cache_type", "mlthres"]).agg(
        pc_oracle_mean=("pc_oracle", "mean"),
        pc_oracle_std=("pc_oracle", "std"),
        amp_avg_mean=("amp_avg", "mean"),
    ).reset_index()
    plt.figure(figsize=(6, 4))
    for cache_type in sorted(aggregate.cache_type.unique()):
        subset = aggregate[aggregate.cache_type == cache_type].sort_values("mlthres")
        x = (subset.amp_avg_mean - 1) * 100
        y = subset.pc_oracle_mean
        error = subset.pc_oracle_std.fillna(0)
        plt.plot(x, y, marker="o", ms=4, linewidth=1, label=cache_type.upper())
        plt.fill_between(x, y - error, y + error, alpha=0.2)
    plt.legend()
    plt.title("Normalised Performance Relative to Oracle by Eviction Policy")
    plt.xlim(-0.1, 5.5)
    plt.xlabel("Overhead %")
    plt.ylabel("Normalised miss rate reduction (%)")
    plt.tight_layout()
    inputs.save("ml_tradeoff_by_eviction_avg")


def cross_dataset_auc_roc_heatmap(inputs: Inputs) -> None:
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = _expiry_ml(inputs)
    sources = [source for source in MODEL_SOURCES if source in inputs.sources]
    models: dict[str, Any] = {}
    for source in sources:
        x, y = _xy(df[df["in"] == source])
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.33, random_state=42
        )
        x_over, y_over = RandomOverSampler(random_state=42).fit_resample(
            x_train, y_train
        )
        model = RandomForestClassifier(
            n_estimators=32,
            criterion="gini",
            min_samples_leaf=1 / 2000,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_over, y_over)
        models[source] = model

    rows = []
    for train_source in sources:
        row: dict[str, Any] = {"train_src": train_source}
        for test_source in sources:
            x_test, y_test = _xy(df[df["in"] == test_source])
            # Published notebook used hard predictions, not probabilities.
            prediction = models[train_source].predict(x_test)
            row[test_source] = roc_auc_score(y_test, prediction)
        rows.append(row)
    matrix = pd.DataFrame(rows).set_index("train_src")
    ticks = sorted(sources)
    tick_labels = [RENAME.get(tick, tick) for tick in ticks]
    plt.figure(figsize=(9, 6))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.6, vmax=1.0
    )
    plt.title(
        "Cross-Dataset Model Transfer Performance (AUC-ROC)",
        fontsize="x-large",
    )
    plt.xlabel("Test Dataset")
    plt.xticks(ticks=np.arange(len(ticks)) + 0.5, labels=tick_labels)
    plt.yticks(ticks=np.arange(len(ticks)) + 0.5, labels=tick_labels)
    plt.ylabel("Training Dataset")
    plt.tight_layout()
    inputs.save("cross_dataset_auc_roc_heatmap")


def miss_over_time(inputs: Inputs) -> None:
    directory = _required(inputs.results_dir / "r10i3_stats_ts")
    pattern = re.compile(r"^\d+_([a-z]+_[a-z])_([0-9.]+)_stats_ts\.csv$")
    frames = []
    for path in sorted(directory.glob("*_stats_ts.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        trace, threshold = match.group(1), float(match.group(2))
        frame = pd.read_csv(path)
        frame["trace"] = trace
        frame["trace_lbl"] = RENAME.get(trace, trace)
        frame["conf_thres"] = threshold
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no stats time-series CSV files found in {directory}")
    ts = pd.concat(frames, ignore_index=True)
    ts["nonmand_miss"] = ts.miss_expired + ts.miss_evicted
    baseline = ts[ts.conf_thres == 1.0][
        ["trace", "all", "nonmand_miss"]
    ].rename(columns={"nonmand_miss": "nonmand_miss_baseline"})
    reduced = ts.merge(baseline, on=["trace", "all"], how="inner")
    reduced["nonmand_miss_red_pct"] = np.where(
        reduced.nonmand_miss_baseline > 0,
        (reduced.nonmand_miss_baseline - reduced.nonmand_miss)
        / reduced.nonmand_miss_baseline * 100.0,
        np.nan,
    )
    reduced = reduced[reduced.conf_thres != 1.0]

    focus_threshold = 0.8
    smooth_window = 55
    data = reduced[reduced.conf_thres == focus_threshold].copy()
    data["time_d"] = data.now_ts / 86400.0
    data["nonmand_miss_remaining"] = 1.0 - data.nonmand_miss_red_pct / 100.0
    traces = sorted(data.trace.unique())
    color_map = plt.get_cmap("tab10")
    colors = {trace: color_map(i) for i, trace in enumerate(traces)}
    fig, ax = plt.subplots(figsize=(6, 4))
    for trace in traces:
        subset = data[data.trace == trace].sort_values("now_ts").copy()
        subset["smoothed"] = subset.nonmand_miss_remaining.rolling(
            smooth_window, min_periods=1, center=True
        ).mean()
        ax.plot(
            subset.time_d, subset.smoothed, label=RENAME.get(trace, trace),
            color=colors[trace], linewidth=1.5,
        )
    ax.axhline(
        1.0, color="black", linewidth=1.0, alpha=0.7,
        linestyle="--", label="baseline",
    )
    ax.set_xlim(4, 21)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Non-mandatory miss ratio\n(fraction of baseline)")
    ax.tick_params(axis="both", labelsize="medium")
    ax.legend(
        title="trace", loc="best", ncol=2,
        fontsize="medium", title_fontsize="medium",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # In the notebook the title was applied after tight_layout; keep that order.
    plt.title("Miss reduction over time")
    inputs.save("miss_over_time", bbox_inches="tight")


GENERATORS: dict[str, Callable[[Inputs], None]] = {
    name: globals()[name] for name in ACTIVE_FIGURES
}


def generate_figures(inputs: Inputs, names: Sequence[str] = ("all",)) -> list[Path]:
    """Generate selected figures and return their output paths."""
    requested: list[str] = []
    for raw_name in names:
        name = raw_name.removesuffix(".png")
        if name == "all":
            requested.extend(ACTIVE_FIGURES)
        elif name in GENERATORS:
            requested.append(name)
        else:
            choices = ", ".join(ACTIVE_FIGURES)
            raise ValueError(f"unknown figure {raw_name!r}; choose from: {choices}, all")
    # Preserve request order but avoid accidentally doing expensive work twice.
    requested = list(dict.fromkeys(requested))
    _configure_plots()
    outputs = []
    for name in requested:
        print(f"generating {name}.png", flush=True)
        GENERATORS[name](inputs)
        outputs.append(inputs.output_dir / f"{name}.png")
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures", nargs="*", default=["all"],
        help="figure stem(s), optionally ending in .png; default: all",
    )
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument(
        "--expiry-dir", type=Path,
        help="expiry-event CSV directory (default: TRACE_ROOT/expiry, then archive fallback)",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sources", nargs="+", choices=ALL_SOURCES, default=ALL_SOURCES,
        help="trace simulator names to include; default: all paper traces",
    )
    parser.add_argument(
        "--list", action="store_true", help="list available figure stems and exit"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list:
        print("\n".join(ACTIVE_FIGURES))
        return 0
    inputs = Inputs(
        trace_root=args.trace_root.expanduser(),
        results_dir=args.results_dir.expanduser(),
        models_dir=args.models_dir.expanduser(),
        output_dir=args.output_dir.expanduser(),
        expiry_dir=(args.expiry_dir or (args.trace_root / "expiry")).expanduser(),
        sources=tuple(args.sources),
    )
    generate_figures(inputs, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

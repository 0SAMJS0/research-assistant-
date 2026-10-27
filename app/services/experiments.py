"""ML experiment log loading and direction-aware comparison with pandas. Files are parsed, never executed."""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from app.core.security import safe_child
from app.models.tools import ExperimentAnalysis, GroupStats, MetricComparison

ALLOWED_SUFFIXES = {".csv", ".json"}
MAX_ROWS = 200_000
MAX_FILES = 20
MAX_METRICS = 12

GROUP_CANDIDATES = ["model", "model_name", "method", "arch", "architecture", "variant"]
EXPERIMENT_CANDIDATES = ["experiment_id", "experiment", "exp_id", "run_id", "run", "run_name"]
SEED_CANDIDATES = ["seed", "random_seed"]
DATASET_CANDIDATES = ["dataset", "dataset_name", "data", "task"]
EPOCH_CANDIDATES = ["epoch", "epochs", "step", "global_step", "iteration", "iter"]
LOWER_IS_BETTER = re.compile(
    r"loss|error|err\b|_err|mse|mae|rmse|perplexity|ppl|latency|time|duration|nll|wer|cer|fid|regret|cost",
    re.IGNORECASE,
)
SOURCE_COL = "source_file"


class ExperimentDataError(ValueError):
    """The experiment file is unreadable or has no analyzable content."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") or f"col{i}"
        for i, c in enumerate(df.columns)
    ]
    return df.loc[:, ~df.columns.duplicated()]


def parse_experiment_bytes(data: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ExperimentDataError("Only .csv and .json experiment files are supported.")
    if not data.strip():
        raise ExperimentDataError("The experiment file is empty.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExperimentDataError("Experiment files must be UTF-8 text.") from exc
    try:
        if suffix == ".csv":
            df = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS)
        else:
            payload = json.loads(text)
            records = _json_records(payload)
            df = pd.json_normalize(records, sep="_")
    except ExperimentDataError:
        raise
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ExperimentDataError(f"Could not parse {suffix[1:].upper()} ({type(exc).__name__}).") from exc
    if df.empty:
        raise ExperimentDataError("The experiment file contains no rows.")
    if len(df) > MAX_ROWS:
        raise ExperimentDataError(f"Experiment files are limited to {MAX_ROWS} rows.")
    return _normalize_columns(df)


def _json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        list_values = [v for v in payload.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if len(list_values) == 1:
            records = list_values[0]
        elif "runs" in payload and isinstance(payload["runs"], list):
            records = payload["runs"]
        else:
            records = [payload]
    else:
        raise ExperimentDataError("JSON must be a list of records or an object containing one.")
    flat: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ExperimentDataError("Every JSON record must be an object.")
        # {"experiment_id": ..., "history": [{epoch..}, ...]} -> one row per history entry
        nested = [k for k, v in record.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if nested:
            base = {k: v for k, v in record.items() if k not in nested}
            for key in nested:
                flat.extend({**base, **entry} for entry in record[key])
        else:
            flat.append(record)
    return flat


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in columns), None)


def metric_direction(metric: str, overrides: dict[str, str] | None = None) -> Literal["lower", "higher"]:
    if overrides and overrides.get(metric) in ("lower", "higher"):
        return overrides[metric]  # type: ignore[return-value]
    return "lower" if LOWER_IS_BETTER.search(metric) else "higher"


def detect_metric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    metrics = []
    for col in df.columns:
        if col in exclude or col == "id" or col.endswith("_id") or col.startswith("unnamed"):
            continue
        series = df[col]
        non_null = series.dropna()
        if non_null.empty or series.dtype == bool:
            continue
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            metrics.append(col)
    return metrics


def _clean(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


def improvement_pct(value: float | None, baseline: float | None, direction: str) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    delta = (value - baseline) if direction == "higher" else (baseline - value)
    return round(delta / abs(baseline) * 100, 4)


def list_experiment_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES)


def load_experiments(directory: Path, files: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    names = files or list_experiment_files(directory)
    if not names:
        raise ExperimentDataError("No experiment files have been uploaded.")
    if len(names) > MAX_FILES:
        raise ExperimentDataError(f"At most {MAX_FILES} files can be analyzed at once.")
    frames = []
    for name in names:
        try:
            path = safe_child(directory, name)
        except ValueError as exc:
            raise ExperimentDataError(str(exc)) from exc
        if path.suffix.lower() not in ALLOWED_SUFFIXES or not path.is_file():
            raise ExperimentDataError(f"Experiment file not found: {name}")
        df = parse_experiment_bytes(path.read_bytes(), path.name)
        df[SOURCE_COL] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False), names


def analyze(
    directory: Path,
    files: list[str] | None = None,
    metrics: list[str] | None = None,
    model: str | None = None,
    seed: int | str | None = None,
    dataset: str | None = None,
    group_by: str | None = None,
    baseline: str | None = None,
    metric_directions: dict[str, str] | None = None,
) -> ExperimentAnalysis:
    df, names = load_experiments(directory, files)
    warnings: list[str] = []
    columns = list(df.columns)

    group_col = group_by.lower() if group_by else None
    if group_col and group_col not in columns:
        warnings.append(f"group_by column '{group_by}' not found; using automatic grouping.")
        group_col = None
    group_col = (
        group_col
        or _first_present(columns, GROUP_CANDIDATES)
        or _first_present(columns, EXPERIMENT_CANDIDATES)
    )
    exp_col = _first_present(columns, EXPERIMENT_CANDIDATES)
    seed_col = _first_present(columns, SEED_CANDIDATES)
    dataset_col = _first_present(columns, DATASET_CANDIDATES)
    epoch_col = _first_present(columns, EPOCH_CANDIDATES)

    filters: dict[str, str] = {}
    for label, col, value in (
        ("model", _first_present(columns, GROUP_CANDIDATES), model),
        ("seed", seed_col, seed),
        ("dataset", dataset_col, dataset),
    ):
        if value is None or value == "":
            continue
        if col is None:
            warnings.append(f"Cannot filter by {label}: no {label} column in the data.")
            continue
        df = df[df[col].astype(str).str.strip().str.lower() == str(value).strip().lower()]
        filters[label] = str(value)
    if df.empty:
        return ExperimentAnalysis(
            files=names,
            rows=0,
            columns=columns,
            group_by=group_col,
            run_key=[],
            epoch_column=epoch_col,
            filters_applied=filters,
            metric_columns=[],
            comparisons=[],
            warnings=[*warnings, "No rows match the filters."],
        )

    exclude = {c for c in (seed_col, epoch_col, exp_col, group_col, dataset_col, SOURCE_COL) if c}
    detected = detect_metric_columns(df, exclude)
    if metrics:
        wanted = [m.lower() for m in metrics]
        missing = [m for m in wanted if m not in detected]
        if missing:
            warnings.append(f"Metric column(s) not found or not numeric: {', '.join(missing)}.")
        detected = [m for m in wanted if m in detected]
    if not detected:
        warnings.append("No numeric metric columns detected.")
    if len(detected) > MAX_METRICS:
        warnings.append(f"Only the first {MAX_METRICS} metric columns are compared.")
        detected = detected[:MAX_METRICS]

    if group_col is None:
        group_col = "_all"
        df = df.assign(_all="all runs")
        warnings.append("No model/experiment column found; all rows are treated as one group.")
    df[group_col] = df[group_col].astype(str).where(df[group_col].notna(), "unknown")

    run_key = list(dict.fromkeys(c for c in (SOURCE_COL, group_col, exp_col, seed_col, dataset_col) if c))
    if epoch_col:
        df[epoch_col] = pd.to_numeric(df[epoch_col], errors="coerce")
        if (
            exp_col is None
            and seed_col is None
            and df.groupby(run_key, dropna=False)[epoch_col].apply(lambda s: s.duplicated().any()).any()
        ):
            warnings.append("Repeated epochs within a run: rows may come from several unlabeled runs.")
    groups_order = list(dict.fromkeys(df[group_col].tolist()))
    if baseline and baseline not in groups_order:
        warnings.append(f"Baseline '{baseline}' not found; using '{groups_order[0]}'.")
        baseline = None
    baseline = baseline or groups_order[0]

    comparisons = []
    for metric in detected:
        direction = metric_direction(metric, metric_directions)
        values = pd.to_numeric(df[metric], errors="coerce")
        frame = df.assign(_v=values).dropna(subset=["_v"])
        if frame.empty:
            warnings.append(f"Metric '{metric}' has no numeric values after filtering.")
            continue
        finals = []
        for key, run in frame.groupby(run_key, dropna=False, sort=False):
            run_sorted = run.sort_values(epoch_col) if epoch_col else run
            label = " / ".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
            finals.append({"group": run[group_col].iloc[0], "run": label, "final": run_sorted["_v"].iloc[-1]})
        finals_df = pd.DataFrame(finals)
        stats: list[GroupStats] = []
        for group in groups_order:
            gf = finals_df[finals_df["group"] == group]
            rows = frame[frame[group_col] == group]
            if gf.empty:
                continue
            best_idx = rows["_v"].idxmin() if direction == "lower" else rows["_v"].idxmax()
            best_row = rows.loc[best_idx]
            best_run = " / ".join(str(best_row[c]) for c in run_key)
            stats.append(
                GroupStats(
                    group=group,
                    runs=len(gf),
                    final_mean=_clean(gf["final"].mean()),
                    final_std=_clean(gf["final"].std(ddof=1)) if len(gf) > 1 else None,
                    best_observed=_clean(best_row["_v"]),
                    best_run=best_run,
                    best_epoch=_clean(best_row[epoch_col]) if epoch_col else None,
                    improvement_vs_baseline_pct=None,
                )
            )
        base_stats = next((s for s in stats if s.group == baseline), None)
        for s in stats:
            s.improvement_vs_baseline_pct = (
                (
                    0.0
                    if s.group == baseline
                    else improvement_pct(s.final_mean, base_stats.final_mean, direction)
                )
                if base_stats
                else None
            )
        ranked = [s for s in stats if s.final_mean is not None]
        winner = None
        if len(ranked) > 1:
            best = (min if direction == "lower" else max)(ranked, key=lambda s: s.final_mean)
            winner = best.group
        comparisons.append(
            MetricComparison(
                metric=metric,
                direction=direction,
                baseline=baseline if base_stats else None,
                groups=stats,
                winner=winner,
            )
        )
    return ExperimentAnalysis(
        files=names,
        rows=len(df),
        columns=[c for c in columns if c != SOURCE_COL],
        group_by=None if group_col == "_all" else group_col,
        run_key=run_key,
        epoch_column=epoch_col,
        filters_applied=filters,
        metric_columns=detected,
        comparisons=comparisons,
        warnings=warnings,
    )

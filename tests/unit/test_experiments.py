import json
import shutil

import pandas as pd
import pytest

from app.services.experiments import (
    ExperimentDataError,
    analyze,
    improvement_pct,
    metric_direction,
    parse_experiment_bytes,
)
from tests.helpers import SAMPLES

CSV = "synthetic_gnn_experiments.csv"
JSON = "synthetic_graphsage_runs.json"


@pytest.fixture
def exp_dir(tmp_path):
    d = tmp_path / "experiments"
    d.mkdir()
    shutil.copy(SAMPLES / CSV, d / CSV)
    shutil.copy(SAMPLES / JSON, d / JSON)
    return d


@pytest.mark.parametrize(
    "metric,direction",
    [
        ("val_loss", "lower"),
        ("train_error", "lower"),
        ("rmse", "lower"),
        ("perplexity", "lower"),
        ("val_accuracy", "higher"),
        ("f1", "higher"),
        ("auc", "higher"),
    ],
)
def test_metric_direction(metric, direction):
    assert metric_direction(metric) == direction


def test_metric_direction_override():
    assert metric_direction("val_loss", {"val_loss": "higher"}) == "higher"


def test_improvement_pct_is_direction_aware():
    assert improvement_pct(0.9, 0.8, "higher") == pytest.approx(12.5)
    assert improvement_pct(0.4, 0.5, "lower") == pytest.approx(20.0)
    assert improvement_pct(0.6, 0.5, "lower") == pytest.approx(-20.0)
    assert improvement_pct(1.0, 0.0, "higher") is None


def test_analysis_matches_independent_pandas_computation(exp_dir):
    result = analyze(exp_dir, files=[CSV], metrics=["val_accuracy", "val_loss"])
    df = pd.read_csv(SAMPLES / CSV)
    finals = df.sort_values("epoch").groupby(["model", "seed"]).tail(1)
    expected = finals.groupby("model")["val_accuracy"].agg(["mean", "std"])
    acc = next(c for c in result.comparisons if c.metric == "val_accuracy")
    assert acc.direction == "higher" and acc.baseline == "GCN"
    for g in acc.groups:
        assert g.runs == 3
        assert g.final_mean == pytest.approx(expected.loc[g.group, "mean"], abs=1e-6)
        assert g.final_std == pytest.approx(expected.loc[g.group, "std"], abs=1e-6)
        assert g.best_observed == pytest.approx(df[df.model == g.group]["val_accuracy"].max())
    gat = next(g for g in acc.groups if g.group == "GAT")
    base = expected.loc["GCN", "mean"]
    assert gat.improvement_vs_baseline_pct == pytest.approx(
        (expected.loc["GAT", "mean"] - base) / base * 100, abs=1e-3
    )
    loss = next(c for c in result.comparisons if c.metric == "val_loss")
    best_loss_model = finals.groupby("model")["val_loss"].mean().idxmin()
    assert loss.direction == "lower" and loss.winner == best_loss_model
    gat_loss = next(g for g in loss.groups if g.group == "GAT")
    assert gat_loss.best_observed == pytest.approx(df[df.model == "GAT"]["val_loss"].min())
    assert gat_loss.improvement_vs_baseline_pct > 0  # lower loss than GCN counts as an improvement


def test_metric_columns_detected_automatically_excluding_ids(exp_dir):
    result = analyze(exp_dir, files=[CSV])
    assert result.metric_columns == ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]
    assert result.epoch_column == "epoch" and result.group_by == "model"


def test_json_nested_history_is_flattened(exp_dir):
    result = analyze(exp_dir, files=[JSON], metrics=["val_accuracy"])
    group = result.comparisons[0].groups[0]
    assert group.group == "GraphSAGE" and group.runs == 2 and result.rows == 40


def test_filters_and_missing_filter_columns(exp_dir):
    result = analyze(exp_dir, files=[CSV], model="gat", seed=1, metrics=["val_accuracy"])
    assert result.filters_applied == {"model": "gat", "seed": "1"}
    assert result.rows == 20 and result.comparisons[0].groups[0].runs == 1
    assert result.comparisons[0].groups[0].final_std is None
    none = analyze(exp_dir, files=[CSV], dataset="does-not-exist")
    assert none.rows == 0 and "No rows match" in none.warnings[-1]


def test_baseline_and_missing_metric_warning(exp_dir):
    result = analyze(exp_dir, baseline="GAT", metrics=["val_accuracy", "bleu"])
    acc = result.comparisons[0]
    assert acc.baseline == "GAT"
    assert next(g for g in acc.groups if g.group == "GAT").improvement_vs_baseline_pct == 0.0
    assert next(g for g in acc.groups if g.group == "GCN").improvement_vs_baseline_pct < 0
    assert any("bleu" in w for w in result.warnings)
    assert {g.group for g in acc.groups} == {"GCN", "GAT", "GraphSAGE"}


def test_inconsistent_columns_are_tolerated(tmp_path):
    d = tmp_path / "e"
    d.mkdir()
    (d / "a.csv").write_text("run,acc,notes\nr1,0.5,ok\nr2,0.7,\nr3,n/a,bad\n")
    (d / "b.json").write_text(json.dumps([{"run": "r4", "acc": 0.9, "extra": "x"}]))
    result = analyze(d)
    assert result.group_by == "run" and result.metric_columns == ["acc"]
    groups = {g.group: g for g in result.comparisons[0].groups}
    assert groups["r4"].final_mean == 0.9 and "r3" not in groups


@pytest.mark.parametrize(
    "name,data,msg",
    [
        ("x.txt", b"a,b\n1,2\n", "Only .csv"),
        ("x.csv", b"   ", "empty"),
        ("x.json", b"{not json", "Could not parse"),
        ("x.json", b"42", "list of records"),
        ("x.csv", b"\xff\xfe\x00bad", "UTF-8"),
    ],
)
def test_parse_rejects_bad_files(name, data, msg):
    with pytest.raises(ExperimentDataError, match=msg):
        parse_experiment_bytes(data, name)


@pytest.mark.parametrize("name", ["../secrets.csv", "/etc/passwd", "..", "sub/dir.csv", ".hidden.csv"])
def test_path_traversal_is_rejected(exp_dir, name):
    with pytest.raises(ExperimentDataError):
        analyze(exp_dir, files=[name])


def test_no_files_uploaded(tmp_path):
    with pytest.raises(ExperimentDataError, match="No experiment files"):
        analyze(tmp_path)

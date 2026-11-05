import pytest

from noc_agent.cli import main
from noc_agent.evaluate import evaluate, pairwise_scores
from noc_agent.mock import MockLLM
from noc_agent.models import AlarmGroup
from noc_agent.pipeline import run_pipeline


def test_pairwise_scores_by_hand():
    labels = {"a": "x", "b": "x", "c": "y", "d": "y"}
    perfect = [AlarmGroup(group_id="1", alarm_ids=["a", "b"]),
               AlarmGroup(group_id="2", alarm_ids=["c", "d"])]
    assert pairwise_scores(perfect, labels)["f1"] == 1.0
    one_blob = [AlarmGroup(group_id="1", alarm_ids=["a", "b", "c", "d"])]
    s = pairwise_scores(one_blob, labels)
    assert s["recall"] == 1.0 and s["precision"] == pytest.approx(2 / 6)


def test_full_run_with_mock_llm(ds):
    run = run_pipeline(ds, "topology", MockLLM(ds.topology))
    m = evaluate(ds, run)
    assert m["root_cause_accuracy"] == 1.0
    assert m["location_accuracy"] == 1.0
    assert m["false_incidents"] == 0
    assert m["reported_incidents"] == len(ds.incidents)


def test_time_window_hurts_diagnosis(ds):
    good = evaluate(ds, run_pipeline(ds, "topology", MockLLM(ds.topology)))
    bad = evaluate(ds, run_pipeline(ds, "time_window", MockLLM(ds.topology)))
    assert bad["pairwise_f1"] < good["pairwise_f1"]
    assert bad["root_cause_accuracy"] < good["root_cause_accuracy"]


def test_cli_end_to_end(tmp_path, capsys):
    main(["generate", "--seed", "2", "--out", str(tmp_path / "data")])
    main(["run", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "run")])
    assert (tmp_path / "run" / "summary.md").exists()
    assert list((tmp_path / "run" / "incidents").glob("G-*.md"))
    main(["eval", "--seeds", "2", "--out", str(tmp_path / "eval")])
    assert "| `topology` |" in (tmp_path / "eval" / "results.md").read_text()
    assert "Grouping F1" in capsys.readouterr().out

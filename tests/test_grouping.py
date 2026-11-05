import json

import pytest

from noc_agent.evaluate import pairwise_scores
from noc_agent.grouping import (
    group_llm,
    group_time_window,
    group_topology,
    validate_groups,
)
from noc_agent.mock import MockLLM

from .conftest import incident


def _group_of(groups, alarm_id):
    return next(g.group_id for g in groups if alarm_id in g.alarm_ids)


def test_every_alarm_in_exactly_one_group(ds):
    for groups in (group_time_window(ds.alarms), group_topology(ds.alarms, ds.topology)):
        ids = [i for g in groups for i in g.alarm_ids]
        assert sorted(ids) == sorted(a.alarm_id for a in ds.alarms)


def test_time_window_merges_concurrent_faults(ds):
    # the generator starts a power outage next to a fiber cut at another site
    power, cut = incident(ds, "POWER_LOSS"), incident(ds, "FIBER_CUT")
    concurrent = [i for i in ds.incidents if i.root_cause.value == "FIBER_CUT"
                  and abs((i.started_at - power.started_at).total_seconds()) < 120]
    assert concurrent, "fixture should contain a concurrent cut"
    cut = concurrent[0]
    groups = group_time_window(ds.alarms)
    assert _group_of(groups, power.alarm_ids[0]) == _group_of(groups, cut.alarm_ids[0])


def test_topology_rules_recover_every_incident(ds):
    groups = group_topology(ds.alarms, ds.topology)
    for inc in ds.incidents:
        assert len({_group_of(groups, a) for a in inc.alarm_ids}) == 1, inc.incident_id
    assert pairwise_scores(groups, ds.truth_labels())["f1"] > 0.95


def test_low_severity_alarms_stay_alone(ds):
    groups = group_topology(ds.alarms, ds.topology)
    by_id = {a.alarm_id: a for a in ds.alarms}
    for g in groups:
        if len(g.alarm_ids) > 1:
            assert any(by_id[i].severity.rank >= 2 for i in g.alarm_ids)


def test_validate_groups_repairs_model_output():
    reply = {"groups": [
        {"alarm_ids": ["A1", "A2", "BOGUS"], "reason": "cut"},
        {"alarm_ids": ["A2", "A3"]},  # A2 already used
        "not a group",
    ]}
    parts = validate_groups(reply, ["A1", "A2", "A3", "A4"])
    assert [p for p, _ in parts] == [["A1", "A2"], ["A3"], ["A4"]]


@pytest.mark.parametrize("reply", [[], {"groups": "x"}, {"other": []}])
def test_validate_groups_rejects_wrong_shape(reply):
    with pytest.raises(ValueError):
        validate_groups(reply, ["A1"])


class _BrokenLLM:
    def chat(self, messages, tools=None):
        return {"role": "assistant", "content": "Sure! Here are the groups: (none)"}


def test_llm_grouping_falls_back_to_rules(ds):
    groups = group_llm(ds.alarms, ds.topology, _BrokenLLM())
    assert any(g.reason.startswith("fallback") for g in groups)
    assert pairwise_scores(groups, ds.truth_labels())["f1"] > 0.95


class _SplittingLLM:
    """Puts every alarm in its own group: hard links must still join the OTDR alarm."""

    def chat(self, messages, tools=None):
        ids = [json.loads(line)["id"] for line in messages[-1]["content"].splitlines()[1:]]
        return {"role": "assistant", "content": json.dumps({"groups": [{"alarm_ids": [i]}
                                                                        for i in ids]})}


def test_hard_links_override_the_model(ds):
    groups = group_llm(ds.alarms, ds.topology, _SplittingLLM())
    for a in ds.alarms:
        if "related_alarm" in a.details:
            assert _group_of(groups, a.alarm_id) == _group_of(groups, a.details["related_alarm"])


def test_llm_grouping_with_mock_matches_rules(ds):
    groups = group_llm(ds.alarms, ds.topology, MockLLM(ds.topology))
    assert pairwise_scores(groups, ds.truth_labels())["f1"] > 0.95

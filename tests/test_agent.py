import json
from datetime import timedelta

import pytest

from noc_agent.agent import DiagnosisAgent
from noc_agent.mock import MockLLM
from noc_agent.models import RootCause
from noc_agent.tools import Toolbox

from .conftest import incident


def _diagnose(ds, inc, llm, **kw):
    alarms = [a for a in ds.alarms if a.alarm_id in inc.alarm_ids]
    toolbox = Toolbox(ds.topology, ds.world, ds.alarms, max(a.raised_at for a in alarms))
    return DiagnosisAgent(llm, **kw).diagnose(alarms, toolbox, ds.topology)


@pytest.mark.parametrize("cause", ["FIBER_CUT", "POWER_LOSS", "HARDWARE_FAILURE", "PLANNED_WORK"])
def test_mock_agent_follows_runbooks(ds, cause):
    inc = incident(ds, cause)
    res = _diagnose(ds, inc, MockLLM(ds.topology))
    assert not res.fallback
    assert res.diagnosis.root_cause.value == cause
    assert res.diagnosis.location == inc.location
    if cause == "FIBER_CUT":
        assert res.diagnosis.break_km == pytest.approx(inc.break_km)
        assert {"run_otdr", "get_maintenance_windows"} <= {t["tool"] for t in res.trace}


def test_tools_do_not_see_the_future(ds):
    cut = next(c for c in ds.world.cuts if c.end is None)
    span = ds.topology.spans_in_cable(cut.cable_id)[0].span_id
    before = Toolbox(ds.topology, ds.world, ds.alarms, cut.start - timedelta(minutes=1))
    after = Toolbox(ds.topology, ds.world, ds.alarms, cut.start + timedelta(minutes=1))
    assert before.run_otdr(span)["break_detected"] is False
    assert after.run_otdr(span)["break_detected"] is True


def test_tool_errors_are_returned_not_raised(ds):
    tb = Toolbox(ds.topology, ds.world, ds.alarms, ds.start)
    assert "error" in tb.call("run_otdr", {"span_id": "SPN-999-9"})
    assert "error" in tb.call("no_such_tool", {})
    assert "error" in tb.call("get_power_status", {"wrong_arg": "x"})


class _Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools=None):
        self.seen.append(messages[-1])
        return self.replies.pop(0)


GOOD = json.dumps({"root_cause": "FIBER_CUT", "location": "CBL-001", "confidence": 0.8,
                   "summary": "cut", "evidence": [], "actions": []})


def test_agent_repairs_one_invalid_answer(ds):
    llm = _Scripted([{"content": "I think it is a fiber cut."},
                     {"content": f"<think>let me format it</think>```json\n{GOOD}\n```"}])
    res = _diagnose(ds, incident(ds, "FIBER_CUT"), llm)
    assert not res.fallback and res.diagnosis.root_cause == RootCause.FIBER_CUT
    assert "not a valid diagnosis" in llm.seen[-1]["content"]


def test_agent_escalates_after_second_invalid_answer(ds):
    llm = _Scripted([{"content": "no"}, {"content": '{"root_cause": "ALIENS"}'}])
    res = _diagnose(ds, incident(ds, "FIBER_CUT"), llm)
    assert res.fallback and res.diagnosis.root_cause == RootCause.UNKNOWN
    assert res.diagnosis.confidence == 0.0


def test_agent_stops_at_max_steps(ds):
    call = {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "get_node_info", "arguments": '{"node": "otn-ard-01"}'}}]}
    res = _diagnose(ds, incident(ds, "FIBER_CUT"), _Scripted([call] * 3), max_steps=3)
    assert res.fallback and res.steps == 3 and len(res.trace) == 3


def test_bad_tool_arguments_reach_the_model(ds):
    call = {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "run_otdr", "arguments": "{not json"}}]}
    llm = _Scripted([call, {"content": GOOD}])
    res = _diagnose(ds, incident(ds, "FIBER_CUT"), llm)
    assert "error" in res.trace[0]["result"]
    assert not res.fallback

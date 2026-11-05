"""A deterministic stand-in for the LLM.

It speaks the same protocol as a real model (tool calls, JSON answers) and follows
the runbooks in the diagnosis prompt step by step, so the whole pipeline runs and
is tested without a GPU or an API key. Its scores are a rule-based baseline, not
a measure of any model: run the benchmark with `--llm openai` for that.
"""

from __future__ import annotations

import json
from datetime import datetime
from itertools import count

from .context import parse_alarms_block
from .grouping import group_topology
from .llm import Message
from .models import Alarm, RootCause, Severity, Source, Topology
from .synth import POWER_ALARMS

OPTICAL_LIGHT_LOSS = {"OPTICAL_LOS", "OSC_LOS", "FIBER_BREAK_LOCATED"}


def _to_alarm(rec: dict) -> Alarm:
    source = {"pwr": Source.POWER, "rtr": Source.IP}.get(rec["node"][:3], Source.OPTICAL)
    return Alarm(
        alarm_id=rec["id"],
        raised_at=datetime.strptime(rec["time"], "%Y-%m-%d %H:%M:%S"),
        source=source,
        site=rec["site"],
        node=rec["node"],
        name=rec["name"],
        severity=Severity(rec["severity"]),
        resource=rec.get("resource"),
        span_id=rec.get("span"),
        details=rec.get("details", {}),
    )


class MockLLM:
    def __init__(self, topo: Topology):
        self.topo = topo
        self._ids = count(1)

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        if tools:
            return self._diagnose_turn(messages)
        return self._grouping_turn(messages)

    # ------------------------------------------------------------------ grouping

    def _grouping_turn(self, messages: list[Message]) -> Message:
        alarms = [_to_alarm(r) for r in parse_alarms_block(messages[-1]["content"])]
        groups = group_topology(alarms, self.topo)
        reply = {"groups": [{"alarm_ids": g.alarm_ids, "reason": g.reason} for g in groups]}
        return {"role": "assistant", "content": json.dumps(reply)}

    # ------------------------------------------------------------------ diagnosis

    def _call(self, *calls: tuple[str, dict]) -> Message:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"call_{next(self._ids)}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}
                for name, args in calls
            ],
        }

    @staticmethod
    def _final(cause: RootCause, location: str, confidence: float, summary: str,
               evidence: list[str], actions: list[str], break_km: float | None = None) -> Message:
        body = {"root_cause": cause.value, "location": location, "confidence": confidence,
                "summary": summary, "evidence": evidence, "actions": actions, "break_km": break_km}
        return {"role": "assistant", "content": json.dumps(body)}

    @staticmethod
    def _results(messages: list[Message]) -> dict[str, list[dict]]:
        names = {}
        for m in messages:
            for c in m.get("tool_calls") or []:
                names[c["id"]] = c["function"]["name"]
        out: dict[str, list[dict]] = {}
        for m in messages:
            if m.get("role") == "tool":
                out.setdefault(names.get(m.get("tool_call_id"), "?"), []).append(
                    json.loads(m["content"]))
        return out

    def _diagnose_turn(self, messages: list[Message]) -> Message:
        alarms = parse_alarms_block(messages[1]["content"])
        done = self._results(messages)

        def named(*names: str) -> list[dict]:
            return [a for a in alarms if a["name"] in names]

        def cite(recs: list[dict]) -> list[str]:
            return [f"{a['id']} {a['name']} on {a['node']}" for a in recs[:4]]

        power = named(*POWER_ALARMS)
        if power:
            site = power[0]["site"]
            if "get_power_status" not in done:
                return self._call(("get_power_status", {"site": site}))
            st = done["get_power_status"][-1]
            if st.get("mains") == "off":
                return self._final(
                    RootCause.POWER_LOSS, site, 0.9,
                    f"Site {site} lost mains power and runs on batteries "
                    f"(~{st.get('estimated_runtime_min')} min left).",
                    cite(power) + [f"get_power_status({site}): mains off, "
                                   f"{st.get('battery_voltage')} V"],
                    [f"Dispatch a mobile generator to {site}",
                     "Ask the utility for an outage ETA"])

        cards = named("CARD_FAILURE")
        if cards:
            c = cards[0]
            return self._final(
                RootCause.HARDWARE_FAILURE, c["node"], 0.85,
                f"Line card {c.get('resource')} failed on {c['node']}; the links on it are down.",
                cite(cards + named("LINK_DOWN")),
                [f"Replace the card in {c.get('resource')} of {c['node']}"])

        optical = named(*OPTICAL_LIGHT_LOSS)
        if optical:
            span = optical[0]["span"]
            if "get_span_info" not in done:
                return self._call(("get_span_info", {"span_id": span}))
            cable = done["get_span_info"][-1].get("cable_id", optical[0].get("cable", "?"))
            if "run_otdr" not in done:
                return self._call(("get_maintenance_windows", {"target": cable}),
                                  ("run_otdr", {"span_id": span}))
            active = [w for w in done["get_maintenance_windows"][-1].get("windows", [])
                      if w.get("active_now")]
            if active:
                w = active[0]
                return self._final(
                    RootCause.PLANNED_WORK, cable, 0.85,
                    f"Light loss on {cable} during planned works {w['window_id']}.",
                    cite(optical) + [f"{w['window_id']}: {w['description']}"],
                    ["No dispatch; confirm with the works owner that service is restored "
                     f"after {w['end']}"])
            otdr = done["run_otdr"][-1]
            located = named("FIBER_BREAK_LOCATED")
            if otdr.get("break_detected") or located:
                km = otdr.get("distance_km_from_a_end")
                if km is None:
                    km = float(located[0]["details"]["distance_km"])
                origin = otdr.get("a_end_site", "the A end")
                return self._final(
                    RootCause.FIBER_CUT, cable, 0.9,
                    f"Fiber cut on {cable}, about {km} km from {origin}.",
                    cite(optical) + [f"run_otdr({span}): break at {km} km"],
                    [f"Dispatch a fiber crew to {cable}, ~{km} km from {origin}",
                     "Check that traffic moved to the protection path"],
                    break_km=km)
            return self._final(RootCause.UNKNOWN, cable, 0.3,
                               "Light loss but a clean OTDR trace and no planned works.",
                               cite(optical), ["Escalate to the transmission engineer"])

        links = named("LINK_DOWN", "LINK_FLAP")
        if links:
            node = links[0]["node"]
            if "get_alarm_history" not in done:
                return self._call(("get_alarm_history", {"node": node}))
            counts = done["get_alarm_history"][-1].get("counts", {})
            flaps = counts.get("LINK_FLAP", 0) + counts.get("LINK_DOWN", 0)
            if flaps >= 5 or len(alarms) == 1:
                return self._final(RootCause.TRANSIENT, node, 0.7,
                                   f"{links[0].get('resource')} on {node} is flapping.",
                                   cite(links) + [f"{flaps} flaps in the last 24h"],
                                   ["Open a low-priority ticket to check the optics on that port"])
            return self._final(RootCause.UNKNOWN, node, 0.3, "Links down without a clear cause.",
                               cite(links), ["Escalate to the IP engineer"])

        first = alarms[0]
        return self._final(RootCause.TRANSIENT, first["node"], 0.6,
                           f"{first['name']} on {first['node']}; no service impact.",
                           cite(alarms), [])

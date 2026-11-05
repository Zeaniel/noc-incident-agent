"""Tools the diagnosis agent can call.

They answer from the simulated network (`World`) as it is at `as_of`, the time of
the newest alarm in the incident, so the agent can never read the future.
Swap this class for one backed by real inventory, OTDR and power APIs and the
agent does not change.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from .models import Alarm, Topology, World


def _fn(name: str, description: str, **params: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
                "required": list(params),
            },
        },
    }


TOOL_SCHEMAS = [
    _fn("get_node_info", "Site, role, optical spans and IP links of a network node.",
        node="node name, e.g. otn-ard-01"),
    _fn("get_span_info", "Ends, cable, length and carried IP links of an optical span, plus the "
        "other spans in the same cable.", span_id="span id, e.g. SPN-003-1"),
    _fn("run_otdr", "Run an OTDR trace on a span. Reports whether the fiber is broken and the "
        "distance of the break from the A end.", span_id="span id"),
    _fn("get_power_status", "Mains state, battery voltage and estimated battery runtime of a site.",
        site="site code, e.g. ARD"),
    _fn("get_maintenance_windows", "Planned works on a cable, span, site or node within 6 hours "
        "of now, with whether each is active now.", target="cable id, span id, site code or node"),
    _fn("get_alarm_history", "Alarm counts per alarm name on a node over the previous 24 hours.",
        node="node name"),
]


class Toolbox:
    def __init__(self, topo: Topology, world: World, alarms: list[Alarm], as_of: datetime):
        self.topo = topo
        self.world = world
        self.alarms = alarms
        self.as_of = as_of
        self._impl: dict[str, Callable[..., Any]] = {
            "get_node_info": self.get_node_info,
            "get_span_info": self.get_span_info,
            "run_otdr": self.run_otdr,
            "get_power_status": self.get_power_status,
            "get_maintenance_windows": self.get_maintenance_windows,
            "get_alarm_history": self.get_alarm_history,
        }

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def call(self, name: str, args: dict) -> Any:
        """Run a tool. Errors come back as data so the model can correct itself."""
        fn = self._impl.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except KeyError as e:
            return {"error": f"not found: {e}"}

    # ------------------------------------------------------------------ tools

    def get_node_info(self, node: str) -> dict:
        n = self.topo.node(node)
        if n is None:
            raise KeyError(node)
        return {
            "node": n.name,
            "site": n.site,
            "role": n.kind,
            "spans": [s.span_id for s in self.topo.spans_at_node(node)],
            "ip_links": [lk.link_id for lk in self.topo.links_at_node(node)],
        }

    def get_span_info(self, span_id: str) -> dict:
        s = self.topo.span(span_id)
        if s is None:
            raise KeyError(span_id)
        return {
            "span_id": s.span_id,
            "cable_id": s.cable_id,
            "a_end": f"{s.a_node} {s.a_port} ({s.a_site})",
            "z_end": f"{s.z_node} {s.z_port} ({s.z_site})",
            "length_km": s.length_km,
            "ip_links": [lk.link_id for lk in self.topo.links_over_span(span_id)],
            "same_cable_spans": [o.span_id for o in self.topo.spans_in_cable(s.cable_id)
                                 if o.span_id != span_id],
        }

    def run_otdr(self, span_id: str) -> dict:
        s = self.topo.span(span_id)
        if s is None:
            raise KeyError(span_id)
        for cut in self.world.cuts:
            active = cut.start <= self.as_of and (cut.end is None or self.as_of < cut.end)
            if cut.cable_id == s.cable_id and active:
                return {"span_id": span_id, "break_detected": True,
                        "distance_km_from_a_end": cut.break_km, "a_end_site": s.a_site,
                        "span_length_km": s.length_km}
        return {"span_id": span_id, "break_detected": False,
                "trace": f"clean to the far end, total loss {0.21 * s.length_km + 1.2:.1f} dB"}

    def get_power_status(self, site: str) -> dict:
        site = site.upper()
        if not any(s.code == site for s in self.topo.sites):
            raise KeyError(site)
        for o in self.world.outages:
            if o.site == site and o.start <= self.as_of < o.end:
                elapsed = (self.as_of - o.start).total_seconds() / 60
                left = max(0.0, o.battery_minutes - elapsed)
                volts = 44.0 + 9.5 * (left / o.battery_minutes)
                return {"site": site, "mains": "off", "on_battery_since": o.start.isoformat(),
                        "battery_voltage": round(volts, 1),
                        "estimated_runtime_min": round(left)}
        return {"site": site, "mains": "on", "battery_voltage": 54.1}

    def get_maintenance_windows(self, target: str) -> dict:
        targets = {target}
        span = self.topo.span(target)
        node = self.topo.node(target)
        if span:
            targets |= {span.cable_id, span.a_site, span.z_site}
        if node:
            targets.add(node.site)
        for s in self.topo.spans_in_cable(target):
            targets |= {s.a_site, s.z_site}
        lo, hi = self.as_of - timedelta(hours=6), self.as_of + timedelta(hours=6)
        windows = [
            {"window_id": w.window_id, "target": w.target, "start": w.start.isoformat(),
             "end": w.end.isoformat(), "active_now": w.start <= self.as_of <= w.end,
             "description": w.description}
            for w in self.world.maintenance
            if w.target in targets and w.start <= hi and w.end >= lo
        ]
        return {"target": target, "windows": windows}

    def get_alarm_history(self, node: str) -> dict:
        if self.topo.node(node) is None:
            raise KeyError(node)
        counts = Counter(self.world.history.get(node, {}))
        since = self.as_of - timedelta(hours=24)
        for a in self.alarms:
            if a.node == node and since <= a.raised_at < self.as_of:
                counts[a.name] += 1
        return {"node": node, "hours": 24, "counts": dict(counts)}

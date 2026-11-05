"""Data types shared by the simulator, the pipeline and the evaluator."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    WARNING = "WARNING"

    @property
    def rank(self) -> int:
        return {"CRITICAL": 3, "MAJOR": 2, "MINOR": 1, "WARNING": 0}[self.value]


class Source(str, Enum):
    """Which management system raised the alarm."""

    OPTICAL = "optical"
    IP = "ip"
    POWER = "power"


class Alarm(BaseModel):
    alarm_id: str
    raised_at: datetime
    source: Source
    site: str
    node: str
    name: str
    severity: Severity
    resource: str | None = None  # port, interface or slot the alarm is about
    span_id: str | None = None  # optical alarms carry the span they monitor
    details: dict[str, str] = Field(default_factory=dict)


class RootCause(str, Enum):
    FIBER_CUT = "FIBER_CUT"
    POWER_LOSS = "POWER_LOSS"
    HARDWARE_FAILURE = "HARDWARE_FAILURE"
    PLANNED_WORK = "PLANNED_WORK"
    TRANSIENT = "TRANSIENT"  # noise, flapping, nothing to dispatch
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- topology


class Site(BaseModel):
    code: str
    name: str


class Node(BaseModel):
    name: str
    site: str
    kind: Literal["otn", "router", "power"]


class Span(BaseModel):
    """One fiber pair between two optical nodes. Several spans can share a cable."""

    span_id: str
    cable_id: str
    a_node: str
    a_port: str
    a_site: str
    z_node: str
    z_port: str
    z_site: str
    length_km: float


class IpLink(BaseModel):
    """A router-to-router link carried over one or more optical spans."""

    link_id: str
    a_node: str
    a_port: str
    z_node: str
    z_port: str
    span_ids: list[str]


class Topology(BaseModel):
    sites: list[Site]
    nodes: list[Node]
    spans: list[Span]
    ip_links: list[IpLink]

    _nodes: dict[str, Node] = PrivateAttr(default_factory=dict)
    _spans: dict[str, Span] = PrivateAttr(default_factory=dict)
    _ports: dict[tuple[str, str], IpLink] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context) -> None:
        self._nodes = {n.name: n for n in self.nodes}
        self._spans = {s.span_id: s for s in self.spans}
        for link in self.ip_links:
            self._ports[(link.a_node, link.a_port)] = link
            self._ports[(link.z_node, link.z_port)] = link

    def node(self, name: str) -> Node | None:
        return self._nodes.get(name)

    def span(self, span_id: str) -> Span | None:
        return self._spans.get(span_id)

    def spans_in_cable(self, cable_id: str) -> list[Span]:
        return [s for s in self.spans if s.cable_id == cable_id]

    def spans_at_node(self, node: str) -> list[Span]:
        return [s for s in self.spans if node in (s.a_node, s.z_node)]

    def link_on_port(self, node: str, port: str | None) -> IpLink | None:
        if port is None:
            return None
        return self._ports.get((node, port))

    def links_at_node(self, node: str) -> list[IpLink]:
        return [lk for lk in self.ip_links if node in (lk.a_node, lk.z_node)]

    def links_over_span(self, span_id: str) -> list[IpLink]:
        return [lk for lk in self.ip_links if span_id in lk.span_ids]


# --------------------------------------------------------------------------- hidden state


class FiberCut(BaseModel):
    cable_id: str
    start: datetime
    end: datetime | None = None
    break_km: float  # distance from the A end of every span in the cable


class PowerOutage(BaseModel):
    site: str
    start: datetime
    end: datetime
    battery_minutes: float


class CardFailure(BaseModel):
    node: str
    slot: int
    start: datetime


class MaintenanceWindow(BaseModel):
    window_id: str
    target: str  # cable id or site code
    start: datetime
    end: datetime
    description: str


class World(BaseModel):
    """What is really happening in the network. Tools read it; the agent never sees it."""

    cuts: list[FiberCut] = Field(default_factory=list)
    outages: list[PowerOutage] = Field(default_factory=list)
    card_failures: list[CardFailure] = Field(default_factory=list)
    maintenance: list[MaintenanceWindow] = Field(default_factory=list)
    # alarm counts per node over the day before the dataset starts
    history: dict[str, dict[str, int]] = Field(default_factory=dict)


# --------------------------------------------------------------------------- ground truth


class TrueIncident(BaseModel):
    incident_id: str
    root_cause: RootCause
    location: str  # cable id, site code or node name, depending on the root cause
    alarm_ids: list[str]
    started_at: datetime
    break_km: float | None = None


class Dataset(BaseModel):
    seed: int
    start: datetime
    end: datetime
    topology: Topology
    world: World
    alarms: list[Alarm]
    incidents: list[TrueIncident]

    def truth_labels(self) -> dict[str, str]:
        """alarm_id -> incident id; every noise alarm gets a label of its own."""
        labels = {a.alarm_id: f"noise:{a.alarm_id}" for a in self.alarms}
        for inc in self.incidents:
            for aid in inc.alarm_ids:
                labels[aid] = inc.incident_id
        return labels

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "dataset.json").write_text(self.model_dump_json(indent=1), encoding="utf-8")
        # a flat alarm feed, the way a monitoring system would export it
        with (path / "alarms.jsonl").open("w", encoding="utf-8") as f:
            for a in self.alarms:
                f.write(a.model_dump_json() + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        text = (Path(path) / "dataset.json").read_text(encoding="utf-8")
        return cls.model_validate(json.loads(text))


# --------------------------------------------------------------------------- pipeline output


class AlarmGroup(BaseModel):
    group_id: str
    alarm_ids: list[str]
    reason: str = ""


class Diagnosis(BaseModel):
    root_cause: RootCause
    location: str
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    break_km: float | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            return min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

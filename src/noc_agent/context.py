"""Turn alarms into compact, topology-enriched records for prompts."""

from __future__ import annotations

import json

from .models import Alarm, Topology


def enrich(alarm: Alarm, topo: Topology) -> dict:
    """The alarm plus what the inventory knows about the resource it is on."""
    rec: dict = {
        "id": alarm.alarm_id,
        "time": alarm.raised_at.strftime("%Y-%m-%d %H:%M:%S"),
        "site": alarm.site,
        "node": alarm.node,
        "name": alarm.name,
        "severity": alarm.severity.value,
    }
    if alarm.resource:
        rec["resource"] = alarm.resource
    if alarm.span_id:
        span = topo.span(alarm.span_id)
        rec["span"] = alarm.span_id
        if span:
            rec["cable"] = span.cable_id
            rec["span_ends"] = f"{span.a_site}-{span.z_site}"
    link = topo.link_on_port(alarm.node, alarm.resource)
    if link:
        rec["ip_link"] = link.link_id
        rec["link_spans"] = link.span_ids
        rec["cable"] = topo.span(link.span_ids[0]).cable_id
    if alarm.details:
        rec["details"] = alarm.details
    return rec


def alarms_block(alarms: list[Alarm], topo: Topology) -> str:
    """One JSON object per line: easy for a model to read, easy for tests to parse."""
    return "\n".join(json.dumps(enrich(a, topo), separators=(",", ":")) for a in alarms)


def parse_alarms_block(text: str) -> list[dict]:
    """Inverse of `alarms_block`, tolerant of surrounding prose."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"id"' in line:
            out.append(json.loads(line))
    return out

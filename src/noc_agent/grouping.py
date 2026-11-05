"""Alarm grouping: which alarms belong to the same incident?

Three strategies, from naive to model-based:

- `time_window`: start a new group whenever the alarm stream goes quiet for a few
  minutes. This is how many NOC rules are written, and it fails as soon as two
  unrelated faults overlap in time.
- `topology`: union alarms that share a resource (span, cable, IP link, site power,
  line card) within a time window. Deterministic and cheap.
- `llm`: the model groups the significant alarms, with topology context in the
  prompt; its answer is validated and then stitched by hard links. If the reply
  cannot be used, that chunk falls back to `topology`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable

from . import prompts
from .context import alarms_block
from .llm import ChatModel, extract_json
from .models import Alarm, AlarmGroup, Severity, Topology
from .synth import POWER_ALARMS, port_slot

log = logging.getLogger(__name__)

DEFAULT_WINDOW_S = 900
# batteries can hold a site for most of an hour before it goes dark
KEY_WINDOW_S = {"pwr": 7200}
STRATEGIES = ("time_window", "topology", "llm")


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def sets(self) -> list[list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return list(out.values())


def _to_groups(sets: Iterable[Iterable[str]], alarms: list[Alarm], reason: str) -> list[AlarmGroup]:
    when = {a.alarm_id: a.raised_at for a in alarms}
    ordered = sorted((sorted(s, key=lambda i: (when[i], i)) for s in sets if s),
                     key=lambda s: when[s[0]])
    return [AlarmGroup(group_id=f"G-{i:03d}", alarm_ids=s, reason=reason)
            for i, s in enumerate(ordered, start=1)]


def _renumber(groups: list[AlarmGroup], alarms: list[Alarm]) -> list[AlarmGroup]:
    when = {a.alarm_id: a.raised_at for a in alarms}
    groups = sorted(groups, key=lambda g: min(when[i] for i in g.alarm_ids))
    for i, g in enumerate(groups, start=1):
        g.group_id = f"G-{i:03d}"
    return groups


def significant(alarm: Alarm) -> bool:
    return alarm.severity.rank >= Severity.MAJOR.rank


# --------------------------------------------------------------------------- time window


def group_time_window(alarms: list[Alarm], gap_s: float = 180) -> list[AlarmGroup]:
    sets: list[list[str]] = []
    last = None
    for a in sorted(alarms, key=lambda a: a.raised_at):
        if last is None or (a.raised_at - last).total_seconds() > gap_s:
            sets.append([])
        sets[-1].append(a.alarm_id)
        last = a.raised_at
    return _to_groups(sets, alarms, f"no gap longer than {gap_s:.0f}s")


# --------------------------------------------------------------------------- topology


def correlation_keys(alarm: Alarm, topo: Topology) -> set[str]:
    """Resources an alarm points at. Alarms sharing a key, close in time, are one incident."""
    keys: set[str] = set()
    if not significant(alarm):
        return keys
    if alarm.span_id:
        keys.add(f"span:{alarm.span_id}")
        span = topo.span(alarm.span_id)
        if span:
            keys.add(f"cable:{span.cable_id}")
    link = topo.link_on_port(alarm.node, alarm.resource)
    if link:
        keys.add(f"link:{link.link_id}")
        for sid in link.span_ids:
            keys.add(f"span:{sid}")
            keys.add(f"cable:{topo.span(sid).cable_id}")
    if alarm.name in POWER_ALARMS:
        keys.add(f"pwr:{alarm.site}")
    if alarm.name == "NODE_UNREACHABLE":
        # a dark node takes down every span and link that ends on it
        keys.add(f"pwr:{alarm.site}")
        keys |= {f"span:{s.span_id}" for s in topo.spans_at_node(alarm.node)}
        keys |= {f"link:{lk.link_id}" for lk in topo.links_at_node(alarm.node)}
    if alarm.name == "CARD_FAILURE" and (alarm.resource or "").startswith("slot "):
        slot = int(alarm.resource.split()[1])
        for lk in topo.links_at_node(alarm.node):
            port = lk.a_port if lk.a_node == alarm.node else lk.z_port
            if port_slot(port) == slot:
                keys.add(f"link:{lk.link_id}")
    return keys


def _hard_links(alarms: list[Alarm], uf: _UnionFind) -> None:
    """An alarm that names another alarm (e.g. an OTDR result) always joins it."""
    for a in alarms:
        ref = a.details.get("related_alarm")
        if ref in uf.parent:
            uf.union(ref, a.alarm_id)


def group_topology(alarms: list[Alarm], topo: Topology,
                   window_s: float = DEFAULT_WINDOW_S) -> list[AlarmGroup]:
    uf = _UnionFind(a.alarm_id for a in alarms)
    by_key: dict[str, list[Alarm]] = defaultdict(list)
    for a in alarms:
        for k in correlation_keys(a, topo):
            by_key[k].append(a)
    for key, members in by_key.items():
        window = KEY_WINDOW_S.get(key.split(":", 1)[0], window_s)
        members.sort(key=lambda a: a.raised_at)
        for prev, cur in zip(members, members[1:], strict=False):
            if (cur.raised_at - prev.raised_at).total_seconds() <= window:
                uf.union(prev.alarm_id, cur.alarm_id)
    _hard_links(alarms, uf)
    return _to_groups(uf.sets(), alarms, "shared topology resource")


# --------------------------------------------------------------------------- llm


def _chunks(alarms: list[Alarm], gap_s: float, max_size: int) -> list[list[Alarm]]:
    chunks: list[list[Alarm]] = []
    last = None
    for a in sorted(alarms, key=lambda a: a.raised_at):
        if (last is None or (a.raised_at - last).total_seconds() > gap_s
                or len(chunks[-1]) >= max_size):
            chunks.append([])
        chunks[-1].append(a)
        last = a.raised_at
    return chunks


def validate_groups(reply: object, alarm_ids: list[str]) -> list[tuple[list[str], str]]:
    """Make a model's grouping usable: every id exactly once, unknown ids dropped.

    Raises ValueError when the reply does not have the expected shape at all.
    """
    if not isinstance(reply, dict) or not isinstance(reply.get("groups"), list):
        raise ValueError("reply has no 'groups' list")
    valid = set(alarm_ids)
    seen: set[str] = set()
    out: list[tuple[list[str], str]] = []
    for g in reply["groups"]:
        if not isinstance(g, dict):
            continue
        ids = [i for i in g.get("alarm_ids") or [] if i in valid and i not in seen]
        seen.update(ids)
        if ids:
            out.append((ids, str(g.get("reason", ""))[:200]))
    out += [([i], "not grouped by the model") for i in alarm_ids if i not in seen]
    return out


def group_llm(alarms: list[Alarm], topo: Topology, llm: ChatModel,
              chunk_gap_s: float = 3600, max_chunk: int = 80) -> list[AlarmGroup]:
    # housekeeping alarms never start an incident: keep them out of the prompt (and the bill)
    minor = [a for a in alarms if not significant(a)]
    major = [a for a in alarms if significant(a)]
    system = prompts.load("grouping")

    groups: list[AlarmGroup] = [AlarmGroup(group_id="", alarm_ids=[a.alarm_id], reason="low severity")
                                for a in minor]
    for chunk in _chunks(major, chunk_gap_s, max_chunk):
        ids = [a.alarm_id for a in chunk]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Alarms, one JSON object per line:\n"
                                        + alarms_block(chunk, topo)},
        ]
        try:
            reply = llm.chat(messages)
            parts = validate_groups(extract_json(reply.get("content") or ""), ids)
            groups += [AlarmGroup(group_id="", alarm_ids=p, reason=r) for p, r in parts]
        except Exception as e:  # noqa: BLE001 - any failure means: use the rules for this chunk
            log.warning("LLM grouping failed for %d alarms (%s); using topology rules", len(ids), e)
            for g in group_topology(chunk, topo):
                g.reason = f"fallback: {g.reason}"
                groups.append(g)

    # deterministic guardrail: hard links win over the model, even across chunks
    uf = _UnionFind(a.alarm_id for a in alarms)
    for g in groups:
        for other in g.alarm_ids[1:]:
            uf.union(g.alarm_ids[0], other)
    _hard_links(alarms, uf)
    reason = {i: g.reason for g in groups for i in g.alarm_ids}
    merged = [AlarmGroup(group_id="", alarm_ids=s, reason=reason[s[0]]) for s in uf.sets()]
    for g in merged:
        g.alarm_ids.sort()
    return _renumber(merged, alarms)


def group_alarms(strategy: str, alarms: list[Alarm], topo: Topology,
                 llm: ChatModel | None = None) -> list[AlarmGroup]:
    if strategy == "time_window":
        return group_time_window(alarms)
    if strategy == "topology":
        return group_topology(alarms, topo)
    if strategy == "llm":
        if llm is None:
            raise ValueError("the llm strategy needs a model")
        return group_llm(alarms, topo, llm)
    raise ValueError(f"unknown grouping strategy {strategy!r}; choose from {STRATEGIES}")

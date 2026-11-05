"""Incident reports the way a NOC shift lead wants to read them."""

from __future__ import annotations

from .models import Topology
from .pipeline import IncidentResult, RunResult

_TITLE = {
    "FIBER_CUT": "Fiber cut",
    "POWER_LOSS": "Site power loss",
    "HARDWARE_FAILURE": "Hardware failure",
    "PLANNED_WORK": "Planned works",
    "UNKNOWN": "Unclassified incident",
}


def incident_markdown(res: IncidentResult, topo: Topology) -> str:
    d = res.diagnosis
    first = min(a.raised_at for a in res.alarms)
    last = max(a.raised_at for a in res.alarms)
    sites = sorted({a.site for a in res.alarms})
    links = sorted({lk.link_id for a in res.alarms
                    if (lk := topo.link_on_port(a.node, a.resource))})
    lines = [
        f"# {_TITLE.get(d.root_cause.value, d.root_cause.value)}: {d.location}",
        "",
        f"**{res.group.group_id}** · {first:%Y-%m-%d %H:%M:%S} → {last:%H:%M:%S} · "
        f"{len(res.alarms)} alarms · sites {', '.join(sites)} · confidence {d.confidence:.0%}",
        "",
        d.summary,
        "",
    ]
    if d.break_km is not None:
        ends = sorted({(s.a_site, s.z_site) for a in res.alarms
                       if a.span_id and (s := topo.span(a.span_id)) and s.cable_id == d.location})
        where = f"{ends[0][0]} (A end of {ends[0][0]}-{ends[0][1]})" if ends else "the A end"
        lines += [f"**Break location:** ~{d.break_km} km from {where}", ""]
    if links:
        lines += [f"**IP links affected:** {', '.join(links)}", ""]
    if d.actions:
        lines += ["## Actions", *[f"- {a}" for a in d.actions], ""]
    if d.evidence:
        lines += ["## Evidence", *[f"- {e}" for e in d.evidence], ""]
    if res.agent and res.agent.trace:
        lines += ["## Tool calls", ""]
        lines += [f"- `{t['tool']}({', '.join(f'{k}={v}' for k, v in (t['args'] or {}).items())})`"
                  for t in res.agent.trace]
        lines.append("")
    lines += ["## Alarms", "", "| Time | Alarm | Node | Resource | Name | Severity |",
              "|---|---|---|---|---|---|"]
    for a in res.alarms:
        lines.append(f"| {a.raised_at:%H:%M:%S} | {a.alarm_id} | {a.node} | {a.resource or ''} "
                     f"| {a.name} | {a.severity.value} |")
    return "\n".join(lines) + "\n"


def run_summary_markdown(run: RunResult, metrics: dict | None = None) -> str:
    actionable = [r for r in run.incidents if r.actionable]
    lines = [
        f"# Run summary (`{run.strategy}` grouping)",
        "",
        f"{len(run.incidents)} groups, {len(actionable)} reported as incidents, "
        f"{run.llm_calls} LLM calls ({run.llm_seconds:.1f} s).",
        "",
        "| Group | Root cause | Location | Alarms | Confidence |",
        "|---|---|---|---:|---:|",
    ]
    for r in actionable:
        d = r.diagnosis
        lines.append(f"| {r.group.group_id} | {d.root_cause.value} | {d.location} | "
                     f"{len(r.alarms)} | {d.confidence:.0%} |")
    if metrics:
        lines += [
            "",
            "## Against ground truth",
            "",
            f"- Grouping pairwise F1: {metrics['pairwise_f1']:.2f}",
            f"- Root-cause accuracy: {metrics['root_cause_accuracy']:.0%}, "
            f"location accuracy: {metrics['location_accuracy']:.0%}",
            f"- False incidents: {metrics['false_incidents']}",
            "",
            "| Incident | True cause | True location | Group | Exact | Predicted | Location |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in metrics["per_incident"]:
            lines.append(f"| {p['incident']} | {p['true_cause']} | {p['true_location']} | "
                         f"{p['group']} | {'yes' if p['exact_group'] else 'no'} | "
                         f"{p['pred_cause']} | {p['pred_location']} |")
    return "\n".join(lines) + "\n"

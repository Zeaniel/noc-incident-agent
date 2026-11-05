"""Score a run against the simulator's ground truth."""

from __future__ import annotations

from collections import Counter
from math import comb

from .models import AlarmGroup, Dataset, RootCause
from .pipeline import RunResult


def pairwise_scores(groups: list[AlarmGroup], labels: dict[str, str]) -> dict[str, float]:
    """Precision/recall over alarm pairs: is each pair put together iff it truly belongs together?"""
    pred = Counter()
    true = Counter(labels.values())
    both = Counter()
    for g in groups:
        pred[g.group_id] = len(g.alarm_ids)
        for lbl, n in Counter(labels[i] for i in g.alarm_ids).items():
            both[(g.group_id, lbl)] = n
    tp = sum(comb(n, 2) for n in both.values())
    pred_pairs = sum(comb(n, 2) for n in pred.values())
    true_pairs = sum(comb(n, 2) for n in true.values())
    p = tp / pred_pairs if pred_pairs else 1.0
    r = tp / true_pairs if true_pairs else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate(ds: Dataset, run: RunResult) -> dict:
    groups = [r.group for r in run.incidents]
    scores = pairwise_scores(groups, ds.truth_labels())
    where = {aid: idx for idx, r in enumerate(run.incidents) for aid in r.group.alarm_ids}

    exact = cause_ok = location_ok = 0
    km_errors: list[float] = []
    per_incident = []
    for t in ds.incidents:
        # the predicted group holding most of this incident's alarms speaks for it
        best = Counter(where[a] for a in t.alarm_ids).most_common(1)[0][0]
        res = run.incidents[best]
        d = res.diagnosis
        is_exact = set(res.group.alarm_ids) == set(t.alarm_ids)
        exact += is_exact
        cause_ok += d.root_cause == t.root_cause
        location_ok += d.location.strip().lower() == t.location.lower()
        if t.break_km is not None and d.root_cause == RootCause.FIBER_CUT and d.break_km is not None:
            km_errors.append(abs(d.break_km - t.break_km))
        per_incident.append({
            "incident": t.incident_id, "true_cause": t.root_cause.value, "true_location": t.location,
            "group": res.group.group_id, "exact_group": is_exact,
            "pred_cause": d.root_cause.value, "pred_location": d.location,
        })

    incident_alarms = {a for t in ds.incidents for a in t.alarm_ids}
    false_incidents = sum(
        1 for r in run.incidents if r.actionable and not set(r.group.alarm_ids) & incident_alarms)
    n = len(ds.incidents) or 1
    return {
        "strategy": run.strategy,
        "pairwise_precision": scores["precision"],
        "pairwise_recall": scores["recall"],
        "pairwise_f1": scores["f1"],
        "exact_groups": exact / n,
        "root_cause_accuracy": cause_ok / n,
        "location_accuracy": location_ok / n,
        "mean_break_error_km": sum(km_errors) / len(km_errors) if km_errors else None,
        "false_incidents": false_incidents,
        "reported_incidents": sum(1 for r in run.incidents if r.actionable),
        "true_incidents": len(ds.incidents),
        "llm_calls": run.llm_calls,
        "llm_seconds": run.llm_seconds,
        "fallbacks": run.fallbacks,
        "per_incident": per_incident,
    }


METRICS = [
    ("pairwise_f1", "Grouping F1", "{:.2f}"),
    ("exact_groups", "Exact groups", "{:.0%}"),
    ("root_cause_accuracy", "Root cause", "{:.0%}"),
    ("location_accuracy", "Location", "{:.0%}"),
    ("false_incidents", "False incidents", "{:.1f}"),
    ("llm_calls", "LLM calls", "{:.0f}"),
]


def average(results: list[dict]) -> dict:
    out = {"strategy": results[0]["strategy"], "runs": len(results)}
    for key, _, _ in METRICS:
        out[key] = sum(r[key] for r in results) / len(results)
    return out


def markdown_table(rows: list[dict]) -> str:
    head = "| Grouping | " + " | ".join(label for _, label, _ in METRICS) + " |"
    sep = "|---|" + "---:|" * len(METRICS)
    lines = [head, sep]
    for r in rows:
        cells = [fmt.format(r[key]) for key, _, fmt in METRICS]
        lines.append(f"| `{r['strategy']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)

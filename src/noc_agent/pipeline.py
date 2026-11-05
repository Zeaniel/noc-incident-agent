"""Alarms in, diagnosed incidents out."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .agent import AgentResult, DiagnosisAgent
from .grouping import group_alarms, significant
from .llm import ChatModel, CountingLLM
from .models import Alarm, AlarmGroup, Dataset, Diagnosis, RootCause
from .tools import Toolbox


@dataclass
class IncidentResult:
    group: AlarmGroup
    alarms: list[Alarm]
    diagnosis: Diagnosis
    agent: AgentResult | None = None  # None when triage closed it without the agent

    @property
    def actionable(self) -> bool:
        return self.diagnosis.root_cause != RootCause.TRANSIENT


@dataclass
class RunResult:
    strategy: str
    incidents: list[IncidentResult]
    llm_calls: int = 0
    llm_seconds: float = 0.0
    fallbacks: dict[str, int] = field(default_factory=dict)


def triage(alarms: list[Alarm]) -> Diagnosis:
    node = Counter(a.node for a in alarms).most_common(1)[0][0]
    return Diagnosis(
        root_cause=RootCause.TRANSIENT,
        location=node,
        confidence=0.9,
        summary="Only low-severity alarms; no incident opened.",
        evidence=[f"{a.alarm_id} {a.name} ({a.severity.value})" for a in alarms],
    )


def run_pipeline(ds: Dataset, strategy: str, llm: ChatModel, max_steps: int = 8) -> RunResult:
    counting = CountingLLM(llm)
    groups = group_alarms(strategy, ds.alarms, ds.topology, counting)
    by_id = {a.alarm_id: a for a in ds.alarms}
    agent = DiagnosisAgent(counting, max_steps=max_steps)

    results: list[IncidentResult] = []
    for g in groups:
        alarms = [by_id[i] for i in g.alarm_ids]
        # only groups with a MAJOR or CRITICAL alarm are worth an agent run
        if not any(significant(a) for a in alarms):
            results.append(IncidentResult(g, alarms, triage(alarms)))
            continue
        toolbox = Toolbox(ds.topology, ds.world, ds.alarms,
                          as_of=max(a.raised_at for a in alarms))
        res = agent.diagnose(alarms, toolbox, ds.topology)
        results.append(IncidentResult(g, alarms, res.diagnosis, res))

    return RunResult(
        strategy=strategy,
        incidents=results,
        llm_calls=counting.calls,
        llm_seconds=counting.seconds,
        fallbacks={
            "grouping": sum(1 for g in groups if g.reason.startswith("fallback")),
            "agent": sum(1 for r in results if r.agent and r.agent.fallback),
        },
    )

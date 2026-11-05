"""Tool-calling diagnosis agent.

A plain ReAct-style loop over the OpenAI tool-calling format: the model asks for
tools, sees their results, and ends with a JSON diagnosis that is validated with
pydantic. It gets one chance to repair an invalid answer; after that, or when
it runs out of steps, the incident is escalated as UNKNOWN instead of guessed.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from . import prompts
from .context import alarms_block
from .llm import ChatModel, extract_json
from .models import Alarm, Diagnosis, RootCause, Topology
from .tools import Toolbox


@dataclass
class AgentResult:
    diagnosis: Diagnosis
    steps: int
    fallback: bool
    trace: list[dict] = field(default_factory=list)  # every tool call with its result


def escalation(alarms: list[Alarm], why: str) -> Diagnosis:
    site = Counter(a.site for a in alarms).most_common(1)[0][0]
    return Diagnosis(
        root_cause=RootCause.UNKNOWN,
        location=site,
        confidence=0.0,
        summary=f"No reliable diagnosis ({why}); escalate to an engineer.",
        evidence=[f"{a.alarm_id} {a.name} on {a.node}" for a in alarms[:10]],
        actions=["Escalate to the on-call transmission engineer"],
    )


class DiagnosisAgent:
    def __init__(self, llm: ChatModel, max_steps: int = 8):
        self.llm = llm
        self.max_steps = max_steps

    def diagnose(self, alarms: list[Alarm], toolbox: Toolbox, topo: Topology) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": prompts.load("diagnose")},
            {"role": "user", "content": "Incident alarms, one JSON object per line:\n"
                                        + alarms_block(alarms, topo)
                                        + "\n\nInvestigate with the tools, then reply with the JSON "
                                          "diagnosis."},
        ]
        trace: list[dict] = []
        repaired = False
        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(messages, tools=toolbox.schemas)
            calls = reply.get("tool_calls") or []
            assistant = {"role": "assistant", "content": reply.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)

            if calls:
                for call in calls:
                    fn = call.get("function") or {}
                    name = fn.get("name", "")
                    raw = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        args = None
                    if isinstance(args, dict):
                        result = toolbox.call(name, args)
                    else:
                        result = {"error": "arguments must be a JSON object"}
                    trace.append({"tool": name, "args": args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                     "name": name, "content": json.dumps(result, default=str)})
                continue

            try:
                diagnosis = Diagnosis.model_validate(extract_json(reply.get("content") or ""))
                return AgentResult(diagnosis, step, False, trace)
            except ValueError as e:  # pydantic's ValidationError is a ValueError too
                if repaired:
                    return AgentResult(escalation(alarms, "invalid answer twice"), step, True, trace)
                repaired = True
                messages.append({"role": "user", "content":
                                 f"That reply was not a valid diagnosis ({str(e)[:300]}). Reply with "
                                 "only the JSON object described in the instructions."})
        return AgentResult(escalation(alarms, f"no answer in {self.max_steps} steps"),
                           self.max_steps, True, trace)

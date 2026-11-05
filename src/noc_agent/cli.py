"""Command line: generate data, run the pipeline, benchmark strategies."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .evaluate import average, evaluate, markdown_table
from .grouping import STRATEGIES
from .llm import ChatModel, OpenAICompatibleLLM
from .mock import MockLLM
from .models import Dataset, Topology
from .pipeline import run_pipeline
from .report import incident_markdown, run_summary_markdown
from .synth import generate_dataset


def make_llm(kind: str, topo: Topology) -> ChatModel:
    if kind == "mock":
        return MockLLM(topo)
    if kind == "openai":
        return OpenAICompatibleLLM.from_env()
    raise ValueError(kind)


def _dataset(args: argparse.Namespace) -> Dataset:
    if args.data:
        return Dataset.load(args.data)
    return generate_dataset(seed=args.seed)


def cmd_generate(args: argparse.Namespace) -> None:
    ds = generate_dataset(seed=args.seed, n_sites=args.sites, hours=args.hours, noise=args.noise)
    path = ds.save(args.out)
    print(f"{len(ds.alarms)} alarms, {len(ds.incidents)} incidents -> {path}")
    for inc in ds.incidents:
        print(f"  {inc.incident_id}  {inc.started_at:%H:%M:%S}  {inc.root_cause.value:<16} "
              f"{inc.location:<12} {len(inc.alarm_ids)} alarms")


def cmd_run(args: argparse.Namespace) -> None:
    ds = _dataset(args)
    run = run_pipeline(ds, args.grouping, make_llm(args.llm, ds.topology))
    metrics = evaluate(ds, run)
    out = Path(args.out or f"runs/{args.grouping}")
    (out / "incidents").mkdir(parents=True, exist_ok=True)
    for r in run.incidents:
        if r.actionable:
            (out / "incidents" / f"{r.group.group_id}.md").write_text(
                incident_markdown(r, ds.topology), encoding="utf-8")
    (out / "summary.md").write_text(run_summary_markdown(run, metrics), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")

    print(f"{len(run.incidents)} groups, {metrics['reported_incidents']} incidents reported "
          f"({metrics['true_incidents']} real), {run.llm_calls} LLM calls")
    for r in run.incidents:
        if r.actionable:
            d = r.diagnosis
            print(f"  {r.group.group_id}  {d.root_cause.value:<16} {d.location:<12} "
                  f"{len(r.alarms):>3} alarms  conf {d.confidence:.2f}")
    print(f"grouping F1 {metrics['pairwise_f1']:.2f} | root cause "
          f"{metrics['root_cause_accuracy']:.0%} | location {metrics['location_accuracy']:.0%}")
    print(f"reports written to {out}")


def cmd_eval(args: argparse.Namespace) -> None:
    strategies = args.grouping.split(",")
    seeds = range(args.start_seed, args.start_seed + args.seeds)
    datasets = [generate_dataset(seed=s) for s in seeds]
    rows, raw = [], []
    for strategy in strategies:
        results = [evaluate(ds, run_pipeline(ds, strategy, make_llm(args.llm, ds.topology)))
                   for ds in datasets]
        raw += results
        rows.append(average(results))
    table = markdown_table(rows)
    print(f"{len(datasets)} synthetic datasets (seeds {seeds.start}-{seeds.stop - 1}), "
          f"LLM: {args.llm}\n")
    print(table)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.md").write_text(table + "\n", encoding="utf-8")
        (out / "results.json").write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="noc-agent", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic dataset")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--sites", type=int, default=12)
    g.add_argument("--hours", type=float, default=6)
    g.add_argument("--noise", type=int, default=40)
    g.add_argument("--out", default="data/sample")
    g.set_defaults(fn=cmd_generate)

    llm_help = "mock = offline rule-based stand-in; openai = any OpenAI-compatible server (env vars)"
    r = sub.add_parser("run", help="group and diagnose one dataset, write incident reports")
    r.add_argument("--data", help="dataset folder (default: generate one from --seed)")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--grouping", choices=STRATEGIES, default="topology")
    r.add_argument("--llm", choices=["mock", "openai"], default="mock", help=llm_help)
    r.add_argument("--out")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="benchmark grouping strategies over several datasets")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--start-seed", type=int, default=1)
    e.add_argument("--grouping", default=",".join(STRATEGIES))
    e.add_argument("--llm", choices=["mock", "openai"], default="mock", help=llm_help)
    e.add_argument("--out")
    e.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()

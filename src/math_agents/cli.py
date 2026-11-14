import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from .backend import ChatBackend, DemoBackend
from .data import load_local, prepare
from .execution import DemoExecutor, DockerExecutor
from .grading import grade
from .metrics import summarize, usage_metrics
from .workflow import Runner
from .optimized import OptimizedRunner
from .consensus import ConsensusRunner

RUNNERS = {"baseline": Runner, "optimized": OptimizedRunner, "consensus": ConsensusRunner}
_worker = {}


def setup_worker(args, prices):
    """Per-process model client and sandbox handle (also used inline when --workers 1)."""
    if args.demo:
        backend, executor = DemoBackend(), DemoExecutor()
    else:
        backend, executor = ChatBackend(args.max_tokens, args.temperature), DockerExecutor()
    _worker.update(args=args, prices=prices, backend=backend, executor=executor)


def solve(task):
    problem, mode = task
    args, prices = _worker["args"], _worker["prices"]
    runner = RUNNERS[args.workflow](_worker["backend"], _worker["executor"], args.max_rounds)
    started = time.monotonic()
    state, status, error = {}, "ok", None
    try:
        # Only problem text crosses the agent boundary. Never pass gold.
        state = runner.run(mode, problem["problem"])
    except Exception as exc:
        status, error = "error", type(exc).__name__
    prediction = state.get("answer", "")
    return {"id": problem["id"], "dataset": problem["dataset"], "mode": mode,
            "demo": args.demo, "workflow": args.workflow, "answer": prediction, "gold": problem["answer"],
            "status": status, "error": error, "rounds": runner.rounds,
            "accepted": state.get("accepted"), "candidates": state.get("candidates"),
            "latency_seconds": time.monotonic() - started,
            **grade(prediction, problem["answer"], problem["dataset"]),
            **usage_metrics(runner.trace, prices), "trace": runner.trace}


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main():
    parser = argparse.ArgumentParser(description="LangGraph mathematics experiment lab")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("prepare")
    download.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    download.add_argument("--output", required=True)
    download.add_argument("--revision", default="main")
    run = sub.add_parser("run")
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True, help="New directory; never overwrite an earlier run")
    run.add_argument("--modes", nargs="+", choices=["direct", "single", "multi"], default=["direct", "single", "multi"])
    run.add_argument("--limit", type=positive)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-rounds", type=positive, default=3)
    run.add_argument("--max-tokens", type=positive, default=4096)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--workflow", choices=list(RUNNERS), default="consensus")
    run.add_argument("--workers", type=positive, default=1, help="Problems solved in parallel (separate processes)")
    run.add_argument("--demo", action="store_true", help="Scripted offline smoke test; NOT benchmark accuracy")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.output, args.revision)
        return
    problems = load_local(args.data, args.limit, args.seed)
    if not problems:
        parser.error("Empty dataset")
    if args.demo and any(r["dataset"] != "demo" for r in problems):
        parser.error("--demo only accepts the synthetic demo dataset")
    prices = {k: float(os.environ[env]) if os.getenv(env) else None for k, env in
              [("input", "MATH_INPUT_PRICE"), ("output", "MATH_OUTPUT_PRICE"), ("cached", "MATH_CACHED_PRICE")]}
    if any(v is not None and v < 0 for v in prices.values()):
        parser.error("Token prices must be nonnegative")
    if args.demo:
        prices = dict.fromkeys(prices, 0.0)
    else:
        if not os.getenv("MATH_MODEL") or not os.getenv("MATH_API_KEY"):
            parser.error("Set MATH_MODEL and MATH_API_KEY, or use --demo")
        if any(m != "direct" for m in args.modes):
            DockerExecutor().check()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runner_type = RUNNERS[args.workflow]
    # The consensus workflow replies in free text; JSON mode applies to the older workflows only.
    json_mode = (not args.demo and args.workflow != "consensus"
                 and os.getenv("MATH_JSON_MODE", "false").lower() == "true")
    manifest = {**vars(args), "model": DemoBackend.model if args.demo else os.environ["MATH_MODEL"],
                "json_mode": json_mode,
                "role_models": {} if args.demo or args.workflow != "consensus" else {
                    k: os.environ[k] for k in ("MATH_REASONER_MODEL", "MATH_CODER_MODEL", "MATH_REFEREE_MODEL")
                    if os.getenv(k, "").strip()},
                "api_uses_proxy": bool(not args.demo and os.getenv("MATH_PROXY_URL", "").strip()),
                "prices_per_million_usd": prices, "prompts": runner_type.prompts,
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "selected_ids": [(r["dataset"], r["id"]) for r in problems],
                "packages": {p: importlib.metadata.version(p) for p in ("langgraph", "math-verify", "sympy", "datasets", "httpx")}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    results = []
    tasks = [(problem, mode) for problem in problems for mode in dict.fromkeys(args.modes)]
    pool = None
    try:
        if args.workers > 1:
            # Processes, not threads: Math-Verify's timeouts need each solver's main thread.
            pool = concurrent.futures.ProcessPoolExecutor(args.workers, initializer=setup_worker,
                                                          initargs=(args, prices))
            rows = pool.map(solve, tasks)       # results come back in task order
        else:
            setup_worker(args, prices)
            rows = map(solve, tasks)
        with (output / "results.jsonl").open("w") as stream:
            for row in rows:
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{row['id']} {row['mode']}: correct={row['correct']} calls={row['llm_calls']} status={row['status']}", flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
        elif isinstance(_worker.get("backend"), ChatBackend):
            _worker["backend"].close()
        summary = summarize(results)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        if summary:
            with (output / "summary.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
    print(f"{'DEMO ONLY — ' if args.demo else ''}Results: {output.resolve()}")


if __name__ == "__main__":
    main()

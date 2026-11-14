from collections import defaultdict
from statistics import mean


def usage_metrics(trace, prices):
    calls = [e for e in trace if e["kind"] == "llm"]
    totals = {}
    for field in ("input_tokens", "output_tokens", "cached_tokens"):
        values = [e["usage"].get(field) for e in calls]
        totals[field] = sum(values) if all(v is not None for v in values) else None
    cost = None
    inp, out, cache = (totals[k] for k in ("input_tokens", "output_tokens", "cached_tokens"))
    if inp is not None and out is not None and cache is not None:
        if prices.get("input") is not None and prices.get("output") is not None and (not cache or prices.get("cached") is not None):
            cost = ((inp - cache) * prices["input"] + out * prices["output"] + cache * (prices.get("cached") or 0)) / 1e6
    return {**totals, "estimated_cost_usd": cost, "llm_calls": len(calls),
            "tool_calls": sum(e["kind"] == "tool" for e in trace)}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["mode"])].append(row)
    output = []
    for (dataset, mode), group in sorted(groups.items()):
        summary = {"dataset": dataset, "mode": mode, "n": len(group),
                   "accuracy": mean(r["correct"] for r in group),
                   "errors": sum(r["status"] == "error" for r in group),
                   "unparsed": sum(r["grading_status"] in ("unparsed", "error", "empty") for r in group)}
        for key in ("input_tokens", "output_tokens", "cached_tokens", "estimated_cost_usd", "llm_calls", "tool_calls", "rounds", "latency_seconds"):
            values = [r[key] for r in group]
            complete = all(v is not None for v in values)
            summary["mean_" + key] = mean(values) if complete else None
            if key in ("input_tokens", "output_tokens", "estimated_cost_usd"):
                summary["total_" + key] = sum(values) if complete else None
        output.append(summary)
    return output

"""Generate an auditable paired report from completed, fixed-sample runs."""
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def read_run(dataset, version):
    directory = ROOT / f"results/validation20-{version}-{dataset}"
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_modes = {"direct", "single", "multi"} if version == "baseline" else {"multi"}
    expected = {(ident, mode) for _, ident in manifest["selected_ids"] for mode in expected_modes}
    assert len(rows) == len(expected) and {(r["id"], r["mode"]) for r in rows} == expected
    assert len(manifest["selected_ids"]) == 20 and (directory / "summary.json").exists()
    return rows, manifest


def wilson(correct, n):
    z = 1.96
    p = correct / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0.0, center - spread), min(1.0, center + spread)]


def stats(rows):
    usage = [t.get("usage", {}) for r in rows for t in r["trace"] if t["kind"] == "llm"]
    known_tokens = sum((u.get("input_tokens") or 0) + (u.get("output_tokens") or 0) for u in usage)
    complete = all(u.get("input_tokens") is not None and u.get("output_tokens") is not None for u in usage)
    tool_events = [t for r in rows for t in r["trace"] if t["kind"] == "tool"]
    accepted = [r for r in rows if r["accepted"] is True]
    correct = sum(r["correct"] for r in rows)
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "wilson95": wilson(correct, len(rows)), "errors": sum(r["status"] == "error" for r in rows),
            "mean_rounds": mean(r["rounds"] for r in rows), "mean_calls": mean(r["llm_calls"] for r in rows),
            "known_tokens": known_tokens, "usage_complete": complete,
            "tool_calls": len(tool_events), "tool_failures": sum(not t["ok"] for t in tool_events),
            "solver_output_format_failures": sum(t["role"] == "solver" and t["ok"] and len(t["output"].strip().splitlines()) != 1 for t in tool_events),
            "grading_unparsed": sum(r["grading_status"] in ("unparsed", "empty", "error") for r in rows),
            "accepted": len(accepted), "accepted_but_wrong": sum(not r["correct"] for r in accepted),
            "parse_errors": sum(bool(t.get("parse_error")) for r in rows for t in r["trace"])}


def paired(old, new):
    left = {r["id"]: r for r in old if r["mode"] == "multi"}
    right = {r["id"]: r for r in new}
    assert left.keys() == right.keys()
    both = [(left[k], right[k]) for k in left if left[k]["status"] == right[k]["status"] == "ok"]
    wins = sum(not a["correct"] and b["correct"] for a, b in both)
    losses = sum(a["correct"] and not b["correct"] for a, b in both)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    old_tokens = sum(a["input_tokens"] + a["output_tokens"] for a, _ in both)
    new_tokens = sum(b["input_tokens"] + b["output_tokens"] for _, b in both)
    return {"paired_complete_n": len(both), "baseline_correct": sum(a["correct"] for a, _ in both),
            "optimized_correct": sum(b["correct"] for _, b in both), "wins": wins, "losses": losses,
            "mcnemar_exact_p": p, "baseline_tokens": old_tokens, "optimized_tokens": new_tokens,
            "token_change_fraction": new_tokens / old_tokens - 1 if old_tokens else None}


def main():
    report = {}
    lines = ["# 扩大样本与优化对照报告", "", "日期：2026-11-14。模型：SiliconFlow / Qwen/Qwen2.5-7B-Instruct。", "",
             "## 设计与变更", "", "- 每个数据集 20 道新题，共 40 题；排除之前的 4 道开发题。样本种子 20261114，题号和哈希见 experiments/validation20.json。",
             "- 原版三种模式共 120 次作答；优化版 multi 40 次。direct 与 single 算法未改，沿用同期原版结果。",
             "- 模型、温度 0、单次输出上限 2048 token、轮数上限 3、JSON 模式、数据与评分器相同。两个数据集并行；原版与优化版先后执行，服务状态差异无法完全排除。",
             "- 优化前镜像 baseline-v1、优化后镜像 optimized-v2 均保留；优化仅使用旧开发题进行，验证题标准答案不进入 Agent。",
             "- 去除代码占位值提示；答案从成功执行的 stdout 提取；SymPy 对象在沙箱中输出为 LaTeX；删除多余的 finalizer 调用。",
             "- 求解器失败不继续验证；验证代码技术故障只修复检查；后续代码崩溃保留上次成功执行的候选答案。达到上限仍按最后保留候选答案评分。",
             "- 28 项容器回归测试通过。", "", "## 全部样本（失败计错）", "",
             "| 数据集 | 版本/模式 | 正确 | 95% Wilson 区间 | API 异常 | 平均轮数 | 平均模型调用 | 已知 token |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for dataset in ("gsm8k", "math500"):
        old, m1 = read_run(dataset, "baseline")
        new, m2 = read_run(dataset, "optimized")
        for key in ("selected_ids", "data_sha256", "model", "max_tokens", "temperature", "max_rounds", "json_mode"):
            assert m1[key] == m2[key], key
        groups = {f"baseline/{mode}": stats([r for r in old if r["mode"] == mode]) for mode in ("direct", "single", "multi")}
        groups["optimized/multi"] = stats(new)
        report[dataset] = {"groups": groups, "paired": paired(old, new)}
        for name, s in groups.items():
            low, high = s["wilson95"]
            token_label = f"{s['known_tokens']:,}" + ("（下界）" if not s["usage_complete"] else "")
            lines.append(f"| {dataset} | {name} | {s['correct']}/{s['n']} | {low:.1%}–{high:.1%} | {s['errors']} | {s['mean_rounds']:.2f} | {s['mean_calls']:.2f} | {token_label} |")
    lines += ["", "## 排除 API 异常的同题配对分析", ""]
    for dataset, data in report.items():
        p = data["paired"]
        lines.append(f"- {dataset}：双方完成 {p['paired_complete_n']} 题，原版正确 {p['baseline_correct']}，优化后正确 {p['optimized_correct']}；由错转对 {p['wins']}，由对转错 {p['losses']}。配对 token 变化 {p['token_change_fraction']:+.1%}；精确 McNemar 双侧 p={p['mcnemar_exact_p']:.4f}（探索性，未校正多重比较）。")
    lines += ["", "## 执行与内部验证", "", "| 数据集/版本 | 工具失败/调用 | 求解输出空/多行 | JSON 解析失败 | 内部接受 | 接受但评分错误 |", "|---|---:|---:|---:|---:|---:|"]
    for dataset, data in report.items():
        for version in ("baseline/multi", "optimized/multi"):
            s = data["groups"][version]
            lines.append(f"| {dataset}/{version} | {s['tool_failures']}/{s['tool_calls']} | {s['solver_output_format_failures']} | {s['parse_errors']} | {s['accepted']} | {s['accepted_but_wrong']} |")
    lines += ["", "## 结果解读", ""]
    for dataset, data in report.items():
        old = data['groups']['baseline/multi']
        new = data['groups']['optimized/multi']
        single = data['groups']['baseline/single']
        lines.append(f"- {dataset}：多 Agent 准确率 {old['accuracy']:.0%} → {new['accuracy']:.0%}，平均模型调用 {old['mean_calls']:.2f} → {new['mean_calls']:.2f}。单 Agent + 工具为 {single['accuracy']:.0%}。")
    lines += ["", "GSM8K 在本次样本上同时改善准确率和成本；MATH-500 总体准确率不变，主要收益是减少 token 和调用。MATH-500 的单 Agent + 工具仍高于多 Agent，不能据此宣称复杂角色协作普遍更好。内部 accepted 不能作为正确性保证。", "",
              "## 限制与复现", "", "20 题/数据集仍是小样本，不能声称全量基准性能或稳定胜出。两版本的每次生成 token 上限和循环上限相同，但调用结构、提示词和实际总预算不同；这是整套流程优化对照，不能将变化单独归因于某个修改。内部验证可能误判，尤其模型求解与验证共享能力局限。",
              "direct 沿用只要求最终 answer 字段的原始提示词，没有显式推理步骤，因此结果不能代表更强的单次推理提示词。评分器对复杂符号答案也可能误判；此处报告自动评分结果，未按人工判断选择性修改分数。",
              "token 使用服务返回的 usage。超时请求可能已在服务端计费，缺失用量不可恢复；表内标记下界而不按零成本处理。未配置价格，不能报告货币费用。", "",
              "逐题预测、完整 trace、manifest 和 summary 位于 results/validation20-{baseline,optimized}-{gsm8k,math500}/。原始失败全部保留，未选择性重试。", "",
              "重新生成报告：`python3 experiments/report_validation.py`。运行新实验需使用新的输出目录，防止覆盖原始记录。"]
    (ROOT / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n")
    (ROOT / "experiments/validation20-summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

import json
import random
from pathlib import Path


def load_local(path, limit=None, seed=42):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k] for k in ("id", "dataset", "problem", "answer")):
            raise ValueError("Each row needs nonempty string id/dataset/problem/answer")
        key = (row["dataset"], row["id"])
        if key in ids:
            raise ValueError(f"Duplicate problem: {key}")
        ids.add(key)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def prepare(dataset, output, revision="main"):
    from datasets import load_dataset
    if dataset == "gsm8k":
        source = load_dataset("openai/gsm8k", "main", split="test", revision=revision)
        rows = [{"id": f"gsm8k/test/{i}", "dataset": dataset, "problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip()} for i, r in enumerate(source)]
    else:
        source = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=revision)
        rows = [{"id": r["unique_id"], "dataset": dataset, "problem": r["problem"],
                 "answer": r["answer"], "subject": r["subject"], "level": r["level"]} for r in source]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps({
        "dataset": dataset, "split": "test", "revision": revision,
        "fingerprint": source._fingerprint, "count": len(rows)}, indent=2))


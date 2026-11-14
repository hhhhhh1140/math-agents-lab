import json
import os
from pathlib import Path
import subprocess
import sys


def test_offline_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"
    command = [sys.executable, "-m", "math_agents.cli", "run", "--demo",
               "--data", str(root / "data/demo.jsonl"), "--output", str(output)]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(row["demo"] and row["status"] == "ok" for row in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary) == 3
    assert all(row["n"] == 2 for row in summary)
    assert (output / "summary.csv").exists()
    duplicate = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert duplicate.returncode != 0
    assert len((output / "results.jsonl").read_text().splitlines()) == 6

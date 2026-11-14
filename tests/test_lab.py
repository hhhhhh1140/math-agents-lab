import json

import httpx
import pytest

from math_agents.backend import ChatBackend, DemoBackend
from math_agents.data import load_local
from math_agents.execution import DemoExecutor
from math_agents.grading import grade
from math_agents.metrics import summarize, usage_metrics
from math_agents.workflow import Runner


@pytest.mark.parametrize("mode,calls,tools", [("direct", 1, 0), ("single", 2, 1), ("multi", 5, 2)])
def test_three_modes(mode, calls, tools):
    runner = Runner(DemoBackend(), DemoExecutor())
    result = runner.run(mode, "Solve 2*x + 1 = 7 for x.")
    assert result["answer"] == "3"
    usage = usage_metrics(runner.trace, {})
    assert (usage["llm_calls"], usage["tool_calls"]) == (calls, tools)
    assert all("gold" not in event.get("context", {}) for event in runner.trace)


class RejectingBackend(DemoBackend):
    def __init__(self, reject_count=1):
        self.rejections = reject_count

    def complete(self, role, instruction, context):
        if role == "reviewer" and self.rejections:
            self.rejections -= 1
            return json.dumps({"accept": False, "feedback": "Check domain constraints"}), {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0}
        return super().complete(role, instruction, context)


def test_rejection_routes_back_to_solver():
    runner = Runner(RejectingBackend(), DemoExecutor(), 3)
    result = runner.run("multi", "Solve 2*x + 1 = 7 for x.")
    assert result["accepted"] is True
    assert result["rounds"] == 2
    solvers = [t for t in runner.trace if t.get("role") == "solver" and t["kind"] == "llm"]
    assert solvers[1]["context"]["feedback"] == "Check domain constraints"


def test_retry_cap():
    runner = Runner(RejectingBackend(99), DemoExecutor(), 2)
    result = runner.run("multi", "Solve 2*x + 1 = 7 for x.")
    assert result["rounds"] == 2
    assert result["accepted"] is False
    assert len([t for t in runner.trace if t["kind"] == "llm"]) == 9


def test_verifier_failure_cannot_be_overridden():
    class FailedExecutor:
        def run(self, code):
            return {"ok": False, "output": "PASS"}
    result = Runner(DemoBackend(), FailedExecutor(), 1).run("multi", "Solve x.")
    assert result["accepted"] is False


@pytest.mark.parametrize("prediction,gold,dataset,correct", [
    ("1,200", "1200", "gsm8k", True),
    ("0.5", "1/2", "gsm8k", True),
    ("There are 3 or 7", "7", "gsm8k", False),
    ("", "0", "gsm8k", False),
    (r"\boxed{\frac{1}{2}}", "0.5", "math500", True),
    (r"\sqrt{8}/2", r"\sqrt{2}", "math500", True),
    (r"(1,2)", r"(2,1)", "math500", False),
    ("-3", "3", "math500", False),
])
def test_grading(prediction, gold, dataset, correct):
    assert grade(prediction, gold, dataset)["correct"] is correct


def test_cost_and_missing_usage():
    trace = [{"kind": "llm", "usage": {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 40}}]
    assert usage_metrics(trace, {"input": 2, "output": 8, "cached": 1})["estimated_cost_usd"] == pytest.approx(0.00032)
    assert usage_metrics(trace, {"input": 2, "output": 8})["estimated_cost_usd"] is None
    trace.append({"kind": "llm", "usage": {}})
    assert usage_metrics(trace, {})["input_tokens"] is None


def test_bad_json_counts_call_and_ends_at_cap():
    class BadBackend:
        def complete(self, *args):
            return "not JSON", {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}
    runner = Runner(BadBackend(), DemoExecutor(), 1)
    result = runner.run("multi", "Solve x.")
    assert result["accepted"] is False
    assert usage_metrics(runner.trace, {})["llm_calls"] == 5


def test_sampling_and_duplicate_detection(tmp_path):
    rows = [{"id": str(i), "dataset": "demo", "problem": "x?", "answer": "1"} for i in range(10)]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_local(path, 3, 42) == load_local(path, 3, 42)
    path.write_text("\n".join(json.dumps(r) for r in [rows[0], rows[0]]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_local(path)


@pytest.mark.parametrize("json_mode", [False, True])
def test_http_adapter(monkeypatch, json_mode):
    monkeypatch.setenv("MATH_JSON_MODE", str(json_mode).lower())
    monkeypatch.setenv("MATH_MODEL", "test-model")
    monkeypatch.setenv("MATH_API_KEY", "fake-test-key")
    backend = ChatBackend()
    backend.client.close()
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload.get("response_format") == ({"type": "json_object"} if json_mode else None)
        assert payload["messages"][1]["content"] == '{"problem": "2+2"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer":"4"}'}}],
                                         "usage": {"prompt_tokens": 9, "completion_tokens": 5}})
    backend.client = httpx.Client(base_url="https://example.test/v1/", transport=httpx.MockTransport(handler))
    raw, usage = backend.complete("direct", "Solve", {"problem": "2+2"})
    assert json.loads(raw)["answer"] == "4"
    assert usage["input_tokens"] == 9
    backend.close()


def test_explicit_api_proxy_ignores_global_environment(monkeypatch):
    monkeypatch.setenv("MATH_MODEL", "test-model")
    monkeypatch.setenv("MATH_API_KEY", "fake-test-key")
    monkeypatch.setenv("MATH_PROXY_URL", "http://192.168.5.2:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://invalid-global-proxy:9999")
    monkeypatch.setenv("NO_PROXY", "*")
    captured = {}
    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr(httpx, "Client", Client)
    ChatBackend()
    assert captured["proxy"] == "http://192.168.5.2:7897"
    assert captured["trust_env"] is False
    assert captured["timeout"].connect == 15
    monkeypatch.setenv("MATH_PROXY_URL", "")
    ChatBackend()
    assert captured["proxy"] is None


def test_summary_includes_failed_attempts():
    base = {"dataset": "demo", "mode": "direct", "correct": True, "status": "ok", "grading_status": "numeric",
            "rounds": 1, "latency_seconds": 1, **usage_metrics([], {})}
    summary = summarize([base, {**base, "correct": False, "status": "error", "grading_status": "empty"}])[0]
    assert summary["n"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["errors"] == 1

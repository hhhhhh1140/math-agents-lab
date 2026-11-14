import httpx
import pytest

from math_agents.backend import ChatBackend, DemoBackend
from math_agents.consensus import ConsensusRunner, boxed_answer, code_block, printed_answer, same
from math_agents.execution import DemoExecutor
from math_agents.grading import grade

USAGE = {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}


class Scripted:
    """Replies per role, in order; the last reply of a role repeats."""
    def __init__(self, **replies):
        self.replies = {role: list(texts) for role, texts in replies.items()}

    def chat(self, role, system, user):
        texts = self.replies[role]
        text = texts.pop(0) if len(texts) > 1 else texts[0]
        if isinstance(text, Exception):
            raise text
        return text, dict(USAGE)


class Outputs:
    """Sandbox stand-in: returns the queued results in order."""
    def __init__(self, *results):
        self.results = list(results)

    def run(self, code):
        return self.results.pop(0)


def roles(runner):
    return [e["role"] for e in runner.trace if e["kind"] == "llm"]


def code(answer):
    return f"Compute it.\n```python\nprint({answer})\n```"


def test_agreement_accepts_without_referee():
    runner = ConsensusRunner(DemoBackend(), DemoExecutor())
    state = runner.run("multi", "Solve 2*x + 1 = 7 for x.")
    assert state["answer"] == "3" and state["accepted"]
    assert roles(runner) == ["reasoner", "coder"]
    assert all("gold" not in str(e.get("context")) for e in runner.trace)


def test_equivalent_forms_count_as_agreement():
    backend = Scripted(reasoner=[r"So x = \boxed{\frac{1}{2}}"], coder=[code("0.5")])
    runner = ConsensusRunner(backend, Outputs({"ok": True, "output": "0.5\n"}))
    state = runner.run("multi", "Solve 2x = 1.")
    assert state["accepted"] and roles(runner) == ["reasoner", "coder"]


@pytest.mark.parametrize("referee,expected,accepted", [
    (r"\boxed{7}", "7", True),      # sides with the reasoner
    (r"\boxed{9}", "9", True),      # sides with the program
    (r"\boxed{11}", "11", False),   # three-way split: referee's answer, not accepted
    ("I cannot tell.", "7", False),  # referee gives nothing: keep the reasoner's answer
])
def test_disagreement_goes_to_referee_and_majority_wins(referee, expected, accepted):
    backend = Scripted(reasoner=[r"\boxed{7}"], coder=[code(9)], referee=[referee])
    runner = ConsensusRunner(backend, Outputs({"ok": True, "output": "9\n"}))
    state = runner.run("multi", "Problem")
    assert (state["answer"], state["accepted"]) == (expected, accepted)
    assert roles(runner) == ["reasoner", "coder", "referee"]
    prompt = runner.trace[-1]["context"]["user"]
    assert "answer: 7" in prompt and "answer: 9" in prompt and "print(9)" in prompt


def test_program_is_repaired_with_the_real_error():
    backend = Scripted(reasoner=[r"\boxed{3}"], coder=[code("x"), code(3)])
    executor = Outputs({"ok": False, "output": "NameError: name 'x' is not defined"}, {"ok": True, "output": "3\n"})
    runner = ConsensusRunner(backend, executor)
    state = runner.run("multi", "Problem")
    assert state["answer"] == "3" and state["accepted"] and state["rounds"] == 2
    assert roles(runner) == ["reasoner", "coder", "coder"]
    assert "NameError" in runner.trace[-2]["context"]["user"]


def test_empty_output_and_missing_code_are_failures_not_answers():
    backend = Scripted(reasoner=["no box here"], coder=["no code here", code("None")], referee=["unsure"])
    runner = ConsensusRunner(backend, Outputs({"ok": True, "output": "\n"}), 2)
    state = runner.run("multi", "Problem")
    assert state["answer"] == "" and not state["accepted"]
    assert roles(runner) == ["reasoner", "coder", "coder", "referee"]


def test_one_agent_failing_keeps_the_other_answer():
    backend = Scripted(reasoner=[httpx.ReadTimeout("slow")], coder=[code(5)], referee=[httpx.ReadTimeout("slow")])
    runner = ConsensusRunner(backend, Outputs({"ok": True, "output": "5\n"}))
    state = runner.run("multi", "Problem")
    assert state["answer"] == "5" and not state["accepted"]
    assert runner.trace[0]["error"] == "ReadTimeout"


def test_total_failure_is_reported_as_error():
    backend = Scripted(reasoner=[httpx.ReadTimeout("slow")])
    with pytest.raises(httpx.ReadTimeout):
        ConsensusRunner(backend, DemoExecutor()).run("direct", "Problem")


def test_direct_and_single_modes():
    runner = ConsensusRunner(DemoBackend(), DemoExecutor())
    assert runner.run("direct", "Solve x.")["answer"] == "3" and roles(runner) == ["reasoner"]
    assert runner.run("single", "Solve x.")["answer"] == "3" and roles(runner) == ["coder"]


def test_extraction_helpers():
    assert boxed_answer(r"first \boxed{1}, finally \boxed{\frac{a}{b}}.") == r"\frac{a}{b}"
    assert boxed_answer("no answer") == "" and boxed_answer(r"\boxed{1") == ""
    assert code_block("a\n```python\nprint(1)\n```\nb\n```python\nprint(2)\n```") == "print(2)"
    assert code_block("print(1)") == ""
    assert printed_answer({"ok": True, "output": "debug\n58.99999999999818\n"}) == "59"
    assert printed_answer({"ok": True, "output": "83.0\n"}) == "83"
    assert printed_answer({"ok": True, "output": "0.125\n"}) == "0.125"
    assert printed_answer({"ok": False, "output": "3\n"}) == ""
    assert same("1,200", "1200") and same(r"\frac{1}{2}", "0.5") and not same("3", "-3") and not same("", "")


@pytest.mark.parametrize("prediction", [r"\$18", "40\\%", r"12 \text{ inches}", "18", r"1{,}200"])
def test_gsm8k_grader_ignores_unambiguous_decoration(prediction):
    gold = {"40\\%": "40", r"12 \text{ inches}": "12", r"1{,}200": "1200"}.get(prediction, "18")
    assert grade(prediction, gold, "gsm8k")["correct"]
    assert not grade("There are 18 or 20", "18", "gsm8k")["correct"]


def test_transient_api_failures_are_retried(monkeypatch):
    monkeypatch.setenv("MATH_MODEL", "test-model")
    monkeypatch.setenv("MATH_API_KEY", "fake-test-key")
    monkeypatch.setenv("MATH_API_RETRY_WAIT", "0")
    backend = ChatBackend()
    backend.client.close()
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": r"\boxed{4}"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 9, "completion_tokens": 5}})
    backend.client = httpx.Client(base_url="https://example.test/v1/", transport=httpx.MockTransport(handler))
    raw, usage = backend.chat("reasoner", "Solve", "2+2")
    assert raw == r"\boxed{4}" and usage["attempts"] == 3 and "response_format" not in calls[-1].content.decode()
    calls.clear()
    backend.client = httpx.Client(base_url="https://example.test/v1/",
                                  transport=httpx.MockTransport(lambda request: httpx.Response(401)))
    with pytest.raises(httpx.HTTPStatusError):
        backend.chat("reasoner", "Solve", "2+2")
    backend.close()

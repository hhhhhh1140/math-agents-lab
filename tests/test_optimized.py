import json

from math_agents.backend import DemoBackend
from math_agents.execution import DemoExecutor
from math_agents.optimized import OptimizedRunner


def test_answer_comes_from_execution_without_finalizer():
    runner = OptimizedRunner(DemoBackend(), DemoExecutor())
    state = runner.run("multi", "Solve 2*x + 1 = 7 for x.")
    assert state["answer"] == "3" and state["accepted"]
    assert [e["role"] for e in runner.trace if e["kind"] == "llm"] == ["planner", "solver", "verifier", "reviewer"]
    verifier = next(e for e in runner.trace if e.get("role") == "verifier" and e["kind"] == "llm")
    assert set(verifier["context"]) == {"problem", "answer"}


def test_solver_failure_skips_downstream_and_never_invents_answer():
    class BrokenExecutor:
        def run(self, code):
            return {"ok": False, "output": "NameError: missing symbol"}
    runner = OptimizedRunner(DemoBackend(), BrokenExecutor(), 2)
    state = runner.run("multi", "Solve x.")
    assert state["answer"] == "" and not state["accepted"] and state["rounds"] == 2
    assert [e["role"] for e in runner.trace if e["kind"] == "llm"] == ["planner", "solver", "solver"]
    assert "NameError" in runner.trace[-2]["context"]["feedback"]


def test_empty_stdout_does_not_count_as_solution():
    class EmptyExecutor:
        def run(self, code):
            return {"ok": True, "output": ""}
    state = OptimizedRunner(DemoBackend(), EmptyExecutor(), 1).run("multi", "Solve x.")
    assert not state["execution"]["ok"] and not state["answer"]


def test_check_repair_preserves_solver_and_candidate():
    class FlakyExecutor(DemoExecutor):
        def __init__(self):
            self.failed = False
        def run(self, code):
            if code == "print('PASS')" and not self.failed:
                self.failed = True
                return {"ok": False, "output": "SyntaxError in checker"}
            return super().run(code)
    runner = OptimizedRunner(DemoBackend(), FlakyExecutor(), 2)
    state = runner.run("multi", "Solve x.")
    roles = [e["role"] for e in runner.trace if e["kind"] == "llm"]
    assert state["answer"] == "3" and state["accepted"] and state["rounds"] == 2
    assert roles == ["planner", "solver", "verifier", "verifier", "reviewer"]
    contexts = [e["context"] for e in runner.trace if e["kind"] == "llm" and e["role"] == "verifier"]
    assert "SyntaxError" in contexts[1]["previous_check"]["output"]


def test_last_executed_answer_survives_later_solver_failure():
    class RejectBackend(DemoBackend):
        def complete(self, role, instruction, context):
            if role == "reviewer":
                return json.dumps({"accept": False, "feedback": "Recheck domain"}), {}
            return super().complete(role, instruction, context)
    class LaterFailure(DemoExecutor):
        attempts = 0
        def run(self, code):
            if code == "print(3)":
                self.attempts += 1
                if self.attempts > 1:
                    return {"ok": False, "output": "Runtime error"}
            return super().run(code)
    state = OptimizedRunner(RejectBackend(), LaterFailure(), 2).run("multi", "Solve x.")
    assert state["answer"] == "3" and not state["accepted"] and not state["execution"]["ok"]

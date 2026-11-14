import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    problem: str
    plan: str
    answer: str
    code: str
    execution: dict
    verification: dict
    feedback: str
    accepted: bool
    rounds: int


PROMPTS = {
    "direct": 'Solve the math problem carefully without tools. Return {"answer":"final value or LaTeX expression only"}.',
    "planner": 'Decompose the problem, identify equations and constraints. Return {"plan":"steps"}.',
    "solver": 'Write Python/SymPy to solve the original problem following the plan and feedback. Print the candidate answer. Return {"code":"executable Python"}.',
    "finalizer": 'Read the solver code and actual execution output. Return {"answer":"final value or LaTeX expression only"}. Do not claim failed execution succeeded.',
    "single": 'Solve the problem using Python/SymPy as needed. Return either {"code":"Python that prints results"} or {"answer":"final value or LaTeX expression only"}. Use tool history to correct errors. If final_only is true you MUST give an answer.',
    "verifier": 'Independently check the candidate answer against the ORIGINAL problem. Write Python/SymPy that substitutes it into original conditions, checks domain constraints and completeness, or independently recomputes the result. Do not merely rerun solver code. Print PASS only if all checks succeed, otherwise FAIL. Return {"code":"Python checks", "reason":"what is checked"}.',
    "reviewer": 'Review the original problem, solution and independent check. Reject incorrect, incomplete or unsupported answers. Tool success alone is not mathematical proof. Return {"accept":true or false,"feedback":"specific corrections or justification"}.',
}


class Runner:
    prompts = PROMPTS
    def __init__(self, backend, executor, max_rounds=3):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.backend, self.executor, self.max_rounds = backend, executor, max_rounds
        self.trace = []
        self.rounds = 0

    def ask(self, role, context):
        event = {"kind": "llm", "role": role, "context": context, "usage": {}}
        self.trace.append(event)
        raw, usage = self.backend.complete(role, self.prompts[role], context)
        event.update(raw=raw, usage=usage)
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("Expected a JSON object")
            return obj
        except (ValueError, TypeError) as exc:
            event["parse_error"] = str(exc)
            return {"parse_error": str(exc)}

    def execute(self, role, code):
        result = self.executor.run(code) if isinstance(code, str) and code.strip() else {"ok": False, "output": "Missing code"}
        self.trace.append({"kind": "tool", "role": role, "code": code, **result})
        return result

    @staticmethod
    def answer(obj):
        value = obj.get("answer", "")
        return str(value).strip() if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ""

    def multi_graph(self):
        def planner(s):
            return {"plan": self.ask("planner", {"problem": s["problem"]}).get("plan", "")}

        def solver(s):
            self.rounds = s.get("rounds", 0) + 1
            obj = self.ask("solver", dict(s))
            execution = self.execute("solver", obj.get("code", ""))
            context = {**s, "code": obj.get("code", ""), "execution": execution}
            answer = self.answer(self.ask("finalizer", context))
            return {"code": obj.get("code", ""), "execution": execution, "answer": answer,
                    "rounds": self.rounds, "accepted": False}

        def verifier(s):
            obj = self.ask("verifier", {"problem": s["problem"], "answer": s["answer"]})
            result = self.execute("verifier", obj.get("code", ""))
            passed = result["ok"] and result["output"].strip() == "PASS"
            return {"verification": {**obj, **result, "passed": passed}}

        def reviewer(s):
            obj = self.ask("reviewer", dict(s))
            accepted = (obj.get("accept") is True and s["verification"]["passed"]
                        and s["execution"]["ok"] and bool(s["answer"]))
            return {"accepted": bool(accepted), "feedback": obj.get("feedback", "Invalid reviewer output; retry.")}

        graph = StateGraph(State)
        for name, node in [("planner", planner), ("solver", solver), ("verifier", verifier), ("reviewer", reviewer)]:
            graph.add_node(name, node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "solver")
        graph.add_edge("solver", "verifier")
        graph.add_edge("verifier", "reviewer")
        graph.add_conditional_edges("reviewer", lambda s: "end" if s["accepted"] or s["rounds"] >= self.max_rounds else "retry",
                                    {"end": END, "retry": "solver"})
        return graph.compile()

    def run(self, mode, problem):
        self.trace, self.rounds = [], 0
        if mode == "multi":
            return self.multi_graph().invoke({"problem": problem, "rounds": 0},
                                             {"recursion_limit": 4 * self.max_rounds + 5})
        if mode == "direct":
            self.rounds = 1
            return {"answer": self.answer(self.ask("direct", {"problem": problem})), "rounds": 1}
        if mode != "single":
            raise ValueError(f"Unknown mode: {mode}")
        history = []
        # One decision per round; reserve the last decision for a final answer.
        for i in range(self.max_rounds):
            self.rounds = i + 1
            obj = self.ask("single", {"problem": problem, "history": history.copy(), "final_only": i == self.max_rounds - 1})
            if self.answer(obj):
                return {"answer": self.answer(obj), "rounds": self.rounds}
            if i == self.max_rounds - 1:
                break
            result = self.execute("single", obj.get("code", ""))
            history.append({"response": obj, "execution": result})
        return {"answer": "", "rounds": self.rounds}

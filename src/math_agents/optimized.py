"""Execution-grounded multi-agent workflow; no access to reference answers."""
from langgraph.graph import END, START, StateGraph

from .workflow import PROMPTS, Runner, State


# Runs INSIDE the isolated sandbox. Preserve exact symbolic values in a format
# understood by the evaluator instead of Python-specific str(SymPy) syntax.
SYMBOLIC_PRINT = """import builtins as _lab_builtins
from sympy import Basic as _lab_Basic, latex as _lab_latex
_lab_original_print = _lab_builtins.print
def _lab_print(*args, **kwargs):
    _lab_original_print(*(_lab_latex(x) if isinstance(x, _lab_Basic) else x for x in args), **kwargs)
_lab_builtins.print = _lab_print
"""


OPTIMIZED_PROMPTS = {
    **PROMPTS,
    "planner": (
        "Analyze the original math problem. Identify quantities, equations, domain restrictions, "
        "and the exact quantity requested. Return one JSON object with a plan string containing "
        "a short concrete solution strategy. Do not output a template or placeholder."
    ),
    "solver": (
        "Solve the original problem by writing a complete executable Python 3 program. "
        "Return a JSON object with exactly one key code whose value is the FULL Python source. "
        "The program MUST print exactly ONE line: the final mathematical answer, with no labels. "
        "For symbolic expressions use sympy.latex(answer); for numbers print(answer). "
        "A bare expression is NOT printed in a script. Include all imports and definitions. "
        "Use exact fractions (sympy.Rational), and radians (degrees*pi/180) for trigonometry. "
        "Do not call .evalf() on Python floats. For inequalities prefer solve_univariate_inequality. "
        "Only Python standard library and SymPy are installed. No network, files, input() or pip. "
        "Read any previous error and correct its cause. Preserve correct calculations. "
        "Never return words such as executable Python in place of actual code. "
        "Example for the unrelated problem 2+3: {\"code\":\"print(2 + 3)\"}."
    ),
    "verifier": (
        "Independently verify the candidate against the ORIGINAL problem. Return one JSON object "
        "with keys code (COMPLETE executable Python source) and reason (short explanation). "
        "Independently recompute the requested quantity or substitute the candidate into original "
        "conditions. Check constraints and completeness. The program must print exactly PASS "
        "if the candidate is correct, otherwise exactly FAIL. Do not print explanations. "
        "Use actual computations and assertions, never unconditional print('PASS'). "
        "Include imports; only standard library and SymPy exist. Do not use parse_latex or "
        "external libraries; translate the provided candidate into a Python/SymPy expression. "
        "Exact symbolic comparisons are preferred; for numerical values use an explicit tolerance. "
        "If previous_check contains an execution error, repair that check using its traceback. "
        "Never return the literal words Python checks. "
        "Example for unrelated problem 2+3, candidate 5: "
        "{\"code\":\"expected = 2 + 3\\nprint('PASS' if expected == 5 else 'FAIL')\","
        "\"reason\":\"Recompute the sum and compare\"}."
    ),
    "reviewer": (
        "Review the original problem and actual solution and verification evidence. Return a JSON "
        "object with accept (boolean) and feedback (specific explanation). Approve only if the "
        "answer solves the requested question, respects the domain, and the independent check "
        "actually establishes correctness. Reject unconditional PASS and checks of unrelated facts. "
        "Do not change the candidate answer. If rejecting, explain the exact issue to fix."
    ),
}


class OptimizedRunner(Runner):
    prompts = OPTIMIZED_PROMPTS

    def execute(self, role, code):
        # DemoExecutor is intentionally a fixed fixture, not a Python runtime.
        from .execution import DockerExecutor
        if role == "solver" and isinstance(self.executor, DockerExecutor) and isinstance(code, str) and code.strip():
            code = SYMBOLIC_PRINT + code
        return super().execute(role, code)

    def multi_graph(self):
        def planner(s):
            return {"plan": self.ask("planner", {"problem": s["problem"]}).get("plan", "")}

        def solver(s):
            self.rounds = s.get("rounds", 0) + 1
            # A broken check is not evidence that a successfully computed answer
            # is wrong. Repair the check without rewriting the solution.
            if s.get("execution", {}).get("ok") and s.get("verification", {}).get("technical_error") and s.get("answer"):
                return {"rounds": self.rounds, "accepted": False}
            context = {"problem": s["problem"], "plan": s.get("plan", ""), "feedback": s.get("feedback", "")}
            if s.get("code"):
                context["previous_attempt"] = {"code": s["code"], "execution": s.get("execution", {})}
            obj = self.ask("solver", context)
            code = obj.get("code", "")
            execution = self.execute("solver", code)
            lines = execution["output"].strip().splitlines()
            valid = execution["ok"] and len(lines) == 1 and bool(lines[0].strip())
            if not valid and execution["ok"]:
                execution = {**execution, "ok": False,
                             "output": "Expected exactly one printed answer line. Actual output: " + execution["output"]}
            # Keep the last executed candidate if a later attempt crashes. Never
            # ask an LLM to invent an answer from a traceback or empty output.
            answer = lines[0].strip() if valid else s.get("answer", "")
            return {"code": code, "execution": execution, "answer": answer,
                    "rounds": self.rounds, "accepted": False}

        def verifier(s):
            if not s["execution"]["ok"] or not s["answer"]:
                return {"verification": {"passed": False, "technical_error": False, "output": "Solver must produce a valid printed answer first."}}
            context = {"problem": s["problem"], "answer": s["answer"]}
            if s.get("verification", {}).get("technical_error"):
                context["previous_check"] = s["verification"]
            obj = self.ask("verifier", context)
            result = self.execute("verifier", obj.get("code", ""))
            output = result["output"].strip()
            technical = not result["ok"] or output not in ("PASS", "FAIL")
            return {"verification": {**obj, **result, "technical_error": technical,
                                     "passed": result["ok"] and output == "PASS"}}

        def reviewer(s):
            if not s["execution"]["ok"]:
                return {"accepted": False, "feedback": "Fix the solver program. " + s["execution"]["output"]}
            if s["verification"].get("technical_error"):
                return {"accepted": False, "feedback": "Independent check failed to execute; repair the check, preserve the candidate."}
            obj = self.ask("reviewer", {"problem": s["problem"], "answer": s["answer"],
                                       "code": s["code"], "verification": s["verification"]})
            return {"accepted": obj.get("accept") is True and s["verification"]["passed"],
                    "feedback": obj.get("feedback", "Review failed; independently recheck the solution.")}

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

"""Reason + program + referee workflow; no access to reference answers.

Why this exists: the earlier workflows force every agent to reply with a bare
JSON answer, so the model never gets to reason, and they let a model-written
PASS/FAIL checker veto the solver even though the checker is wrong more often
than the solver. Here two agents solve the problem independently in different
ways (step-by-step reasoning, and an executed Python/SymPy program). Agreement
between the two is the acceptance signal. Only on disagreement does a third
agent adjudicate, and the final answer is chosen by majority.
"""
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .grading import grade, numeric
from .workflow import Runner

# Runs INSIDE the sandbox: print SymPy values as simplified exact LaTeX, so that
# e.g. an unexpanded radical expression equal to 40 is reported as 40.
SIMPLIFYING_PRINT = """import builtins as _lab_builtins
import sympy as _lab_sympy
_lab_original_print = _lab_builtins.print
def _lab_show(x):
    if not isinstance(x, _lab_sympy.Basic):
        return x
    try:
        if isinstance(x, _lab_sympy.Expr) and not x.atoms(_lab_sympy.Float) and _lab_sympy.count_ops(x) < 150:
            x = _lab_sympy.simplify(x)
    except Exception:
        pass
    return _lab_sympy.latex(x)
def _lab_print(*args, **kwargs):
    _lab_original_print(*(_lab_show(x) for x in args), **kwargs)
_lab_builtins.print = _lab_print
"""
# One line, so tracebacks shown to the coder are off by a single line only.
PRELUDE = "exec(" + repr(SIMPLIFYING_PRINT) + ")\n"

FORMAT = (
    "Finish with the final answer inside \\boxed{...}. The box holds only the answer in simplest "
    "exact form (LaTeX fraction, radical or pi rather than a rounded decimal, unless the problem "
    "asks for a decimal), without units and without restating the question."
)

CONSENSUS_PROMPTS = {
    "reasoner": (
        "You are a careful mathematician. Solve the problem step by step, showing the work. "
        "Restate what is asked, then derive the result, then check it against every condition "
        "of the problem before answering. " + FORMAT
    ),
    "coder": (
        "You solve math problems by writing a Python 3 program. First describe the approach in "
        "two or three sentences, then give ONE complete program in a ```python fenced block.\n"
        "Rules: only the standard library and SymPy are installed; no input, files or network. "
        "Compute the answer from the conditions of the problem, by exact arithmetic "
        "(sympy.Rational, sympy.sqrt, integers) or by brute-force enumeration when the search "
        "space is small. Do not hard-code a result worked out in your head. The LAST line the "
        "program prints must be the final answer alone, with no label; printing a SymPy "
        "expression gives exact LaTeX. A bare expression prints nothing, so call print()."
    ),
    "referee": (
        "You are the referee for two independent attempts at the same math problem. They do not "
        "agree, so at least one is wrong. Solution A is step-by-step reasoning. Solution B is a "
        "Python program with its real output; the arithmetic it performed is reliable, but it "
        "may model the problem incorrectly. Reread the problem, find the first mistake in each "
        "attempt, and work out the correct answer yourself step by step. " + FORMAT
    ),
}


class ConsensusState(TypedDict, total=False):
    problem: str
    reasoning: str
    code: str
    execution: dict
    candidates: dict
    answer: str
    accepted: bool
    rounds: int


def boxed_answer(text):
    """Content of the last \\boxed{...}; empty if the model never committed to one."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return ""
    start = text.index("{", start) + 1
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i].strip()
    return ""


def code_block(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S | re.I)
    return blocks[-1].strip() if blocks else ""


def printed_answer(execution):
    """Last printed line of a successful run, with float noise such as 58.99999999999818 removed."""
    if not execution.get("ok"):
        return ""
    lines = [line.strip() for line in execution["output"].splitlines() if line.strip()]
    if not lines or len(lines[-1]) > 300:
        return ""
    answer = lines[-1]
    if re.fullmatch(r"[+-]?\d+\.\d+(?:e[+-]?\d+)?", answer, re.I):
        value = round(float(answer), 9)
        answer = str(int(value)) if value == int(value) and abs(value) < 1e15 else repr(value)
    return answer


def same(a, b):
    """Mathematical equivalence of two candidate answers (never uses a reference answer)."""
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return numeric(a) == numeric(b)
    except (ValueError, ZeroDivisionError):
        return grade(a, b, "math500")["correct"]


class ConsensusRunner(Runner):
    prompts = CONSENSUS_PROMPTS

    def say(self, role, user):
        event = {"kind": "llm", "role": role, "context": {"user": user}, "usage": {}}
        self.trace.append(event)
        try:
            raw, usage = self.backend.chat(role, self.prompts[role], user)
        except Exception as exc:
            # One agent failing must not discard what the others already found.
            event["error"] = type(exc).__name__
            self.failure = self.failure or exc
            return ""
        event.update(raw=raw, usage=usage)
        return raw

    def execute(self, role, code):
        from .execution import DockerExecutor
        if isinstance(self.executor, DockerExecutor):
            code = PRELUDE + code
        return super().execute(role, code)

    def reason(self, problem):
        raw = self.say("reasoner", "Problem:\n" + problem)
        return {"reasoning": raw, "answer": boxed_answer(raw)}

    def program(self, problem):
        """Write and run a program; on a crash or empty output, repair it with the real error."""
        user = "Problem:\n" + problem
        code, execution, answer = "", {"ok": False, "output": "No program was produced."}, ""
        for attempt in range(self.max_rounds):
            self.rounds = attempt + 1
            raw = self.say("coder", user)
            written = code_block(raw)
            if written:
                code, execution = written, self.execute("coder", written)
                if execution["ok"] and not printed_answer(execution):
                    execution = {"ok": False, "output": "The program printed no usable final answer line. Output was: "
                                 + execution["output"][-500:]}
            else:
                execution = {"ok": False, "output": "The reply contained no ```python code block."}
            answer = printed_answer(execution)
            if answer:
                break
            user = ("Problem:\n" + problem + "\n\nYour previous program:\n```python\n" + code + "\n```\n"
                    "It failed with:\n" + execution["output"][-1500:] + "\n\nExplain the cause in one "
                    "sentence, then give the full corrected program. If the same approach keeps "
                    "failing, use a simpler one (direct arithmetic, enumeration, or numerical "
                    "evaluation followed by sympy.nsimplify).")
        return {"code": code, "execution": execution, "answer": answer}

    def referee(self, s):
        c = s["candidates"]
        b_output = s["execution"].get("output", "")[-1500:]
        user = ("Problem:\n" + s["problem"] + "\n\n=== Solution A (reasoning), answer: "
                + (c["reasoner"] or "none") + " ===\n" + (s.get("reasoning") or "(no reply)")[-6000:]
                + "\n\n=== Solution B (program), answer: " + (c["coder"] or "none") + " ===\n```python\n"
                + (s.get("code") or "(no program)")[-4000:] + "\n```\nProgram output:\n" + b_output)
        return boxed_answer(self.say("referee", user))

    @staticmethod
    def decide(candidates):
        """Majority of the three; the referee breaks a three-way split."""
        a, b, r = candidates["reasoner"], candidates["coder"], candidates.get("referee", "")
        if same(a, b):
            return a, True
        if same(r, a):
            return a, True
        if same(r, b):
            return b, True
        return r or a or b, False

    def multi_graph(self):
        def reasoner(s):
            out = self.reason(s["problem"])
            return {"reasoning": out["reasoning"], "candidates": {"reasoner": out["answer"]}}

        def coder(s):
            out = self.program(s["problem"])
            return {"code": out["code"], "execution": out["execution"], "rounds": self.rounds,
                    "candidates": {**s["candidates"], "coder": out["answer"]}}

        def referee(s):
            return {"candidates": {**s["candidates"], "referee": self.referee(s)}}

        def decide(s):
            answer, accepted = self.decide(s["candidates"])
            return {"answer": answer, "accepted": accepted}

        def agree(s):
            c = s["candidates"]
            return "decide" if same(c["reasoner"], c["coder"]) else "referee"

        graph = StateGraph(ConsensusState)
        for name, node in [("reasoner", reasoner), ("coder", coder), ("referee", referee), ("decide", decide)]:
            graph.add_node(name, node)
        graph.add_edge(START, "reasoner")
        graph.add_edge("reasoner", "coder")
        graph.add_conditional_edges("coder", agree, {"decide": "decide", "referee": "referee"})
        graph.add_edge("referee", "decide")
        graph.add_edge("decide", END)
        return graph.compile()

    def run(self, mode, problem):
        self.trace, self.rounds, self.failure = [], 0, None
        if mode == "multi":
            state = self.multi_graph().invoke({"problem": problem, "rounds": 0})
        elif mode == "direct":      # one call, step-by-step reasoning, no tools
            self.rounds = 1
            state = {"answer": self.reason(problem)["answer"], "rounds": 1}
        elif mode == "single":      # one agent with the Python tool and error repair
            state = {"answer": self.program(problem)["answer"], "rounds": self.rounds}
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if not state.get("answer") and self.failure:
            raise self.failure      # nothing was recovered: report it as an API error
        return state

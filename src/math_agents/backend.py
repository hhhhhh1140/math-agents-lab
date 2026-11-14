import json
import os
import time
from typing import Protocol

import httpx


class Backend(Protocol):
    def complete(self, role: str, instruction: str, context: dict) -> tuple[str, dict]: ...


class ChatBackend:
    """Chat Completions compatible API. Token counts are never estimated.

    Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried
    up to MATH_API_RETRIES times; the number of attempts is recorded in usage.
    """

    def __init__(self, max_tokens=2048, temperature=0.0):
        self.model = os.environ["MATH_MODEL"]
        # Optional per-role overrides, e.g. MATH_REFEREE_MODEL for a stronger
        # model that is only consulted when the cheap agents disagree.
        self.role_models = {role: os.environ[key].strip() for role, key in
                            [("reasoner", "MATH_REASONER_MODEL"), ("coder", "MATH_CODER_MODEL"),
                             ("referee", "MATH_REFEREE_MODEL")] if os.getenv(key, "").strip()}
        # Route only model requests through this proxy. Ignore inherited global
        # proxy/NO_PROXY variables so container routing is explicit and stable.
        self.proxy_url = os.getenv("MATH_PROXY_URL", "").strip() or None
        self.json_mode = os.getenv("MATH_JSON_MODE", "false").lower() == "true"
        self.client = httpx.Client(
            base_url=os.getenv("MATH_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/",
            headers={"Authorization": "Bearer " + os.environ["MATH_API_KEY"]},
            proxy=self.proxy_url,
            trust_env=False,
            timeout=httpx.Timeout(float(os.getenv("MATH_API_TIMEOUT", "120")), connect=15),
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = max(0, int(os.getenv("MATH_API_RETRIES", "2")))
        self.retry_wait = float(os.getenv("MATH_API_RETRY_WAIT", "2"))

    def _post(self, messages, json_mode, role=None):
        model = self.role_models.get(role, self.model)
        payload = {"model": model, "temperature": self.temperature,
                   "max_tokens": self.max_tokens, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post("chat/completions", json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == self.retries:
                    raise
                time.sleep(self.retry_wait * (attempt + 1))
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        choice = data["choices"][0]
        return choice["message"].get("content") or "", {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "attempts": attempt + 1, "finish_reason": choice.get("finish_reason"), "model": model,
        }

    def complete(self, role, instruction, context):
        """Structured call: the model must answer with a single JSON object."""
        return self._post([
            {"role": "system", "content": instruction + "\nReturn exactly one JSON object, no markdown fences."},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ], self.json_mode)

    def chat(self, role, system, user):
        """Free-text call: leaves the model room to reason step by step."""
        return self._post([{"role": "system", "content": system}, {"role": "user", "content": user}], False, role)

    def close(self):
        self.client.close()


class DemoBackend:
    """Scripted fixture, never a benchmark model. Only supports data/demo.jsonl."""
    model = "scripted-demo-NOT-A-BENCHMARK"

    def chat(self, role, system, user):
        answer = "12" if "A box" in user else "3"
        text = f"```python\nprint({answer})\n```" if role == "coder" else f"Scripted fixture. \\boxed{{{answer}}}"
        return text, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    def complete(self, role, instruction, context):
        answer = "12" if context["problem"].startswith("A box") else "3"
        outputs = {
            "planner": {"plan": "Translate the conditions into an equation, solve, then substitute."},
            "solver": {"code": f"print({answer})"},
            "finalizer": {"answer": answer},
            "direct": {"answer": answer},
            "single": {"code": f"print({answer})"} if not context.get("history") else {"answer": answer},
            "verifier": {"code": "print('PASS')", "reason": "Scripted test fixture only."},
            "reviewer": {"accept": True, "feedback": "Scripted test fixture only."},
        }
        return json.dumps(outputs[role]), {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

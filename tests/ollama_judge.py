"""
A DeepEval judge model backed by local Ollama.

DeepEval defaults to OpenAI, which would mean an API key, per-test cost, and
sending this corpus to a third party. Every other model in this workspace runs
locally; the evaluator should too.

    from tests.ollama_judge import OllamaJudge
    metric = FaithfulnessMetric(threshold=0.7, model=OllamaJudge())

## The one thing that makes this work

DeepEval's metrics ask the judge for **structured JSON** (verdict lists, scores,
reasons) and parse it. A judge that returns prose fails every metric with a
parse error rather than a low score, which looks like a broken metric instead of
a broken model.

Two defences here:

1. `format="json"` on the Ollama call, which constrains decoding to valid JSON.
2. `generate` accepts DeepEval's optional `schema` argument (a pydantic model)
   and validates into it, because newer DeepEval versions pass one and expect a
   model instance back rather than a string.

## Judge choice

`qwen2.5:7b` at temperature 0. Determinism matters more than eloquence: a judge
that scores the same input differently on two runs is not a measurement
instrument, and every metric here is used as a **regression gate**.

Be honest about the ceiling: a 7B judge cannot reliably grade claims it does not
itself understand. These metrics are for catching *regressions* between runs of
your own system, not for certifying absolute quality.
"""

from __future__ import annotations

import json
import os

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
JUDGE_MODEL = os.environ.get("DEEPEVAL_JUDGE", "qwen2.5:7b")


class OllamaJudge:
    """DeepEvalBaseLLM implementation over a local Ollama server."""

    def __init__(self, model: str = JUDGE_MODEL, url: str = OLLAMA_URL,
                 temperature: float = 0.0, timeout: float = 300.0):
        self.model = model
        self.url = url
        self.temperature = temperature
        self.timeout = timeout

    # ---- DeepEvalBaseLLM interface -------------------------------------

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"

    def generate(self, prompt: str, schema=None, **kwargs):
        text = self._chat(prompt, json_mode=schema is not None)
        if schema is None:
            return text
        return self._coerce(text, schema)

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        # Ollama is a local HTTP call; running it sync inside the async path is
        # simpler than maintaining a second client and costs nothing here.
        return self.generate(prompt, schema=schema, **kwargs)

    # ---- internals ------------------------------------------------------

    def _chat(self, prompt: str, json_mode: bool) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"
        r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    @staticmethod
    def _coerce(text: str, schema):
        """Parse the judge's reply into DeepEval's expected pydantic model."""
        try:
            return schema.model_validate_json(text)
        except Exception:
            pass
        # salvage the first {...} span if the model wrapped it in prose
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            try:
                return schema.model_validate(json.loads(text[a:b + 1]))
            except Exception as exc:
                raise ValueError(
                    f"judge returned unparseable JSON for {schema.__name__}: "
                    f"{text[:200]}"
                ) from exc
        raise ValueError(f"judge returned no JSON for {schema.__name__}: {text[:200]}")


def ollama_available(url: str = OLLAMA_URL, model: str = JUDGE_MODEL) -> bool:
    """True if Ollama is up and the judge model is pulled.

    Used to skip DeepEval tests rather than fail them: a missing local server is
    an environment gap, not a regression in the code under test.
    """
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        return any(m["name"].startswith(model.split(":")[0])
                   for m in r.json().get("models", []))
    except Exception:
        return False

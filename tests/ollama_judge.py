"""
The DeepEval judge model, backed by local Ollama.

DeepEval defaults to OpenAI, which would mean an API key, per-test cost, and
sending this corpus to a third party. Every other model in this workspace runs
locally; the evaluator should too.

    from tests.ollama_judge import make_judge
    metric = FaithfulnessMetric(threshold=0.7, model=make_judge())

## Use the built-in, not a hand-rolled adapter

DeepEval 4.x ships `deepeval.models.OllamaModel`, which already handles the part
that actually matters: metrics ask the judge for **structured JSON** (verdict
lists, scores, reasons) and parse it into pydantic schemas. A judge that returns
prose fails every metric with a *parse error* rather than a low score — which
looks like a broken metric instead of a broken model.

I originally wrote a custom `DeepEvalBaseLLM` subclass for this before finding
the built-in. It is kept below as `CustomOllamaJudge` because it is a useful
reference for wrapping *any* local model DeepEval doesn't support — but
`make_judge()` returns the built-in, because reimplementing a
well-tested integration is how you inherit its bugs without its fixes.

(The custom one also had a subtle defect worth remembering: it was duck-typed
rather than subclassing `DeepEvalBaseLLM`, and DeepEval does an `isinstance`
check — so it failed at construction with
`TypeError: Unsupported type for model`.)

## Judge choice

`qwen2.5:7b` at temperature 0. Determinism matters more than eloquence: a judge
that scores the same input differently across runs is not a measurement
instrument, and every metric here is used as a **regression gate**.

The ceiling, stated honestly: a 7B judge cannot reliably grade claims it does not
itself understand. These metrics catch *regressions* between runs of your own
system; they do not certify absolute quality.
"""

from __future__ import annotations

import json
import os

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
JUDGE_MODEL = os.environ.get("DEEPEVAL_JUDGE", "qwen2.5:7b")


def make_judge(model: str = JUDGE_MODEL, url: str = OLLAMA_URL, temperature: float = 0.0):
    """The judge every DeepEval metric in this suite should use."""
    from deepeval.models import OllamaModel

    return OllamaModel(model=model, base_url=url, temperature=temperature)


def ollama_available(url: str = OLLAMA_URL, model: str = JUDGE_MODEL) -> bool:
    """True if Ollama is up and the judge model is pulled.

    Used to *skip* DeepEval tests rather than fail them: a missing local server
    is an environment gap, not a regression in the code under test.
    """
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        return any(m["name"].startswith(model.split(":")[0])
                   for m in r.json().get("models", []))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reference implementation, unused by the suite.
#
# Keep this for the case DeepEval does not support your runtime (vLLM, TGI, a
# bespoke endpoint). Note it MUST subclass DeepEvalBaseLLM -- duck typing fails
# an isinstance check inside `initialize_model`.
# ---------------------------------------------------------------------------

try:
    from deepeval.models import DeepEvalBaseLLM

    class CustomOllamaJudge(DeepEvalBaseLLM):
        """Hand-rolled equivalent of the built-in, for unsupported runtimes."""

        def __init__(self, model: str = JUDGE_MODEL, url: str = OLLAMA_URL,
                     temperature: float = 0.0, timeout: float = 300.0):
            self.model_name = model
            self.url = url
            self.temperature = temperature
            self.timeout = timeout
            super().__init__(model_name)

        def load_model(self):
            return self.model_name

        def get_model_name(self) -> str:
            return f"ollama/{self.model_name}"

        def generate(self, prompt: str, schema=None, **kwargs):
            text = self._chat(prompt, json_mode=schema is not None)
            return text if schema is None else self._coerce(text, schema)

        async def a_generate(self, prompt: str, schema=None, **kwargs):
            return self.generate(prompt, schema=schema, **kwargs)

        def _chat(self, prompt: str, json_mode: bool) -> str:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            if json_mode:
                # constrain decoding to valid JSON, or metrics fail on parsing
                payload["format"] = "json"
            r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()

        @staticmethod
        def _coerce(text: str, schema):
            try:
                return schema.model_validate_json(text)
            except Exception:
                pass
            a, b = text.find("{"), text.rfind("}")
            if a != -1 and b > a:
                return schema.model_validate(json.loads(text[a:b + 1]))
            raise ValueError(f"judge returned no JSON for {schema.__name__}: {text[:200]}")

except ImportError:  # pragma: no cover - deepeval not installed
    CustomOllamaJudge = None  # type: ignore

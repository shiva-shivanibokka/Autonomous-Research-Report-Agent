"""
NLI support judges (no LLM). Runs on CPU with models already in the local
Hugging Face cache (HF_HUB_OFFLINE=1, nothing is downloaded):

  primary : cross-encoder/nli-deberta-v3-large
  second  : microsoft/deberta-xlarge-mnli

A claim is judged against an evidence text by sliding a window over the
evidence (the models take 512 tokens) and taking the maximum entailment
probability over windows. Decision rule, fixed in advance and not tuned:
SUPPORTED iff entailment is the argmax label in at least one window.

Needs torch + transformers; this repo's requirements do not include them, so
these scripts are run with an existing interpreter that has them (see
RESULTS.md, "Reproduce").
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("USE_TF", "0")  # base env has a broken TensorFlow install
os.environ.setdefault("USE_TORCH", "1")

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

MODELS = {
    "deberta-v3-large": "cross-encoder/nli-deberta-v3-large",
    "deberta-xlarge-mnli": "microsoft/deberta-xlarge-mnli",
}
MAX_LEN = 512
WINDOW_TOKENS = 380  # evidence tokens per window (leaves room for the claim)
STRIDE = 300
MAX_WINDOWS = 12  # evidence beyond ~3,600 tokens is not read


class NLIJudge:
    def __init__(self, name: str):
        self.name = name
        path = MODELS[name]
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = labels["entailment"]

    def _windows(self, evidence: str) -> list[str]:
        ids = self.tok(evidence, add_special_tokens=False)["input_ids"]
        if not ids:
            return [""]
        out = []
        for start in range(0, len(ids), STRIDE):
            out.append(self.tok.decode(ids[start : start + WINDOW_TOKENS]))
            if start + WINDOW_TOKENS >= len(ids) or len(out) >= MAX_WINDOWS:
                break
        return out

    @torch.inference_mode()
    def score(self, evidence: str, claim: str) -> dict:
        wins = self._windows(evidence)
        best_p, any_argmax = 0.0, False
        for i in range(0, len(wins), 4):
            batch = wins[i : i + 4]
            enc = self.tok(
                batch,
                [claim] * len(batch),
                truncation="only_first",
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            )
            probs = self.model(**enc).logits.softmax(-1)
            best_p = max(best_p, float(probs[:, self.ent].max()))
            any_argmax = any_argmax or bool((probs.argmax(-1) == self.ent).any())
        return {"p_entail": best_p, "supported": any_argmax, "n_windows": len(wins)}

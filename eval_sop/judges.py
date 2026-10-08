"""
LLM judges (local, via Ollama). Every label they produce is an LLM judgment,
not a human label, and is stored as such.

Two judges from model families different from the generator (qwen2.5):
  llama3.1:8b (Ollama, Q4_K_M) and gemma2:9b (Ollama, Q4_0)
Both run with temperature 0, seed 0, num_ctx 8192.
"""

from __future__ import annotations

from eval_sop import common

EVIDENCE_CHARS = 6000

SUPPORT_SYSTEM = (
    "You are a strict attribution judge. You decide whether a claim is fully "
    "supported by a piece of evidence text. Use only the evidence, not your own knowledge."
)

SUPPORT_USER = """Question / context: {context}

Claim: {claim}

Evidence:
\"\"\"
{evidence}
\"\"\"

Is every part of the claim supported by the evidence above? Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."""

CORRECT_SYSTEM = "You are a strict grader for a question-answering benchmark."

CORRECT_USER = """Question: {question}

Ground-truth answer: {reference}

Predicted answer: {prediction}

Is the predicted answer correct? It is correct if it gives the same final answer as the ground truth: paraphrases, equivalent formatting, and extra correct detail are fine. It is incorrect if the final answer differs, is missing, refuses, or hedges between several different answers. Answer with exactly one word: CORRECT or INCORRECT."""


def _parse(text: str, pos: str, neg: str) -> bool | None:
    t = (text or "").strip().upper().replace("-", "_")
    if neg in t or t.startswith("NOT") or t.startswith("IN"):
        return False
    if pos in t:
        return True
    return None


async def judge_support(model: str, context: str, claim: str, evidence: str) -> dict:
    msgs = [
        {"role": "system", "content": SUPPORT_SYSTEM},
        {
            "role": "user",
            "content": SUPPORT_USER.format(
                context=context or "(none)",
                claim=claim,
                evidence=(evidence or "")[:EVIDENCE_CHARS],
            ),
        },
    ]
    text, *_ = await common.chat(
        model, msgs, max_tokens=8, seed=0, temperature=0.0,
        num_ctx=common.JUDGE_NUM_CTX, agent="judge_support"
    )
    label = _parse(text, "SUPPORTED", "NOT_SUPPORTED")
    return {"raw": text, "supported": label}


async def judge_correct(model: str, question: str, reference: str, prediction: str) -> dict:
    msgs = [
        {"role": "system", "content": CORRECT_SYSTEM},
        {
            "role": "user",
            "content": CORRECT_USER.format(
                question=question, reference=reference, prediction=prediction or "(no answer)"
            ),
        },
    ]
    text, *_ = await common.chat(
        model, msgs, max_tokens=8, seed=0, temperature=0.0,
        num_ctx=common.JUDGE_NUM_CTX, agent="judge_correct"
    )
    return {"raw": text, "correct": _parse(text, "CORRECT", "INCORRECT")}

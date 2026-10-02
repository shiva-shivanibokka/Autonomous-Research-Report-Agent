"""
Build the two fixed input sets for the evaluation. Run once; outputs are committed.

  eval_sop/data/questions.jsonl          FRAMES sample (+ open-ended questions)

Provenance
----------
FRAMES: google/frames-benchmark, file test.tsv (824 rows), downloaded from
  https://huggingface.co/datasets/google/frames-benchmark/resolve/main/test.tsv
  We shuffle the row indices with random.Random(20261001) and keep that order;
  the evaluation uses a prefix of it, so shrinking n never changes which
  questions come first. Reference answers are FRAMES' human-written "Answer".
Open-ended: five questions with NO reference answer. The first is the query of
  the repo's own showcase run (frontend/public/demo/run.json); the other four
  were written for this evaluation. They are used only for claim-level metrics.
(An AttributionBench sample was downloaded in an earlier session and then
removed: it was not an approved download. Judge validation now uses the
human-labelled RAGTruth copy already on this machine; see nli_validate.py.)
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FRAMES_URL = "https://huggingface.co/datasets/google/frames-benchmark/resolve/main/test.tsv"

OPEN_ENDED = [
    (
        "open_0",
        "Do AI coding assistants actually make software developers more productive?",
        "repo showcase query (frontend/public/demo/run.json)",
    ),
    (
        "open_1",
        "What is the scientific evidence on intermittent fasting for weight loss "
        "compared with continuous calorie restriction?",
        "written for this eval",
    ),
    (
        "open_2",
        "How have lithium-ion battery pack prices changed over the past decade "
        "and what drove the change?",
        "written for this eval",
    ),
    (
        "open_3",
        "What are the main approaches to carbon capture and storage, and what are "
        "their current costs and deployment levels?",
        "written for this eval",
    ),
    (
        "open_4",
        "What does published research say about the effect of remote work on "
        "employee productivity?",
        "written for this eval",
    ),
]


def _get(url: str) -> bytes:
    for attempt in range(5):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:  # noqa: BLE001 - retry any transient failure
            time.sleep(2 + attempt * 2)
    raise RuntimeError(f"could not fetch {url}")


def build_frames() -> list[dict]:
    raw = _get(FRAMES_URL)
    sha = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines(), delimiter="\t"))
    order = list(range(len(rows)))
    random.Random(20261001).shuffle(order)
    out = []
    for rank, idx in enumerate(order):
        r = rows[idx]
        out.append(
            {
                "qid": f"frames_{idx}",
                "rank": rank,
                "question": r["Prompt"].strip(),
                "reference": r["Answer"].strip(),
                "reasoning_types": r.get("reasoning_types", ""),
                "source": "google/frames-benchmark test.tsv",
                "source_row": idx,
                "source_sha256": sha,
            }
        )
    return out


def main() -> None:
    DATA.mkdir(exist_ok=True)
    frames = build_frames()
    with open(DATA / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in frames:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
        for qid, text, prov in OPEN_ENDED:
            f.write(
                json.dumps(
                    {
                        "qid": qid,
                        "rank": None,
                        "question": text,
                        "reference": None,
                        "source": prov,
                    }
                )
                + "\n"
            )
    print(f"frames={len(frames)} open={len(OPEN_ENDED)}")


if __name__ == "__main__":
    sys.exit(main())

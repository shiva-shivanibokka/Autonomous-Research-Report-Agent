# SOP evaluation — Autonomous Research Report Agent

Branch `sop-eval`, based on `main` @ `c0ae667`. Work done 2026-10-01 → 2026-10-04.

**Status.** The planned comparison (conditions a–d on FRAMES) **has not been
run yet**, so this file contains **no accuracy number for the pipeline**. No
model was available to run it: the shared local Ollama server produced no
completions, and paid APIs were out of scope until a budget was approved.

What exists:
- a judge validated against human labels;
- an audit of the one real end-to-end run in the repo;
- two pipeline fixes, each reproduced by a failing test first;
- a paid-run harness with hard spend controls, tested only against a fake
  client.

§6 describes the run, which is designed to stay under **$8**.

---

## 1. Setup

| Item | Value |
|---|---|
| Code under test | `agents/` at `c0ae667` plus the fixes in the Change log |
| Pipeline run audited | `frontend/public/demo/run.json`. Model `claude-sonnet-4-5`, `max_rounds=2`, recorded 2026-08-24. Query: *"Do AI coding assistants actually make software developers more productive?"* |
| NLI judges (no LLM) | `cross-encoder/nli-deberta-v3-large` and `microsoft/deberta-xlarge-mnli`. Both were already in the local HF cache and run offline on CPU (`OMP_NUM_THREADS=2`). Evidence is read in 380-token windows, stride 300, max 12 windows. **SUPPORTED iff entailment is the argmax in at least one window.** This rule was written down before any judging and never tuned. |
| Judge validation data | RAGTruth test split (human span-level hallucination labels), local copy, sha256 `2fc4fb70…3bbd`. 300 balanced sentences from the QA and Summary tasks, 75 per task × label, drawn with `random.Random(2026)`. Construction is in `eval_sop/nli_validate.py`. |
| Question set (built, not yet run) | FRAMES: Krishna et al., 2024, *Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation* (arXiv:2409.12941). Hugging Face dataset `google/frames-benchmark`, file `test.tsv`, sha256 `4255093c…69ff`. All 824 rows are kept, in a fixed order from `random.Random(20261001)`; the eval uses a prefix of that order. Plus 5 open-ended questions; provenance is given per row in `eval_sop/data/questions.jsonl`. |
| Statistics | Percentile bootstrap 95% CI (2,000–10,000 resamples, seed 12345). Cohen's κ, balanced accuracy, AUROC (sklearn). Exact McNemar for paired correctness. Rogan-Gladen correction for judge error. |
| Python | Repo dependencies: a venv at `%TEMP%\sopv` (see Deviations). NLI scripts: the existing `anaconda3` base env (torch + transformers, TensorFlow disabled). |

## 2. Results

### 2a. Citation-support judge vs human labels (RAGTruth, n = 300 balanced sentences)

| Judge | Balanced acc. [95% CI] | κ vs human [95% CI] | AUROC of p(entail) [95% CI] | Sensitivity / specificity | Predicted "supported" |
|---|---|---|---|---|---|
| deberta-v3-large (listed first before results were committed) | **0.640** [0.589, 0.692] | 0.280 [0.177, 0.382] | 0.725 [0.664, 0.784] | 0.47 / 0.81 | 33% |
| deberta-xlarge-mnli | **0.723** [0.673, 0.772] | 0.447 [0.347, 0.542] | 0.769 [0.712, 0.823] | 0.78 / 0.67 | 56% |
| Inter-judge (same 300 items) | raw agreement 0.697 | κ = 0.415 | | | |

Balanced accuracy by task:
- deberta-v3-large: 0.69 on QA, 0.59 on Summary.
- deberta-xlarge-mnli: 0.77 on QA, 0.67 on Summary.

Raw labels are in `eval_sop/results/nli_ragtruth_raw.jsonl`.

Both judges beat chance, but both are weak. v3-large misses about half of the
truly supported sentences. xlarge is better on every metric. I report both
rather than switching to the better one after seeing the results.

### 2b. Audit of the recorded showcase run (n = 1 run)

**Citation support.**
- The report has 38 sentences with an inline `(https://…)` citation: 49
  sentence–URL pairs over 17 distinct URLs.
- Every cited page was re-fetched on 2026-10-02 with the pipeline's own
  scraper.
- 5 of the 17 URLs could not be fetched (Playwright is not installed; see §4).
  That excluded 12 of the 38 sentences.
- **n = 26 sentences from one demo run** remain.

| Judge | Raw rate: sentence entailed by ≥1 cited page [95% CI] | Rogan-Gladen corrected rate [95% CI] |
|---|---|---|
| deberta-v3-large | 0.38 [0.19, 0.58] | 0.68 [0.00, 1.00] |
| deberta-xlarge-mnli | 0.69 [0.50, 0.85] | 0.80 [0.36, 1.00] |

The corrected rate adjusts each judge's raw rate using its RAGTruth sensitivity
and specificity. Its CI resamples the validation items and the showcase
sentences together. The correction assumes the judges' error rates carry over
from RAGTruth to these web pages, which I have not tested.

Bottom line: raw judge rates are **roughly 40–70%**, and the corrected point
estimates are **about 0.68 and 0.80**, with very wide intervals.

**Self-reported quality block vs the recorded data.** These are deterministic
checks; no judge is involved.

| Check | Reported (LLM self-report) | Computed from the same run | Verdict |
|---|---|---|---|
| contradiction_rate | 0.12 | 0 of 60 claims contested, 0 contradictions mapped → **0.00** | Inconsistent. The Critic generates this number (`critic_agent.py:116` prompt) and it is passed through unchanged (`writer_agent.py:158` at c0ae667; `:156` on this branch). |
| source_diversity_score | 0.52 | Listed citations: 25 domains / 30 URLs = 0.83. Inline-cited: 12 domains / 17 URLs = 0.71 | Neither obvious definition reproduces it. n = 1, so this is not a calibration estimate. |
| coverage_score, overall_quality_score | 0.75, 0.58 | Not recomputable from what the run stored | Also LLM self-report. All **four** quality scores come from the Critic LLM; none is computed. |
| Fact-checking | Round 1 flagged 5 claims; the final round flagged 0 | Fact-checker **skipped**; verified = 0 | Confirmed. In a 2-round run, round-1 flags go to re-research, never to the fact-checker (`test_two_round_run_never_fact_checks_round_one_flags`). |
| Citations | `total_sources_consulted = 81` | The report lists 30 (`citations[:30]`, `writer_agent.py:463`). **13 of the 17 URLs cited in the text are missing from that list.** | New finding. Not fixed; see §10. |
| Token budget | 80,000 | Used 78,591 | Under budget in this run. The code does not enforce a hard budget: `call_llm` only clamps each call's *output* to `max(256, remaining)` (`llm_client.py:301` at c0ae667, `:302` on this branch). |

### 2c. Conditions (a) closed-book, (b) search + summarise, (c) pipeline r=1, (d) pipeline r=2

**Not run.** The harness exists, is described in §6, and has only been tested
offline.

### 2d. Calibration of the Critic's scores and the per-claim confidence labels

**Not measured.** It needs the (c)/(d) runs. The analysis is already
implemented in `score.py analyze`:
- AUROC of the confidence ordinal vs NLI support;
- AUROC of Critic quality and coverage vs correctness;
- Spearman correlations.

## 3. What the numbers do and don't support

**Supported:**
- All four of the pipeline's quality scores are LLM self-report. In the one
  recorded run, one of them (contradiction_rate) contradicts the run's own
  data.
- The fact-checker did not run in the showcase. By construction, it never sees
  round-1 flags when `max_rounds=2`.
- Round-1 claims were dropped before the fix. This is reproduced by a test.
- An off-the-shelf NLI judge, checked against human attribution labels, reaches
  balanced accuracy 0.64–0.72 on RAGTruth sentences.

**Not supported:**
- Any accuracy, citation, or calibration advantage of the pipeline over a
  baseline. That comparison has not been run.
- Generalising the showcase support rates beyond one query.

## 4. Threats to validity

- **One pipeline run.** The showcase query was chosen by the author as a demo.
- **Noisy judges.** κ vs human is 0.28 and 0.45. Validation used RAGTruth
  (news and QA passages), not the web pages audited here, so the Rogan-Gladen
  correction rests on an untested transfer assumption.
- **Page drift and fetch failures.**
  - Pages were fetched about 5 weeks after the run.
  - 5 of 17 URLs could not be fetched because **Playwright is not installed**.
    The deployed pipeline falls back to Playwright for JavaScript-heavy pages;
    this eval's scraper does not.
  - The paid run in §6 has the same scraper difference, and I disclose it rather
    than install a browser.
  - The scraper keeps 8,000 characters per page, and the analyst sees only the
    first 3,000.
- **Heuristic extraction.** The regex catches 49 of 53 URL mentions.
  RAGTruth sentence labels are derived: any overlap with a labelled span marks
  the whole sentence unsupported.
- **Second judge.** It was chosen because it was already cached, not for any
  prior evidence of quality.
- **For the planned paid run:**
  - Haiku 4.5 is not the deployed configuration (Sonnet 4.5).
  - "Seeds" are replicate indices only: the pipeline does not forward
    temperature (`llm_client.py:288` at c0ae667, `:289` on this branch) and the API takes no seed.
  - Correctness labels come from local LLM judges, plus a deterministic string
    match.
  - c and d share their round-1 calls through the response cache, so they are
    not independent samples.

## 5. SOP-ready sentences (strictly true as of this branch)

1. "I audited my multi-agent research pipeline and found that four of its
   quality scores were self-reported by the critic LLM rather than computed.
   In the recorded demo run, the reported contradiction rate (0.12)
   contradicted the run's own data (0 contested claims). I reproduced, with a
   failing end-to-end test, a bug that silently dropped the first research
   round's claims, and fixed it."
2. "To check citation faithfulness without hand-labelling, I validated two
   off-the-shelf NLI models against RAGTruth's human labels (balanced accuracy
   0.64 and 0.72 on 300 sentences). On the 26 checkable cited sentences of one
   demo report (12 of 38 were excluded because the cited page could not be
   fetched), the judges found roughly 40–70% supported by the cited page.
   Corrected for each judge's measured error rates, that is about 0.68 and
   0.80, with wide confidence intervals."

Do **not** claim any gain over a baseline until §2c has numbers.

## 6. Paid run design (≤ $8 target, $12 hard cap) — prepared, not run

- **Model.** Every pipeline agent and every baseline uses
  `claude-haiku-4-5-20251001` ($1 / $5 per 1M tokens), one model for the whole
  comparison. The transport refuses any other model id; that also keeps
  thinking-only 5.x models out, since thinking is not handled.
- **Correctness judges.** Local Ollama `llama3.1:8b` and `gemma2:9b` (LLM
  labels, disclosed), plus deterministic strict string match.
- **Citation support.** The two RAGTruth-validated NLI judges.
- **Questions.** The first 30 FRAMES questions in the fixed order, 1
  replicate.
- **Conditions.** Pass 1: (a), (b), (d). Pass 2: (c), only if budget remains.
  (c) replays d's identical round-1 calls from the cache; a test shows the
  round-1 orchestrator and analyst calls are not paid twice.
- **Primary analysis.** Exact McNemar on per-question correctness for d vs b
  and d vs a, intention-to-treat (an errored run counts as wrong). Also
  reported: c vs d, the bootstrap CIs, and citation support per condition.
- **Tavily.** Capped at 600 credits. Headroom on 2026-10-04 was 1,000 of
  1,000 (`eval_sop/tavily_usage.py`); that key's usage covers every project
  that uses it.

**Dry-run output** (`--backend anthropic --usd-cap 8 --dry-run`, no API calls):

| Design | Worst-case USD | Expected USD (estimate) | Tavily worst / expected |
|---|---|---|---|
| (a, b, d) × 30 | **$18.41 → REFUSED without admission control** | $4.64 | 660 / 420 |
| (a, b, d) × 30 with `--admission-control` | never exceeds the cap ($8) | $4.64 | ≤ 600 / 420 |
| (a, b, c, d) × 30 with `--admission-control` | never exceeds the cap ($8) | $5.71 | ≤ 600 / 480 |
| (a, b, d) × 12 | $7.36 (accepted) | $1.86 | 264 / 168 |

How the two estimates are built:
- **Worst case** assumes every call uses its full `max_tokens` (including the
  Writer's 16k-token draft plus its 24k-token retry), at most 8 sub-questions
  per round, and 2.5 characters per input token (`eval_sop/plan.py`).
- **Expected** is the showcase run's 78,591 tokens repriced at Haiku rates
  with an *assumed* 80/20 input/output split. It is an estimate, not a
  measurement.

**Recommended execution:**
1. `--canary`: one known-answer question through (a), (b), (d). Worst case
   about $0.61.
2. (a, b, d) × 30 with `--admission-control --usd-cap 8`. Runs go question by
   question, and a run starts only if the remaining budget covers that run's
   own worst case. So the $8 can never be crossed, and a cut-off design stays
   paired.
3. `--conditions c,d` with the same flags, to add (c) from whatever remains.

**Guarantees, all tested** against a fake Anthropic client in
`eval_sop/tests/test_anthropic_budget.py`, 18 tests in that file:
- **Ledger.** A persisted USD ledger records actual `usage` after every call.
- **Pre-call check.** Before every call, the ledger checks that money already
  spent + worst cases of calls in flight + this call's worst case ≤ the cap.
- **Stops the agents can't swallow.** A cap hit, a billing error, or a
  401/402/403 raises `BudgetStop`. It is a `BaseException`, and a process-wide
  flag backs it up, so the agents' `except Exception` cannot swallow it. The
  run in progress is not saved.
- **Retries.** The SDK's own retries are off. 429, 5xx and connection errors
  get at most 2 retries, honouring `retry-after`. Any other 4xx fails at once.
  A connection error is charged at its worst case, because billing is unknown.
- **Crash recovery.** Every response is cached as it arrives, so a crashed run
  resumes without paying twice.
- **Errored runs are marked.** A run is marked errored if it raised, if
  `fatal_error` or `errors` is set, or if any call used a model other than the
  pinned one.

## 7. Human labels

None are needed for any reported number:
- Correctness uses FRAMES reference answers.
- Support uses the NLI judges validated on RAGTruth.

An optional `claims_for_optional_human_review.csv` is written by
`score.py analyze` once runs exist. No headline result depends on it.

## 8. Reproduce

```bash
# repo deps (Python 3.12): pip install -r requirements.txt -r requirements-dev.txt python-dotenv
pytest tests -q            # 87 passed (product suite, incl. tests/unit/test_graph_e2e.py)
pytest eval_sop/tests -q   # 22 passed (harness: fake Anthropic client, caps, stats)
python -m eval_sop.smoke_test
# NLI (torch + transformers, models in the HF cache):
OMP_NUM_THREADS=2 python -m eval_sop.nli_validate          # needs eval_sop/cache/ragtruth/test.parquet
python -m eval_sop.showcase_audit fetch && OMP_NUM_THREADS=2 python -m eval_sop.showcase_audit judge && python -m eval_sop.showcase_audit analyze
# Paid run (needs approval + a key in the env file; nothing here prints keys):
python -m eval_sop.tavily_usage --env-file PATH
python -m eval_sop.run_conditions --backend anthropic --env-file PATH --conditions a,b,d --n-frames 30 --usd-cap 8 --dry-run
python -m eval_sop.run_conditions ... --canary
python -m eval_sop.run_conditions ... --admission-control
python -m eval_sop.score judge && python -m eval_sop.score support && python -m eval_sop.score analyze
```

## 9. Change log

Each entry gives what changed, why, the evidence, and what was preserved.

1. **Carry Critic-approved claims across rounds** (`befe899`).
   - *What:* added `ResearchState.carried_claims` and `collected_claims()`.
     `increment_round` fills `carried_claims` from `approved_claims` before
     clearing `analyst_outputs`. The Critic and the Writer read
     `collected_claims()`.
   - *Why:* `graph.py:51-53` (c0ae667) dropped round-1 claims while keeping
     their citations.
   - *Evidence:* `tests/unit/test_graph_e2e.py` fails at c0ae667 and passes
     after the fix (`eval_sop/evidence/repro_round1_claim_loss_*.txt`).
   - *Preserved:* all comments; flagged round-1 claims are still dropped by
     design.
   - *Note:* to reproduce, I temporarily ran `git checkout -- agents` in the
     worktree and re-applied my own patch. This happened before the
     no-destructive-git rule was given.
2. **Tests documenting fact-checker scope** (`befe899`). No code change.
3. **Price `claude-sonnet-5` at $2/$10** (`2c2c825`).
   - *What/why:* `llm_client.py` listed it at the Sonnet 4.x rate ($3/$15).
   - *Evidence:* `test_compute_cost_sonnet_5_price` fails before and passes
     after (`eval_sop/evidence/price_sonnet5_*.txt`). Sonnet 5.5 is priced via
     the same prefix.
4. **Eval harness** (`f225e07`, `b01aef8`): datasets, transports, caches,
   scoring, offline smoke test. No pipeline logic changed.
5. **Removed an unapproved dataset** (`883f669`). The AttributionBench sample
   is still in the history of `f225e07`; I did not rewrite history.
6. **NLI judges + RAGTruth validation** (`4f8e517`). **Showcase audit**
   (`96292bc`).
7. **Hard spend controls** (`a4ce714`): the Anthropic transport, USD ledger,
   `BudgetStop`, bounded retries, response cache, fail-visibly, dry-run,
   canary and admission control.
   - *Why:* review findings. The Tavily cap was swallowed by
     `search_agent.py:61` / `fact_checker_agent.py:54`, and runs with
     `fatal_error` were saved as successes.
   - *Evidence:* both reproduced by `eval_sop/tests/test_repro_review.py`
     before the fix (`eval_sop/evidence/repro_review_findings_BEFORE.txt`);
     both pass after (`…_AFTER.txt`).
   - *Bug the tests caught during development:* routing the per-job creds as
     provider `"anthropic"` sent `call_llm` down its native branch with no
     client. The transport now registers as `eval-<backend>`.
   - *Scope:* this is one commit containing several related controls. They
     share files and tests, so I did not split it further.
8. **McNemar (exact, intention-to-treat)** (`8b6f04c`). **Rogan-Gladen
   correction** (`86427b2`). **Tavily headroom check and keys via
   `--env-file`** (`a3016e9`). The hardcoded path to the original repo's
   `.env` is gone.
9. Ollama model aliases (`sop-qwen`, `sop-llama`, `sop-gemma`) were briefly
   created on the shared server on day 1, then deleted.

## 10. Proposed, not done

- Compute `contradiction_rate` and `source_diversity_score` in code
  (implementations are in `eval_sop/quality_metrics.py`), and label the
  Critic's four scores as self-reported in `QualityReport`. This changes a
  public schema and the UI.
- Writer: stop truncating the citation list to 30 while the text cites URLs
  beyond it (`writer_agent.py:257,463`).
- Fact-check flags from every round, or document that only final-round flags
  are checked.
- Enforce the token budget: stop when `remaining <= 0`, and count input tokens.
- Triangulation (`analyst_agent.py:97-116`) trusts the LLM's own
  `supporting_sources` count. Count distinct cited URLs or domains instead.
- README corrections (README not edited): the "hard token budget" is not
  enforced, and the quality metrics are LLM self-assessments.
- Product defaults (`ReportRequest.model = "claude-sonnet-4-5"`) are unchanged.
  The eval pins its own model.

## Deviations from the brief

- **Virtual environment.** The original repo had no `.venv`, and creating one
  there would have written to the read-only tree. Worktree and scratchpad
  paths exceed Windows MAX_PATH for the `anthropic` wheel. So dependencies went
  into `%TEMP%\sopv`, which can be deleted. `python-dotenv` was added there on
  2026-10-04.
- **NLI environment.** The NLI scripts ran in the existing anaconda base env,
  with no installs.
- **CPU-only Ollama test.** On day 1 I briefly started a private CPU-only
  `ollama serve` on port 11500 to measure speed (3.5 tok/s), then stopped it.

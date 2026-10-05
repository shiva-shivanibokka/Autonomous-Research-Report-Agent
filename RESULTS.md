# SOP evaluation — Autonomous Research Report Agent

Branch `sop-eval`, based on `main` @ `c0ae667`. Work done 2026-10-01 → 2026-10-04.

**Status.** The planned comparison (conditions a–d on FRAMES) **has not been
run yet**, so this file contains **no accuracy number for the pipeline**. No
model was available to run it, for two reasons:
- **Local Ollama.** On 2026-10-01 the shared server produced no completions.
  On 2026-10-04 a single `/api/tags` request returned HTTP 200 in 1.8 s and
  listed `qwen2.5:7b`, `llama3.1:8b` and `gemma2:9b`. No generation was tested
  then, and the slot was reserved for another eval.
- **Paid APIs.** These were out of scope until a budget was approved.

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
| NLI judges (no LLM) | `cross-encoder/nli-deberta-v3-large` and `microsoft/deberta-xlarge-mnli`. Both were already in the local HF cache and run offline on CPU (`OMP_NUM_THREADS=2`). Evidence is read in 380-token windows, stride 300, max 12 windows. **SUPPORTED iff entailment is the argmax in at least one window.** The rule was not tuned on the validation data. |
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
sentences together. The correction assumes that each judge's sensitivity and,
above all, its **specificity** transfer from RAGTruth (news and QA passages)
to long scraped web pages. That assumption is **untested**.

The corrected numbers are reported here only. They are too uncertain for the
SOP: v3-large's interval is [0.00, 1.00], which says nothing. The usable
statement is the raw one: **roughly 40–70%** of checkable cited sentences are
judged supported, depending on the judge.

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
  - Correctness labels come from local LLM judges (`llama3.1:8b`,
    `gemma2:9b`), **which have not been validated against human correctness
    labels**. A deterministic strict string match is reported beside them.
    Their agreement with each other and with string match is computed.
  - **Scraper bias against the pipeline.** Only (c) and (d) fetch pages; (b)
    uses Tavily snippets and (a) uses nothing. Without Playwright,
    JavaScript-heavy pages fail for (c)/(d) only, so the missing fallback
    biases the comparison *against* the pipeline. Failed fetches are not
    cached across invocations. Within one invocation they are reused, so (c)
    and (d) see identical inputs.
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
   fetched), the two judges found roughly 40–70% of them supported by the page
   they cite."

Do **not** claim any gain over a baseline until §2c has numbers.

## 6. Paid run design ($8 hard cap) — prepared, not run

- **Model.** Every pipeline agent and every baseline uses
  `claude-haiku-4-5-20251001` ($1 / $5 per 1M tokens), one model for the whole
  comparison. The transport refuses any other model id; that also keeps
  thinking-only 5.x models out, since thinking is not handled.
- **Correctness judges.** Local Ollama `llama3.1:8b` and `gemma2:9b` (not
  human-validated; LLM
  labels, disclosed), plus deterministic strict string match.
- **Citation support.** The two RAGTruth-validated NLI judges.
- **Questions.** The first 30 FRAMES questions in the fixed order, 1
  replicate.
- **Conditions.** (a), (b), (d) and (c), in **one invocation**, ordered
  a → b → d → c per question.
  - (c) replays d's identical round-1 calls from the response cache; a test
    shows they are not paid twice.
  - The sharing needs (c) and (d) in the same process. Failed page fetches are
    retried in later invocations, so a later (c) pass could see different
    pages than (d) saw.
- **Scoring policy, fixed before any paid run.** These are separate from the
  spend controls below.
  - *Primary analysis:* exact McNemar on per-question correctness, d vs b and
    d vs a.
  - A run counts as **wrong** only if it raised, if the pipeline set
    `fatal_error`, or if a non-pinned model was used.
  - Runs with non-fatal pipeline `errors` are **scored normally**. Examples
    are the Writer's "malformed JSON; raw output preserved" and a schema
    mismatch: the report is still there and still answers. These runs are
    flagged in `nonfatal_errors`.
  - *Sensitivity analysis:* a strict intention-to-treat McNemar
    (`mcnemar_itt_strict`) that also counts those runs as wrong.
  - Also reported: c vs d, the bootstrap CIs, and citation support per
    condition.
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

**The cap bounds the spend, not the completion.** The design the runbook
actually documents — `(a, b, c, d) × 30` with `--admission-control` — has an
unconditional worst case of **$27.16** against a **$8** cap (and 810 Tavily
credits against 600). Admission control does not shrink that number; it only
refuses to start a question whose own worst case would cross the remaining
headroom. So the guarantee is one-sided: **$8 is never exceeded, but the run is
not guaranteed to finish.** Which of the two safe things happens depends on the
flag, and an earlier version of this paragraph named only the less likely one.
**Without** `--admission-control` the design check refuses to start at all: exit
**2**, nothing spent — measured, `REFUSED` on both the $27.16 and the 810-credit
bound. **With** it, the run starts and stops partway if calls bill near their
worst case throughout, exiting **4** with a partial, still-paired result set, and
the remaining questions are simply not run. Completing all four conditions × 30
questions depends on actual billing coming in near the expected **$5.71**, as
it did in the showcase run; it is an expectation, not a bound. Budget for
re-running the remainder under a second cap rather than treating exit 4 as a
failure.

How the two estimates are built:
- **Worst case** assumes every call uses its full `max_tokens` (including the
  Writer's 16k-token draft plus its 24k-token retry), at most 8 sub-questions
  per round, and 2.5 characters per input token (`eval_sop/plan.py`).
- **Expected** is the showcase run's 78,591 tokens repriced at Haiku rates
  with an *assumed* 80/20 input/output split. It is an estimate, not a
  measurement.

**Recommended execution:**
1. `--canary`: one known-answer question through (a), (b), (d). Worst case
   about $0.61. Its output goes to `canary_runs/`, never into the analysis.
2. `python -m eval_sop.score judge --canary`. This runs the local correctness
   judges on the canary output alone, checking them end to end after the
   canary and before the main spend. It needs the Ollama slot.
3. `--conditions a,b,c,d --n-frames 30 --admission-control --usd-cap 8`, in a
   single invocation.
   - A run starts only if the remaining budget covers that run's own worst
     case, so $8 can never be crossed and a cut-off design stays paired.
   - Exit codes: 0 done, 1 canary failed, 2 refused to start (design over
    the cap, an invalid cap, a corrupt ledger, or another evaluation process
    holds the lock), 3 cap or billing stop, 4 admission control stopped
    early.
  - The canary skips the design-level check (it is one question); every
    call's own check still applies.

**Guarantees, all tested** against fake clients in `eval_sop/tests/`:
- **Pending-row ledger.** Before every request attempt, a row is inserted with
  `status='pending'` and cost = that attempt's worst case.
  - A success updates it to the actual `usage` cost.
  - A model the provider substituted for the pinned one settles at the
    pinned worst case and stops the whole evaluation: its real price is not
    known here, and pricing it at the pinned rate would have let repeated
    substitution authorise several times the cap.
  - An HTTP-level 429 or 4xx rejection sets it to $0.
  - Every other exit keeps the worst-case charge, deliberately over-counting:
    - HTTP-level 5xx/529, labelled `http_<code>` and retried;
    - mid-stream errors, labelled `stream_error:<type>` and retried; this
      includes a status-200 SSE `overloaded_error`;
    - raw transport errors from `httpx` or from `httpx2`, which the SDK uses,
      such as `ReadTimeout` and `RemoteProtocolError`;
    - KeyboardInterrupt, cancellation, a crash.
  - Fakes raise both on stream open and inside `get_final_message()`.
- **Pre-call check.** Money already spent, including pending rows, plus this
  call's worst case must be ≤ the cap.
  - The check and the pending insert are one `BEGIN IMMEDIATE` transaction,
    committed before the request is sent.
  - 20 concurrent calls against a $0.05 cap, and 4 processes against a $0.50
    cap, never exceed it.
  - The cap must be a finite number in (0, **$8**]. `--usd-cap 12`,
    `SOP_USD_CAP=12`, `nan`, `inf` and non-positive values are all refused.
    `nan` previously slipped through every comparison, including each
    per-call check, which disabled the cap altogether.
- **Worst-case input estimate.** max(characters / 2.5, UTF-8 bytes / 3), so
  CJK text is not undercounted.
- **One ledger and one process per project.**
  - The USD ledger, the Tavily credit ledger and a run lock live in
    `~/.sop_eval/research_report/`, outside every checkout. The directory is
    derived from `Path.home()` on every platform, with no environment-variable
    override, so the worktree and the main checkout share one ledger and no
    setting can point the spend at a fresh file. `SOP_LEDGER_PATH` is gone.
    Earlier rounds resolved this directory from `%LOCALAPPDATA%`, which a
    reviewer defeated: a shell with that variable pointed elsewhere saw an
    empty ledger — a fresh cap's worth of headroom, the canary's spend
    forgotten — and the run lock moved with it, so two paid runs could
    overlap. Fixed in round 5; asserted by `eval_sop/tests/test_state_lock.py`
    (the path is under `Path.home()`; it does not move when that variable is
    changed and both modules are reloaded; the lock resolves beside the
    ledger). The old location held 0 ledger rows and $0.0000 spent, and
    0 Tavily credits, so nothing was migrated — see §9.
  - `install()` takes an exclusive lock file for the life of the process. It
    is released on exit and on Ctrl-C. A second process is refused. A stale
    lock's error message says how to recover.
  - The jsonl export is atomic.
- **Tavily.** Credits are booked atomically (`UPDATE … v=v+?` inside
  `BEGIN IMMEDIATE`) before each search starts. Failed searches count.
  `install()` no longer stacks wrappers, which had charged one search 5
  credits in tests. 4 processes against a cap of 200 book exactly 200; before
  this fix the round-3 reviewer's script recorded 303 searches.
- **No bypass.** After `install()`, `llm_client.call_llm`'s raw
  `anthropic.AsyncAnthropic()` fallback refuses. The model-mismatch check
  uses the model id the API reports back.
- **Stops the agents can't swallow.** A cap hit, a billing error, or a
  401/402/403 raises `BudgetStop`. It is a `BaseException`, and a process-wide
  flag backs it up, so the agents' `except Exception` cannot swallow it. The
  run in progress is not saved.
- **Retries.** The SDK's own retries are off. 429, 5xx, stream errors and
  connection errors get at most 2 retries, honouring `retry-after`. Any other
  4xx fails at once.
- **Crash recovery.** Every response is cached as it arrives, so a crashed run
  resumes without paying twice.
- **Errored runs are marked.** A run is errored if it raised, if
  `fatal_error` is set, or if any call used a model other than the pinned one.
  Non-fatal errors are flagged, and scored as described above.

## 7. Human labels

None are needed for any reported number:
- Correctness uses FRAMES reference answers.
- Support uses the NLI judges validated on RAGTruth.

An optional `claims_for_optional_human_review.csv` is written by
`score.py analyze` once runs exist. No headline result depends on it.

## 8. Reproduce

```bash
# repo deps (Python 3.12): pip install -r requirements.txt -r requirements-dev.txt
# (requirements-dev.txt now also carries the eval harness's own imports —
#  numpy, scipy, scikit-learn, python-dotenv. Before round 5 they were
#  undeclared, so a venv built from this line alone could not even collect
#  eval_sop/tests.)
pytest tests -q            # 87 passed  (product suite, incl. tests/unit/test_graph_e2e.py)
pytest eval_sop/tests -q   # 78 passed (80 with anthropic 1.x / httpx2 installed)
python -m eval_sop.smoke_test
# NLI (torch + transformers, models in the HF cache):
OMP_NUM_THREADS=2 python -m eval_sop.nli_validate          # needs eval_sop/cache/ragtruth/test.parquet
python -m eval_sop.showcase_audit fetch && OMP_NUM_THREADS=2 python -m eval_sop.showcase_audit judge && python -m eval_sop.showcase_audit analyze
```

**Paid-run runbook.** `eval_sop/tests/test_runbook.py` runs every command in
the block below exactly as written, with `PATH` set to an env file, a fake
Anthropic client, fake search and pages, and a fake correctness judge. All of
them must exit 0.

**Where the paid key goes.** Only in the file passed as `--env-file`, which
sits outside the repository. Never in the repo's `.env`, and never exported
into the environment. Nothing here prints keys. The reason is that other code
in this repository spends without any ledger, cap or lock:
`scripts/record_demo_run.py` runs the real pipeline on `claude-sonnet-4-5`
($3/$15), and a job that reaches `api/worker.py` without a key lets the
pipeline's own client read `ANTHROPIC_API_KEY` from the environment. Only
`eval_sop/` meters spending.

Both paths are gated by default, so spending an exported key takes two
deliberate acts:
- `record_demo_run.py` refuses without `--i-want-to-spend-real-money`.
- The API rejects a keyless request with HTTP 400 unless
  `ALLOW_SERVER_KEY_FALLBACK` is explicitly set (`api/main.py:83`, enforced at
  `api/main.py:236`); it is off when unset. The residual risk is exporting the
  key *and* opting in.

**Accepted residual: the Celery task itself is ungated.** The keyless check
above lives only in the HTTP layer. `api/worker.py`'s task signature is
`api_key: str | None = None` (`api/worker.py:65`) with no gate of its own, and
it passes that value straight into the pipeline state (`api/worker.py:97`);
when it is `None` the pipeline's own client falls back to `ANTHROPIC_API_KEY`
from the worker process's environment. So anything that can publish to the
broker — not just the API — can start a spend that `eval_sop/`'s ledger, cap
and lock never see. This is *accepted, not fixed*: reaching it requires broker
access, which already implies the ability to run arbitrary tasks, and the paid
evaluation key is never exported into any environment (above), so the fallback
has nothing to find on this machine. Anyone deploying the worker with a key in
its environment should re-gate it at `api/worker.py` rather than rely on
`api/main.py:236`.

Only one evaluation process can run at a time (a lock under `~/.sop_eval/research_report/`).

<!-- runbook -->
```bash
python -m eval_sop.tavily_usage --env-file PATH
python -m eval_sop.run_conditions --backend anthropic --env-file PATH --conditions a,b,c,d --n-frames 30 --usd-cap 8 --admission-control --dry-run
python -m eval_sop.run_conditions --backend anthropic --env-file PATH --conditions a,b,d --usd-cap 8 --canary
python -m eval_sop.score judge --canary
python -m eval_sop.run_conditions --backend anthropic --env-file PATH --conditions a,b,c,d --n-frames 30 --usd-cap 8 --admission-control
python -m eval_sop.score judge
python -m eval_sop.score support  # torch env (NLI judges); stubbed in the test
python -m eval_sop.score analyze
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
3. **Price `claude-sonnet-5` at $2/$10** (`5af8be4`).
   - *What/why:* `llm_client.py` listed it at the Sonnet 4.x rate ($3/$15).
   - *Evidence:* `test_compute_cost_sonnet_5_price` fails before and passes
     after (`eval_sop/evidence/price_sonnet5_*.txt`). Sonnet 5.5 is priced via
     the same prefix.
4. **Eval harness** (`f75bf36`, `49fb4f8`): datasets, transports, caches,
   scoring, offline smoke test. No pipeline logic changed.
5. **Removed an unapproved dataset.** The AttributionBench sample was deleted
   in a follow-up commit, and the blob stayed in this branch's history. It has
   since been purged: someone else rewrote `sop-eval` (see "History rewrite"
   below), so that commit no longer exists on the branch.
6. **NLI judges + RAGTruth validation** (`169ac83`). **Showcase audit**
   (`48a7a73`).
7. **Hard spend controls** (`41903a3`): the Anthropic transport, USD ledger,
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
8. **McNemar (exact, intention-to-treat)** (`8d1cc00`). **Rogan-Gladen
   correction** (`659d65c`). **Tavily headroom check and keys via
   `--env-file`** (`ab80a2a`). The hardcoded path to the original repo's
   `.env` is gone.
9. Ollama model aliases (`sop-qwen`, `sop-llama`, `sop-gemma`) were briefly
   created on the shared server on day 1, then deleted.

**Round-2 review fixes.** Each one was first reproduced by a test that fails;
before/after output is in `eval_sop/evidence/`.

10. **Pending-row ledger** (`2e2d6c9`). Fixes three gaps: mid-stream failures,
    raw httpx errors, and interrupts were recorded at $0 or not at all. Also
    adds the fixed per-project ledger path. Evidence:
    `review2_ledger_*.txt`, 7 failing before.
11. **Token estimate** max(chars/2.5, bytes/3) (`fe80e56`). `token_estimate_*`.
12. **Tavily credits booked before the await; failures count; idempotent
    install** (`c0bf255`). `tavily_credits_*`.
13. **Failed page fetches not cached across invocations** (`bcff526`).
    `scrape_cache_*`.
14. **Scoring policy** (`5dfd630`). Non-fatal errors are scored; strict ITT is
    the sensitivity analysis. `scoring_policy_*`.
15. **Canary output kept apart and judgeable with `score judge --canary`**
    (`a454d3e`). The conftest now blocks real Tavily clients in tests: one
    test had reached the network with a fake key. `canary_judging_*`.
16. **Atomic export; admission STOP exits 4** (`7da7a01` + `2e54dbf`).
    `7da7a01` was committed by mistake *before* the fix. A failed in-place
    edit did not stop the shell chain, so that commit holds only the failing
    tests and a mislabelled "after" file. `2e54dbf` applies the fix and
    corrects the evidence files. No history was rewritten.
17. **Docs** (this commit): SOP sentence 2 without the corrected rates, the
    specificity-transfer caveat, the judge-validation and scraper-bias
    disclosures, a softened claim about the NLI decision rule, and the
    scoring policy.

**Round-3 review fixes.** Each one was first reproduced by a failing test or
by the reviewer's scripts (`review3/rr/race.py`, `attack3.py`,
`httpx2_attack.py`); output is in `eval_sop/evidence/`.

18. **Hard maximum $8** (`8ac010d`). `hard_max_*`.
19. **Per-user state directory, run lock, atomic cross-process caps, and
    list-of-blocks content in the token estimate** (`eb4e3a6`).
    `race_BEFORE`, `attack3_BEFORE`, `list_content_BEFORE`,
    `state_lock_AFTER`. This is one commit: the state directory, lock and
    transactions are intertwined. The empty ledger that tests had created
    inside the worktree was deleted.
20. **Canary skips the design-level check; the runbook is tested verbatim**
    (`69bee6a`). `canary_runbook_*`. pytest's temporary directories now stay
    out of `%TEMP%`.
21. **Product CI lint gate kept green** (`0336164`).
    - `ruff.toml` excludes `eval_sop/`. This is the only change to product
      configuration.
    - Before it, `ruff check .` reported 144 findings and 24 unformatted
      files, all under `eval_sop/`.
    - `test_lint.py` runs the gate and checks `eval_sop` for pyflakes and
      syntax errors. `ci_lint_*`.
22. **httpx2 transport errors caught** (`0c04033`). `httpx2_*`.
23. **Model mismatch uses the reported model** (`c797561`).
    `model_reported_*`.
24. **HTTP-level 5xx/529 labelled `http_<code>`, still charged at worst**
    (`7119f93`). `http529_*`. The "before" test asserted $0 at the time; the
    final rule over-counts on purpose.
25. **Raw-client fallback blocked** (`1449ce7`). `no_bypass_*`.
26. **User-profile paths scrubbed from the evidence files** (`4f3380e`).
    `user_paths_*`. That commit changed only the working tree going forward:
    the account-name string stayed in 19 earlier commits of this branch, so
    the scrub was not yet a history property. It became one in round 5 — see
    "History rewrite" below. While doing the original scrub I reverted my own
    uncommitted first attempt on one evidence file with `git checkout --`,
    because its regex had been mangled by shell quoting.
27. **Docs**.

**History rewrite (not my work).** `sop-eval` has been rewritten twice, and
neither rewrite was mine.

1. Between round 3 and round 4, to purge the AttributionBench blob. The
   pre-rewrite commits are kept on `backup/pre-filter-researchreport`, which
   is the only place that blob's path still appears.
2. Before round 5, to remove the local account name from the history itself.
   The round-4 scrub (item 26) had only fixed the tree at its own tip, so the
   string survived in 19 earlier commits: 7 of them in a hardcoded absolute
   env-file path inside `eval_sop/common.py`, the rest inside the committed
   "before" evidence transcripts. The pre-rewrite commits are kept on
   `backup/pre-filter-researchreport2`.

Neither backup branch is ever pushed. Both rewrites preserved every subject
and the final tree, so only the hashes changed; every hash cited in this
document was re-derived against the current history by matching subjects
(1:1, no subject appears twice).

**Round-5 verification of the second rewrite.** Checked per commit, not just
at the tip, with the backup branch used as a positive control so the search
was proven able to find the strings on the pre-rewrite side before the
post-rewrite side was called clean. Searching case-insensitively for the
account name, for absolute-path prefixes under the Windows user directory and
for the cloud-sync folder name:

- `backup/pre-filter-researchreport2`: 19 of its 38 commits carry the account
  name in tracked content (the control fires).
- `sop-eval`: 0 of its 38 commits carry it, and none match the path-prefix or
  sync-folder patterns either.
- Commit messages, author and committer identity: the account name appears in
  none of the 38 on either side, so that part of the claim was already true
  before the rewrite rather than a result of it.
- The AttributionBench path appears in no tree of any of the 38 commits
  (`git ls-tree -r`); the remaining textual mentions are prose in
  `RESULTS.md`, `eval_sop/build_datasets.py`, `eval_sop/score.py` and
  `eval_sop/validate_judge.py`, which is intended.
- Final trees byte-identical: `git diff sop-eval backup/pre-filter-researchreport2`
  is empty.

Ancestry was re-checked with `git merge-base --is-ancestor`, not
`git cat-file -e`: the backup branches keep the superseded objects reachable,
so `cat-file` succeeds for a hash that is no longer on the branch and would
have hidden the stale citations. All hashes cited above are now ancestors of
`sop-eval`. The round-4 pass had instead cited the hashes produced by the
*first* rewrite, 29 of which the second rewrite had already superseded; all 29
are remapped here. The one comparison that intentionally spans a rewrite
boundary is the byte-identity check above, and it is now written as a
branch-to-branch diff rather than as a pair of hashes, so it cannot go stale
the next time history moves.

**Round-4 spend-safety review fixes.** The reviewer confirmed the ledger,
lock, atomic reservation, crash behaviour and retry accounting hold up, and
found four defects. Each was reproduced by a failing test first; before/after
output is in `eval_sop/evidence/`.

28. **A substituted model stopped being priced as the pinned one**
    (`2695bd8`).
    - Cost came from the REQUESTED id, so a response reporting
      `claude-opus-4-8` ($5/$25) was billed at Haiku's $1/$5 and the run
      continued; only the finished run was flagged. Sustained substitution
      could therefore authorise roughly $40 against an $8 ledger.
    - The transport now settles that call at the pinned worst case, stores the
      served id, and trips `BudgetStop`, so no further call is made.
    - The round-3 test only checked the post-run marker; it now asserts the
      stop. `r4_model_cap_*`.
29. **`nan` caps disabled the cap entirely** (`2695bd8`). `nan` fails every
    comparison, so both the hard-max guard and each per-call check were False;
    the reviewer settled 201 calls totalling $72.37 with no trip.
    `budget.check_cap` now requires a finite value in (0, $8] and backs the
    ledger, `cap_from_env` and argparse. `r4_model_cap_*`.
30. **The documented test command aborted at collection** (`a97b079`). An
    unconditional `import httpx2` in a test module broke
    `pytest eval_sop/tests -q` on an environment built from
    requirements.txt (anthropic 0.93 on httpx only), so the mid-stream,
    KeyboardInterrupt and concurrency guarantees went unexercised there and
    section 8 showed a red suite. The import is now optional and the tests
    are parametrised over the installed transports;
    `test_optional_imports.py` guards against a repeat. Production code was
    never affected. `r4_httpx2_import_*`.
31. **Unmetered spend outside the ledger** (`13f0cb2`).
    `scripts/record_demo_run.py` refuses without
    `--i-want-to-spend-real-money`, and the runbook says the paid key goes
    only in `--env-file`. `api/main.py` and `api/worker.py` still honour a
    server-side key and are not gated — they are servers, not scripts, so I
    only documented them. `r4_unledgered_*`.
32. **Exit-code collision** (`13f0cb2`): a corrupt ledger escaped as exit 1,
    which the runbook documents as "canary failed". Any failure to start is
    now 2.
33. **Docs**: the test counts in section 8 come from an actual run on this
    branch — 87 product, and 75 harness (77 where anthropic 1.x brings
    `httpx2`, which adds the two `[httpx2]` parametrisations of the
    mid-stream tests).
34. **Docs** (this commit), after re-verifying both on this machine:
    - the harness count, which had been reported from an environment that has
      `httpx2`, contradicting entry 30's reference environment;
    - the claim that the API server was ungated. It is gated by default:
      `ALLOW_SERVER_KEY_FALLBACK` is off unless set, and a keyless request is
      refused with HTTP 400.

**Round-5 merge-gate fixes.** Two blockers and five smaller corrections. The
code blocker was reproduced by a failing test first.

35. **The ledger and the run lock moved with `%LOCALAPPDATA%`** — the gate
    failure, and the one behavioural defect in this round.
    `budget.default_state_dir()` resolved its base from that environment
    variable, so a shell with it set elsewhere got an empty ledger (a fresh
    $8 of headroom, the canary's spend forgotten) and a relocated `run.lock`,
    which let two paid runs overlap. The directory is now
    `Path.home() / ".sop_eval" / "research_report"` on every platform, with no
    environment override, following the sibling CodePilot-SWE project's
    `_ledger_home()` (`codepilot/llm.py`) and keeping a project-specific
    subdirectory name.
    - Failing test first: three cases in `test_state_lock.py` —
      `test_state_dir_is_under_the_home_directory`,
      `test_state_dir_ignores_localappdata_in_a_fresh_process` (resolves the
      paths in a child process with both modules reloaded, which is what a new
      shell does) and `test_lock_sits_beside_the_ledger`. The first two fail on
      the pre-fix `default_state_dir` and pass after; verified both ways by
      reverting the one function. `r5_state_dir_*`. (The pre-existing
      `test_state_lives_outside_the_checkout_and_out_dir_cannot_move_it` also
      fails before, because it now asserts the new directory name.)
    - **Migration: nothing to migrate, and nothing was copied.** The old
      location was read before the change: `usd_ledger.sqlite` held **0 rows**
      in `calls` and **$0.0000** total, and `tavily_credits.sqlite` held
      `spent = 0`. Both files are left in place, untouched; no stale ledger was
      copied over the new one. The new directory starts empty, which is the
      same state. If a ledger with real rows is ever found at the old path, it
      must be moved deliberately and said so here — a silent copy would make
      the spend history unauditable.
    - Section 6 and section 8 previously presented the `%LOCALAPPDATA%` path
      as the guarantee; both are corrected.
36. **`eval_sop/stats.py`'s dependencies were undeclared.** It imports
    `numpy`, `scipy` and `scikit-learn`, none of which appeared in
    `requirements.txt`, `requirements-dev.txt` or the install line in section
    8, so a venv built exactly as documented aborted collection of
    `eval_sop/tests` with `ModuleNotFoundError: numpy` — zero tests run.
    `python-dotenv` was in the same state, carried only as a loose extra on
    the documented `pip install` line. All four are now declared in
    `requirements-dev.txt` with lower bounds, matching that file's existing
    style, and the install line no longer needs an extra. Verified by building
    a fresh venv in a short temp directory from the documented line alone:
    `eval_sop/tests` collected and **80 passed**, and `tests` 87 passed. That
    venv resolves `httpx2`, so 80 is the with-`httpx2` variant; without it the
    harness suite is 78.
37. **Every commit hash cited in this document was stale.** The round-4 pass
    had remapped them onto the *first* history rewrite; the second rewrite
    superseded 29 of those. Re-derived with
    `git merge-base --is-ancestor` — `git cat-file -e` is useless here, since
    the backup branches keep the superseded objects reachable and it succeeds
    for a hash that is no longer on the branch. See "History rewrite" above
    for the per-commit verification and the positive control.
38. **Item 26 overstated what it had done**, and the "History rewrite"
    paragraph described only the AttributionBench rewrite. Item 26's commit
    changed the working tree only; the account name survived in 19 earlier
    commits until the second rewrite. Both passages are corrected.
39. **`README.md` said the product suite is 83 tests.** It is 87 (70 in
    `tests/unit` + 17 in `tests/integration`, counted on this branch).
40. **The `$8` cap bounds spend, not completion.** Section 6 now states
    plainly that the documented `(a, b, c, d) × 30` design is $27.16 worst
    case (and 810 Tavily credits against 600), so it cannot be guaranteed to
    finish; expected cost is $5.71. An independent check corrected the exit
    code: without `--admission-control` the design is refused before anything
    is spent (**exit 2**), and only with it does the run stop partway (**exit
    4**). Both are safe; the original text named only the second.
41. **Smaller corrections.**
    - `budget.check_cap`'s docstring claimed it was used as argparse's
      `type`. It is not: the CLI uses plain `float`
      (`run_conditions.py:468`) and validates separately at
      `run_conditions.py:373`. Docstring corrected rather than rewiring the
      CLI, which would change the message and exit path the runbook tests
      assert.
    - `eval_sop/data/questions.jsonl` redistributes 824 FRAMES rows with
      per-row provenance and a paper citation but carried no licence notice.
      FRAMES is Apache-2.0; `eval_sop/data/NOTICE.md` now carries the notice,
      the disclaimer and the citation, and `build_datasets.py` points at it.
    - The ungated Celery task (`api/worker.py:65`) is now documented as an
      **accepted residual** in section 8, with the reason it is accepted and
      what a deployer should do instead. Entry 31 above predates entry 34's
      correction that `api/main.py` *is* gated by default; only the worker is
      not.

42. **The Tavily credit cap is now a project hard maximum too — a spend hole an
    independent check found after the round-5 state-dir work.** The USD cap was
    hardened at four layers; the credit cap had none. `TAVILY_CREDIT_CAP` was a
    bare `int(os.environ.get("SOP_TAVILY_CAP", "600"))`, so **`SOP_TAVILY_CAP`
    could raise it without limit** (measured: 999999 accepted, and
    `--tavily-cap 999999` printed `dry run OK`), and `--tavily-cap -5` was
    accepted unvalidated, which makes every headroom comparison meaningless.
    Credits are a paid resource on a key that may be shared with other
    projects, so this is a real exposure, not untidiness. `common.check_credit_cap`
    and `common.credit_cap_from_env` now mirror `budget.check_cap` /
    `cap_from_env` exactly, with `PROJECT_HARD_MAX_CREDITS = 600`; the env var
    and `--tavily-cap` may only **lower** it, and `run_conditions.py` validates
    the parsed value *before* `common.install`, so a bad cap is exit 2 with its
    reason. An empty `SOP_TAVILY_CAP` is refused rather than silently treated as
    600, matching the USD path. Failing tests first: 22 new cases in
    `test_hard_max.py` across `check_credit_cap`, `credit_cap_from_env` and a
    CLI run, covering 601, 999999, 0, -5, -1, nan, inf, the empty string, `abc`
    and `1e9`. `eval_sop/tests` 80 → **104** (24 new cases, not the 22 first
    claimed). Two of them, the CLI `0` and `-5` cases, originally asserted only
    `code == 2` and **passed on the unfixed code as well** -- the old path also
    returned 2, from the downstream headroom check: the right exit for the wrong
    reason. An independent check caught it. They now spy on `common.install` to
    assert it is never reached and match the refusal message, and all four CLI
    cases fail on the reverted code. `httpx2` is a hard transitive dependency of
    `anthropic`, `openai` and `langsmith`, so the two `[httpx2]` parametrisations
    always collect, so **104** is what the documented install produces. An
    intermediate claim of "102, or 104 with optional httpx2" had it backwards.
    A pre-existing environment without `httpx2` reports 102; that is the only
    way to see that number, and it is not the documented route.
43. **`default_state_dir`'s docstring was overstated.** It said "There is
    deliberately no environment-variable override." `Path.home()` on Windows
    reads `USERPROFILE`, so that one variable does move the path. Stated
    explicitly now rather than glossed, with the honest judgement that it is not
    the hole `LOCALAPPDATA` was: the OS sets `USERPROFILE` at logon and
    redirecting it breaks the whole session, whereas tools set `LOCALAPPDATA`
    per process routinely. Eight other variables were measured as leaving the
    path unmoved, individually and all at once. The substance of the round-5 fix
    holds; the absolute wording did not.
44. **README's install line was insufficient.** The Testing section documented
    `pip install -r requirements.txt`, which does not contain `pytest`, so the
    documented route could not run the documented tests. It now installs
    `requirements-dev.txt` as well and names the harness suite's count.

**Round-4 items I did not take.** Both are unreachable today and the reviewer
left them optional.
- `worst_call_cost` ignores `cache_write` pricing. Nothing in the eval sets
  `cache_control`, so no call can write a cache entry; if one ever does, the
  worst case would be understated by the $1.25/MTok write premium.
- `worst_input_tokens` has no per-block floor for non-text content, so a
  200 KB base64 image block estimates about 50 tokens. The pipeline sends
  text only.

**How strong the "fails before the fix" evidence is.**
- Behavioural reproductions, where the old code ran and gave the wrong
  result: `test_graph_e2e.py`, `test_llm_client.py`'s price test,
  `test_repro_review.py` (cap swallowed, failed run saved as success),
  `test_review2_ledger.py`, `test_token_estimate.py`,
  `test_tavily_credits.py`, `test_scrape_cache.py`,
  `test_scoring_policy.py` and `test_nits.py`.
- Not behavioural: most tests in `test_anthropic_budget.py` (and the
  `ledger_path` / `CANARY_RUNS` checks) would fail on pre-fix commits only by
  ImportError or AttributeError, because the code they exercise did not exist
  yet. They show the new code works; they do not show an old behaviour was
  wrong.

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

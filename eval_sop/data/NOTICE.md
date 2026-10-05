# Third-party data redistributed in this directory

The repository itself is MIT (see `LICENSE` at the root). The data files here
are *not* original to this repository and carry their own terms.

## `questions.jsonl` — FRAMES

824 of the 829 rows are derived from the **FRAMES** benchmark
(`google/frames-benchmark`, file `test.tsv`), redistributed here with the
upstream question text, the upstream human-written answer as `reference`, the
upstream `reasoning_types`, and per-row provenance (`source`, `source_row`,
`source_sha256` of the upstream file). Only the row order is ours
(`random.Random(20261001)`); see `eval_sop/build_datasets.py`.

FRAMES is distributed by Google under the **Apache License, Version 2.0**.
A copy of that licence is available at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software and data
distributed under the Apache License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

Paper:

> Krishna, S., Krishna, K., Mohananey, A., Schwarcz, S., Stambler, A.,
> Upadhyay, S., & Faruqui, M. (2024). *Fact, Fetch, and Reason: A Unified
> Evaluation of Retrieval-Augmented Generation.* arXiv:2409.12941.

The remaining 5 rows are open-ended questions written for this evaluation and
have no reference answer; they are covered by the repository's own MIT licence.

# TODO — picking this back up

The core product is **built and pushed to `main`**. What's left is deployment,
CI wiring, and some optional polish/hardening. Start at the top.

---

## Already done (so you don't re-do it)

- Multi-provider **BYOK** LLM layer (Anthropic native + OpenAI/Groq/Gemini via one
  OpenAI-compatible path); key never persisted; live model-list endpoint.
- **Next.js frontend** (`frontend/`) — provider/model dropdowns, BYOK key, live
  pipeline view, quality meters, Markdown report. Builds clean, design verified.
- **Cloud Run-ready backend** — inline job mode (no Redis/Celery), `$PORT`-aware
  image, tracing no-ops without a collector.
- Security: CORS allowlist, no exception-detail leaks.
- Unit tests (21 passing), README, `DEPLOY.md`.

---

## 1. Do first — go live (needs your secrets)

- [ ] Create a free **Neon** Postgres → get `DATABASE_URL`.
- [ ] Get a **Tavily** API key → `TAVILY_API_KEY`.
- [ ] Deploy backend to Cloud Run (APIs already enabled on `ml-feature-store-sb`).
      Command is in [`DEPLOY.md`](./DEPLOY.md) §1. Copy the printed Service URL.
- [ ] Deploy frontend: `cd frontend && vercel link`, set
      `NEXT_PUBLIC_API_URL` = Cloud Run URL, `vercel --prod`. (`DEPLOY.md` §2.)
- [ ] Tighten CORS: `gcloud run services update research-agent-api
      --region us-central1 --update-env-vars ALLOWED_ORIGINS=https://YOUR-APP.vercel.app`.
- [ ] **Verify end-to-end** — this has never run the full pipeline live. Open the
      Vercel URL, pick a provider, paste a key, click Load, run a query, confirm
      the pipeline, quality scores, and report all render.
- [ ] Put the live URL + a screenshot in `README.md` (the "Live demo" line).

## 2. Should do

- [ ] **Wire up CI.** `.github/workflows/ci.yml` exists but is untracked and won't
      push without a `workflow`-scoped token (that's why it was removed before).
      Commit it (add a frontend `npm ci && build` job), and make the integration
      job not depend on real LLM keys. Push via a token with `workflow` scope or
      add the file through the GitHub web UI.
- [ ] **Remove the legacy Gradio UI** now that Next.js is the frontend: delete
      `ui/app.py`, the `gradio` dep in `requirements.txt`, and the `gradio` stage in
      the Dockerfile + `docker-compose.yml`. (Its report formatting now lives in
      `agents/report_format.py`.)
- [ ] **OpenAI new-model compatibility.** `call_llm` sends `max_tokens`; newer
      OpenAI reasoning models (o-series / gpt-5) expect `max_completion_tokens` and
      reject `max_tokens`. Add a per-model param mapping in the OpenAI branch of
      `agents/llm_client.py` if you want those models to work.
- [ ] **Secret Manager** for `DATABASE_URL` / `TAVILY_API_KEY` on Cloud Run
      (`--set-secrets`) instead of plaintext env vars.

## 3. Nice to have / roadmap

- [ ] **Cloud Tasks worker** for durable scale-to-zero — inline mode loses a job if
      the instance is reclaimed mid-run. Add an internal `/run` endpoint + Cloud
      Tasks enqueue as a third `JOB_BACKEND`.
- [ ] **Compute-abuse guard** on the public API — the per-IP rate limit (10/min)
      exists; consider an optional shared-secret header for `/report/generate` so
      randoms can't burn your Cloud Run minutes (BYOK already removes the LLM-cost
      risk).
- [ ] More tests: graph transitions / critic-loop logic, mock-LLM agent tests;
      make `tests/integration` cheap (no real API calls).
- [ ] Minor: `ResearchState` uses the deprecated `class Config` — switch to
      `model_config = ConfigDict(arbitrary_types_allowed=True)`.
- [ ] Frontend polish: verify mobile layout visually; optional light theme.

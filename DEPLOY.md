# Deployment

Two tiers: the FastAPI backend on **Google Cloud Run** (scale-to-zero) and the
Next.js frontend on **Vercel**. Deploy the backend first — the frontend needs its
URL.

## Prerequisites

- `gcloud` authenticated (`gcloud auth login`) with a project set.
- `vercel` CLI authenticated (`vercel login`).
- *Optional:* a managed Postgres connection string — **`DATABASE_URL`** (e.g. a
  free [Neon](https://neon.tech) project), format `postgresql://user:pass@host/db`.
  Leave it unset and the service uses its in-process job store instead: fine for a
  single scale-to-zero instance, but jobs are lost when the instance is reclaimed.
  `/health` reports which store is live.
- A **`TAVILY_API_KEY`** (web search; [tavily.com](https://tavily.com)).

> LLM keys are **not** needed at deploy time — they're BYOK, entered per request
> in the UI. Set `ANTHROPIC_API_KEY` only if you want a server-side fallback.

---

## 1. Backend → Cloud Run

```bash
# from the repo root
PROJECT=ml-feature-store-sb
REGION=us-central1

gcloud services enable run.googleapis.com cloudbuild.googleapis.com --project "$PROJECT"

gcloud run deploy research-agent-api \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 \
  --timeout 3600 \
  --set-env-vars "JOB_BACKEND=inline" \
  --set-env-vars "DATABASE_URL=YOUR_NEON_URL" \
  --set-env-vars "TAVILY_API_KEY=YOUR_TAVILY_KEY"
```

Notes:
- `--source .` builds the Dockerfile's final stage (the `api` stage) with Cloud
  Build and deploys it. 2 GiB memory covers headless Chromium; `--timeout 3600`
  (the 60-min max) lets long inline jobs finish.
- The command prints a **Service URL** — copy it for step 2.
- After you know the Vercel URL, tighten CORS without re-passing secrets
  (`--update-env-vars` merges; it does not replace all env vars):
  `gcloud run services update research-agent-api --region "$REGION" \
    --update-env-vars "ALLOWED_ORIGINS=https://YOUR-APP.vercel.app"`
- **Hardening:** put `DATABASE_URL` / `TAVILY_API_KEY` in Secret Manager and
  reference them with `--set-secrets` instead of `--set-env-vars`.

Verify:
```bash
curl https://research-agent-api-XXXX.run.app/health
```

---

## 2. Frontend → Vercel

```bash
cd frontend
vercel link                       # link this dir to a Vercel project on your git repo
vercel env add NEXT_PUBLIC_API_URL production   # paste the Cloud Run Service URL
vercel --prod                     # first production deploy
```

Because the project is linked to the GitHub repo, subsequent `git push` to
`main` auto-deploys. In the Vercel project settings, set the **Root Directory**
to `frontend` if it wasn't detected.

---

## 3. Close the loop

Update the backend's `ALLOWED_ORIGINS` to the Vercel production URL (see step 1),
then open the Vercel URL, pick a provider, paste your key, click **Load**, and run
a query.

"""
Print the Tavily key's credit usage (free endpoint, spends no credits).

  python -m eval_sop.tavily_usage --env-file PATH

The key is read in-process and never printed. The eval's own ledger only
knows what this eval spent; this shows usage by every project on the key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
from dotenv import dotenv_values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", type=Path, required=True)
    key = dotenv_values(ap.parse_args().env_file).get("TAVILY_API_KEY")
    if not key:
        raise SystemExit("no TAVILY_API_KEY in that file")
    r = httpx.get("https://api.tavily.com/usage", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    acct = r.json().get("account", {})
    out = {k: acct.get(k) for k in ("current_plan", "plan_usage", "plan_limit", "paygo_usage", "paygo_limit")}
    if out["plan_limit"] is not None and out["plan_usage"] is not None:
        out["headroom"] = out["plan_limit"] - out["plan_usage"]
    print(json.dumps(out))


if __name__ == "__main__":
    main()

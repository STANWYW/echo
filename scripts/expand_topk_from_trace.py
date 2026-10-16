#!/usr/bin/env python3
"""Expand per-step trace candidates from an ECHO attack JSONL.

The main attack log keeps the final best response plus, for each trace step, the
highest proxy-scored candidates evaluated at that step. This script deduplicates
those candidates per goal, sorts them by HarmSig proxy score, and emits a flat
JSONL suitable for StrongREJECT evaluation and top-m curve computation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_goal_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("type") != "statistics":
                yield obj


def collect_candidates(row: dict[str, Any], max_candidates: int | None) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    cands: list[dict[str, Any]] = []

    def add(prompt: str, response: str, proxy: float, original: float, source: str) -> None:
        if not prompt or not response:
            return
        key = (prompt, response)
        if key in seen:
            return
        seen.add(key)
        cands.append({
            "goal_index": int(row.get("goal_index", -1)),
            "goal": row.get("goal", ""),
            "candidate_proxy_score": float(proxy),
            "candidate_original_score": float(original),
            "best_prompt": prompt,
            "best_response": response,
            "source": source,
        })

    add(
        str(row.get("best_prompt", "")),
        str(row.get("best_response", "")),
        float(row.get("best_score", 0.0)),
        float(row.get("best_original_score", row.get("best_score", 0.0))),
        "final_best",
    )

    for step in row.get("optimization_trace", []) or []:
        if not isinstance(step, dict):
            continue
        for cand in step.get("top_candidates", []) or []:
            if not isinstance(cand, dict):
                continue
            add(
                str(cand.get("prompt", "")),
                str(cand.get("response", "")),
                float(cand.get("score", 0.0)),
                float(cand.get("original_score", cand.get("score", 0.0))),
                f"trace_step_{step.get('step')}",
            )

    cands.sort(key=lambda x: x["candidate_proxy_score"], reverse=True)
    for i, cand in enumerate(cands):
        cand["candidate_rank"] = i
    if max_candidates is not None:
        return cands[:max_candidates]
    return cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("attack_jsonl", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-candidates", type=int, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_goals = 0
    n_rows = 0
    with args.out.open("w", encoding="utf-8") as wf:
        for row in iter_goal_rows(args.attack_jsonl):
            n_goals += 1
            for cand in collect_candidates(row, args.max_candidates):
                wf.write(json.dumps(cand, ensure_ascii=False) + "\n")
                n_rows += 1
    print(f"expanded {n_rows} candidates from {n_goals} goals -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

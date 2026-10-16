#!/usr/bin/env python3
"""Compute mean StrongREJECT@k from an expanded candidate JSONL."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("expanded_sr_jsonl", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-k", type=int, default=25)
    args = ap.parse_args()

    by_goal: dict[int, list[dict]] = defaultdict(list)
    with args.expanded_sr_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "statistics":
                continue
            if obj.get("strongreject_finetuned_score") is None:
                raise ValueError("missing strongreject_finetuned_score; run eval_main_results_strongreject.py first")
            by_goal[int(obj["goal_index"])].append(obj)

    rows = []
    for k in range(1, args.max_k + 1):
        vals = []
        for cands in by_goal.values():
            ordered = sorted(cands, key=lambda x: int(x.get("candidate_rank", 10**9)))[:k]
            if ordered:
                vals.append(max(float(x["strongreject_finetuned_score"]) for x in ordered))
        rows.append({"k": k, "mean_sr": statistics.mean(vals) if vals else 0.0})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "mean_sr"])
        w.writeheader()
        for row in rows:
            w.writerow({"k": row["k"], "mean_sr": f"{row['mean_sr']:.8f}"})
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

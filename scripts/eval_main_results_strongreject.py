#!/usr/bin/env python3
"""Score ECHO attack logs with the finetuned StrongREJECT evaluator.

Each per-goal JSONL row with ``goal`` and ``best_response`` receives a
``strongreject_finetuned_score`` field in the output file. Statistics rows are
copied unchanged.

The StrongREJECT finetuned judge downloads ``qylu4156/strongreject-15k-v1`` and
its gated base model ``google/gemma-2b`` unless ``--local-peft-cache`` is used.
Accept the Gemma license on Hugging Face and provide credentials via
``HF_TOKEN`` or ``huggingface-cli login``. Never commit tokens to this repository.

Examples::

    export HF_TOKEN=...
    python scripts/eval_main_results_strongreject.py --gpu 0 --files attack.jsonl

    python scripts/eval_main_results_strongreject.py \
        --hf-token-file ~/.config/hf_token --files attack.jsonl

    python scripts/eval_main_results_strongreject.py --local-peft-cache --files attack.jsonl
"""

from __future__ import annotations

import os
import sys


def _apply_cuda_visible_devices_from_argv() -> None:
    """Parse --gpu before the first torch import so CUDA_VISIBLE_DEVICES takes effect."""
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == "--gpu":
            if i + 1 < len(argv) and not str(argv[i + 1]).startswith("-"):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[i + 1]).strip()
            return
        if tok.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = tok.split("=", 1)[1].strip()
            return


_apply_cuda_visible_devices_from_argv()

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from huggingface_hub import get_token


def _is_goal_row(obj: dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("type") == "statistics":
        return False
    if "goal" not in obj or "best_response" not in obj:
        return False
    if not isinstance(obj.get("best_response"), str) or not obj["best_response"].strip():
        return False
    return True


def score_file(path: Path, batch_size: int, empty_model_cache: bool) -> list[dict[str, Any]]:
    """Return list of dicts (same order as lines) with SR scores added where applicable."""
    raw_lines: list[dict[str, Any]] = []
    indices: list[int] = []
    goals: list[str] = []
    responses: list[str] = []

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {i + 1}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {i + 1}: expected object, got {type(obj)}")
            raw_lines.append(obj)
            if _is_goal_row(obj):
                indices.append(len(raw_lines) - 1)
                goals.append(str(obj["goal"]))
                responses.append(str(obj["best_response"]))

    if not goals:
        return raw_lines

    from strong_reject.evaluate import evaluate_dataset

    ds = Dataset.from_dict({"forbidden_prompt": goals, "response": responses})
    ev = evaluate_dataset(
        ds,
        ["strongreject_finetuned"],
        batch_size=batch_size,
        empty_model_cache=empty_model_cache,
    )
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            print(
                f"[strongreject-eval] CUDA ok: device_count={torch.cuda.device_count()}, "
                f"current={idx}, name={torch.cuda.get_device_name(idx)}",
                flush=True,
            )
        else:
            print(
                "[strongreject-eval] WARNING: torch.cuda.is_available() is False; "
                "scoring will run on CPU. Install a CUDA-enabled PyTorch build for GPU scoring.",
                flush=True,
            )
    except Exception as ex:
        print(f"[strongreject-eval] (CUDA self-check skipped: {ex})", flush=True)

    scores = ev["score"]
    if len(scores) != len(indices):
        raise RuntimeError(
            f"{path}: score count {len(scores)} != goal rows {len(indices)}"
        )

    out = [dict(r) for r in raw_lines]
    for j, idx in enumerate(indices):
        out[idx]["strongreject_finetuned_score"] = float(scores[j])
    return out


def _hub_snapshot_dir(repo_id: str) -> Path:
    """Resolve newest local snapshot for a Hub repo id (offline StrongREJECT)."""
    slug = "models--" + repo_id.replace("/", "--")
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / slug / "snapshots"
    if not root.is_dir():
        raise FileNotFoundError(f"no HF cache for {repo_id}: {root}")
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    if not snaps:
        raise FileNotFoundError(f"empty HF cache for {repo_id}: {root}")
    return snaps[-1]


def seed_local_strongreject_cache(
    *,
    adapter_repo: str = "qylu4156/strongreject-15k-v1",
    base_repo: str = "google/gemma-2b",
    adapter_path: Path | None = None,
    base_path: Path | None = None,
) -> None:
    """Preload finetuned judge into strong_reject.cached_models (no Hub round-trip)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from strong_reject.evaluate import cached_models

    if "strongreject_finetuned" in cached_models:
        return

    adapter = adapter_path or _hub_snapshot_dir(adapter_repo)
    base = base_path or _hub_snapshot_dir(base_repo)
    base_m = AutoModelForCausalLM.from_pretrained(
        str(base), local_files_only=True, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_m, str(adapter), local_files_only=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter), local_files_only=True, padding_side="left", truncation_side="left"
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    cached_models["strongreject_finetuned"] = model, tokenizer
    print(
        f"[eval_main_results_strongreject] seeded local cache: base={base} adapter={adapter}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="StrongREJECT finetuned scores for main_results JSONL logs.")
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        metavar="ID",
        help="Physical GPU id; parsed into CUDA_VISIBLE_DEVICES before torch import",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "main_results",
        help="Directory containing *.jsonl attack logs",
    )
    p.add_argument("--glob", default="*.jsonl", help="Glob under results-dir (default: *.jsonl)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write scored JSONL here (default: same as input, name *_sr.jsonl)",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for finetuned evaluator")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Require CUDA by default; pass this flag to allow CPU scoring",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Write to <stem>_sr.jsonl even if it exists (default: skip existing)",
    )
    p.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help="Read one token line from this file into HF_TOKEN; do not commit this file",
    )
    p.add_argument(
        "--local-peft-cache",
        action="store_true",
        help="Load the StrongREJECT LoRA only from the local Hugging Face cache",
    )
    p.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=None,
        metavar="PATH",
        help="Explicit .jsonl files; can be combined with positional paths",
    )
    p.add_argument(
        "jsonl_paths",
        nargs="*",
        type=Path,
        help="Optional positional .jsonl paths; equivalent to --files and can be combined",
    )
    args = p.parse_args()

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"[eval_main_results_strongreject] CUDA_VISIBLE_DEVICES={vis!r} "
        f"(if --gpu is used, it was parsed before importing datasets)",
        flush=True,
    )
    if args.gpu is not None and vis != str(args.gpu).strip():
        print(
            f"WARNING: --gpu={args.gpu!r} does not match current CUDA_VISIBLE_DEVICES={vis!r}.",
            "start the command with ``--gpu N`` instead of relying on later argument passing.",
            file=sys.stderr,
            flush=True,
        )

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        print(
            "[eval_main_results_strongreject] "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"cuda_available={cuda_ok}, device_count={torch.cuda.device_count()}",
            flush=True,
        )
        if cuda_ok:
            idx = torch.cuda.current_device()
            print(
                f"[eval_main_results_strongreject] using cuda:{idx} "
                f"({torch.cuda.get_device_name(idx)})",
                flush=True,
            )
        elif not args.allow_cpu:
            print(
                "ERROR: CUDA is unavailable and CPU scoring is disabled. "
                "Install a CUDA-enabled PyTorch build, or pass --allow-cpu intentionally.",
                file=sys.stderr,
                flush=True,
            )
            return 1
    except Exception as ex:
        if not args.allow_cpu:
            print(f"ERROR: CUDA self-check failed and CPU scoring is disabled: {ex}", file=sys.stderr, flush=True)
            return 1
        print(f"[eval_main_results_strongreject] CUDA self-check failed; continuing on CPU: {ex}", flush=True)

    if args.hf_token_file is not None:
        pth = args.hf_token_file.expanduser().resolve()
        raw = pth.read_text(encoding="utf-8").strip().splitlines()
        if not raw or not raw[0].strip():
            print(f"ERROR: {pth} is empty or does not contain a readable token.", file=sys.stderr)
            return 1
        os.environ["HF_TOKEN"] = raw[0].strip()

    if args.local_peft_cache:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    elif not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()):
        print(
            "ERROR: No Hugging Face credentials detected. Set HF_TOKEN=... "
            "or run huggingface-cli login. The gated model google/gemma-2b "
            "requires an account with accepted license terms.",
            file=sys.stderr,
        )
        return 1

    explicit: list[Path] = []
    if args.files:
        explicit.extend(args.files)
    if args.jsonl_paths:
        explicit.extend(args.jsonl_paths)
    if explicit:
        paths = [Path(f).resolve() for f in explicit]
    else:
        results_dir = args.results_dir.resolve()
        if not results_dir.is_dir():
            print(f"Not a directory: {results_dir}", file=sys.stderr)
            return 1
        paths = sorted(results_dir.glob(args.glob))

    if not paths:
        print("No JSONL files to process.", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve() if args.output_dir else None

    work: list[tuple[Path, Path]] = []
    for path in paths:
        if not path.is_file():
            print(f"Skip (not a file): {path}", file=sys.stderr)
            continue
        if path.name.endswith("_sr.jsonl"):
            print(f"Skip (already scored output): {path}", file=sys.stderr)
            continue

        dest = out_dir / f"{path.stem}_sr.jsonl" if out_dir else path.parent / f"{path.stem}_sr.jsonl"
        if dest.exists() and not args.overwrite:
            print(f"Skip (exists): {dest}")
            continue
        work.append((path, dest))

    if not work:
        print("Nothing to do (all skipped or no matches).")
        return 0

    if args.local_peft_cache:
        seed_local_strongreject_cache()

    for k, (path, dest) in enumerate(work):
        last = k == len(work) - 1
        try:
            rows = score_file(path, batch_size=args.batch_size, empty_model_cache=last)
        except Exception as e:
            print(f"Failed {path}: {e}", file=sys.stderr)
            return 1

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as wf:
            for row in rows:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_scored = sum(1 for r in rows if "strongreject_finetuned_score" in r)
        print(f"Wrote {dest} ({n_scored} rows scored, {len(rows)} lines total)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

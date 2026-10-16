#!/usr/bin/env python3
"""Shared semantic-anchor beam support used by the ECHO V2 runner."""

from __future__ import annotations

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nltk.corpus import wordnet
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt_attack_utils import check_prompt_keyword_validity  # noqa: E402
from unified_attack import BeamSearchOptimizer, check_asr_success  # noqa: E402


STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "be", "that", "this", "these",
}

SOFTENERS = [
    "please", "clearly", "specifically", "concretely", "directly", "carefully",
    "precisely", "systematically",
]

DETAIL_MARKERS = [
    "detailed", "practical", "complete", "step-by-step", "specific", "actionable",
]

STRUCTURE_MARKERS = [
    "outline", "explain", "describe", "provide", "include",
]


def _tokens(text: str) -> list[str]:
    return text.split()


def _norm_token(token: str) -> str:
    return re.sub(r"(^[^A-Za-z0-9]+|[^A-Za-z0-9]+$)", "", token).lower()


def _protected_set(keyword_map: dict[str, set[str]] | None) -> set[str]:
    protected: set[str] = set()
    if not keyword_map:
        return protected
    for key, syns in keyword_map.items():
        protected.add(_norm_token(str(key)))
        for syn in syns:
            for part in str(syn).split():
                protected.add(_norm_token(part))
    return {p for p in protected if p}


def _valid_synonym_candidate(word: str, cand: str) -> bool:
    cand = cand.replace("_", " ").strip()
    if " " in cand:
        return False
    if not cand.isascii() or not cand.isalpha():
        return False
    if len(cand) < 3 or len(cand) > 14:
        return False
    if cand.lower() == word.lower():
        return False
    return True


def semantic_synonym(prompt: str, protected: set[str]) -> tuple[str, dict[str, Any]]:
    words = _tokens(prompt)
    candidates = [
        i for i, w in enumerate(words)
        if _norm_token(w) not in protected
        and _norm_token(w) not in STOP_WORDS
        and len(_norm_token(w)) > 3
    ]
    random.shuffle(candidates)
    for idx in candidates[:8]:
        word = _norm_token(words[idx])
        syns: list[str] = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                cand = lemma.name().replace("_", " ")
                if _valid_synonym_candidate(word, cand):
                    syns.append(cand)
        if not syns:
            continue
        # Prefer shorter, common-looking alternatives over rare WordNet artifacts.
        syns = sorted(set(syns), key=lambda x: (abs(len(x) - len(word)), len(x)))
        replacement = random.choice(syns[: min(5, len(syns))])
        old = words[idx]
        words[idx] = replacement
        return " ".join(words), {"op": "semantic_synonym", "index": idx, "from": old, "to": replacement}
    return prompt, {"op": "semantic_synonym", "noop": True}


def functional_insert(prompt: str, protected: set[str]) -> tuple[str, dict[str, Any]]:
    words = _tokens(prompt)
    family = random.choices(
        ["detail", "softener", "structure"],
        weights=[0.50, 0.30, 0.20],
        k=1,
    )[0]
    if family == "detail":
        token = random.choice(DETAIL_MARKERS)
    elif family == "softener":
        token = random.choice(SOFTENERS)
    else:
        token = random.choice(STRUCTURE_MARKERS)

    # Put structure words near the front; detail/softener words near an existing
    # non-protected content word so they modify rather than replace the goal.
    if family == "structure":
        idx = 0
    else:
        content_positions = [
            i for i, w in enumerate(words)
            if _norm_token(w) not in protected and _norm_token(w) not in STOP_WORDS
        ]
        idx = random.choice(content_positions) if content_positions else random.randint(0, len(words))
    words.insert(idx, token)
    return " ".join(words), {"op": "functional_insert", "family": family, "token": token, "index": idx}


def conservative_delete(prompt: str, protected: set[str]) -> tuple[str, dict[str, Any]]:
    words = _tokens(prompt)
    if len(words) <= 6:
        return prompt, {"op": "conservative_delete", "noop": True}
    removable = [
        i for i, w in enumerate(words)
        if _norm_token(w) in set(SOFTENERS + DETAIL_MARKERS)
        and _norm_token(w) not in protected
    ]
    if not removable:
        # Remove duplicate non-protected tokens only.
        seen = set()
        for i, w in enumerate(words):
            nw = _norm_token(w)
            if nw in seen and nw not in protected and nw not in STOP_WORDS:
                removable.append(i)
            seen.add(nw)
    if not removable:
        return prompt, {"op": "conservative_delete", "noop": True}
    idx = random.choice(removable)
    removed = words.pop(idx)
    return " ".join(words), {"op": "conservative_delete", "index": idx, "removed": removed}


def light_swap(prompt: str, protected: set[str]) -> tuple[str, dict[str, Any]]:
    words = _tokens(prompt)
    pairs = [
        i for i in range(len(words) - 1)
        if _norm_token(words[i]) not in protected
        and _norm_token(words[i + 1]) not in protected
    ]
    if not pairs:
        return prompt, {"op": "light_swap", "noop": True}
    idx = random.choice(pairs)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words), {"op": "light_swap", "index": idx}


class SemanticAnchorBeamOptimizer(BeamSearchOptimizer):
    """Beam search with semantic-anchor mutation operators and full trace."""

    def __init__(
        self,
        *args,
        top_k_trace: int = 5,
        store_optimization_trace: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.top_k_trace = top_k_trace
        self.store_optimization_trace = store_optimization_trace
        self.operator_weights = {
            "semantic_synonym": 0.42,
            "functional_insert": 0.36,
            "conservative_delete": 0.17,
            "light_swap": 0.05,
        }
        self.operator_stats = Counter()
        self.operator_noops = Counter()
        self.operator_rejects = Counter()

    def reset_stats(self):
        super().reset_stats()
        self.operator_stats = Counter()
        self.operator_noops = Counter()
        self.operator_rejects = Counter()

    def _semantic_mutate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        protected = _protected_set(self._keyword_to_synonyms_map)
        ops = {
            "semantic_synonym": semantic_synonym,
            "functional_insert": functional_insert,
            "conservative_delete": conservative_delete,
            "light_swap": light_swap,
        }
        names = list(self.operator_weights)
        weights = [self.operator_weights[n] for n in names]
        name = random.choices(names, weights=weights, k=1)[0]
        mutated, meta = ops[name](prompt, protected)
        self.operator_stats[name] += 1
        if meta.get("noop") or mutated == prompt:
            self.operator_noops[name] += 1
        return mutated, meta

    def _mutate_with_keyword_check(self, prompt: str) -> str:
        if not self.use_keywords:
            mutated, _ = self._semantic_mutate(prompt)
            return mutated
        for _ in range(self.max_mutation_retries):
            mutated, meta = self._semantic_mutate(prompt)
            if check_prompt_keyword_validity(
                mutated, self._keyword_to_synonyms_map, self.min_prompt_keywords
            ):
                return mutated
            self.stats["mutation_retries"] += 1
            self.operator_rejects[meta.get("op", "unknown")] += 1
        return prompt

    def expand_beam_with_trace(self, beam: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
        candidates: list[str] = []
        traces: list[dict[str, Any]] = []
        for parent_rank, prompt in enumerate(beam):
            for _ in range(self.branch_factor):
                before = prompt
                after = self._mutate_with_keyword_check(before)
                candidates.append(after)
                traces.append({
                    "parent_rank": parent_rank,
                    "parent_prompt": before,
                    "candidate_prompt": after,
                })
        return candidates, traces

    def _top_items(self, evaluated, trace_by_prompt: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """If ``top_k_trace <= 0``, keep *all* evaluated rows but omit response text (compact)."""
        out = []
        if self.top_k_trace <= 0:
            slice_eval = evaluated
            include_response = False
        else:
            slice_eval = evaluated[: min(self.top_k_trace, len(evaluated))]
            include_response = True
        for prompt, score, response, original_score, keyword_count in slice_eval:
            traces = trace_by_prompt.get(prompt, [])
            entry: dict[str, Any] = {
                "prompt": prompt,
                "score": float(score),
                "original_score": float(original_score),
                "keyword_count": int(keyword_count),
                "candidate_trace": traces[:3],
            }
            if include_response:
                entry["response"] = response
            out.append(entry)
        return out

    def optimize(self, goal: str, max_steps: int = 20, use_template: bool = False, **kwargs) -> dict:
        print(f"\n{'=' * 80}\nGoal: {goal}\nAlgorithm: Semantic Anchor Beam\n{'=' * 80}")
        initial_beam = self.initialize_beam(goal, use_template=False)
        evaluated_beam = self._batch_evaluate(goal, initial_beam)
        evaluated_beam.sort(key=lambda x: x[1], reverse=True)

        current_beam = [prompt for prompt, *_ in evaluated_beam[: self.beam_width]]
        best_prompt, best_score, best_response, best_orig, best_kw = evaluated_beam[0]
        score_history = [best_score]
        trace: list[dict[str, Any]] = []
        if self.store_optimization_trace:
            trace.append({
                "step": -1,
                "event": "initial",
                "global_best_score": float(best_score),
                "global_best_original_score": float(best_orig),
                "global_best_keyword_count": int(best_kw),
                "global_best_prompt": best_prompt,
                "global_best_response": best_response,
                "top_candidates": self._top_items(evaluated_beam, {}),
            })

        for step in tqdm(range(max_steps), desc="Semantic Beam"):
            candidates, cand_traces = self.expand_beam_with_trace(current_beam)
            trace_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for tr in cand_traces:
                trace_by_prompt[tr["candidate_prompt"]].append(tr)
            evaluated = self._batch_evaluate(goal, candidates)
            evaluated.sort(key=lambda x: x[1], reverse=True)
            current_beam = [prompt for prompt, *_ in evaluated[: self.beam_width]]

            cur_prompt, cur_score, cur_response, cur_orig, cur_kw = evaluated[0]
            improved = False
            if cur_score > best_score:
                best_prompt, best_score, best_response, best_orig, best_kw = (
                    cur_prompt, cur_score, cur_response, cur_orig, cur_kw
                )
                self.stats["improvements"] += 1
                improved = True
                print(f"  step {step}: new best {best_score:.3f} (orig={cur_orig:.3f}, kw={cur_kw})")

            score_history.append(best_score)
            self.stats["total_steps"] += 1
            if self.store_optimization_trace:
                trace.append({
                    "step": step,
                    "event": "step",
                    "improved": improved,
                    "global_best_score": float(best_score),
                    "global_best_original_score": float(best_orig),
                    "global_best_keyword_count": int(best_kw),
                    "global_best_prompt": best_prompt,
                    "global_best_response": best_response,
                    "step_best_score": float(cur_score),
                    "step_best_original_score": float(cur_orig),
                    "step_best_keyword_count": int(cur_kw),
                    "step_best_prompt": cur_prompt,
                    "step_best_response": cur_response,
                    "top_candidates": self._top_items(evaluated, trace_by_prompt),
                    "operator_stats": dict(self.operator_stats),
                    "operator_noops": dict(self.operator_noops),
                    "operator_rejects": dict(self.operator_rejects),
                })
            if best_score >= self.early_stop_score:
                print(f"  step {step}: early stop at {best_score:.3f}")
                break

        asr = check_asr_success(best_response)
        result = self._build_result(
            goal, False, initial_beam[0], best_prompt, best_response,
            score_history[0], best_score, best_orig, best_kw, score_history, asr,
        )
        trace_ver = (
            "semantic_anchor_beam_topk_full_response_v1"
            if self.store_optimization_trace and self.top_k_trace > 0
            else (
                "semantic_anchor_beam_compact_trace_no_response_v1"
                if self.store_optimization_trace
                else "semantic_anchor_beam_notrace_v1"
            )
        )
        result.update({
            "algorithm": "semantic_anchor_beam",
            "operator_design": {
                "semantic_synonym": "protected-keyword WordNet substitution with artifact filtering",
                "functional_insert": "typed insertion: detail, softener, structure",
                "conservative_delete": "delete only inserted softeners/detail markers or duplicates",
                "light_swap": "downweighted adjacent non-core swap",
                "weights": self.operator_weights,
            },
            "optimization_trace": trace,
            "operator_stats": dict(self.operator_stats),
            "operator_noops": dict(self.operator_noops),
            "operator_rejects": dict(self.operator_rejects),
            "trace_version": trace_ver,
        })
        return result

"""Inference wrapper for the fine-tuned harm-score evaluator.

HarmSig inference wrapper used by the ECHO reproduction. The fine-tune architecture
(see ``evaluator/finetune/train.py``):

  backbone (Qwen3-1.7B + LoRA) -> EOS hidden state -> MLP head -> sigmoid -> [0, 1]

The trained model scores *responses only* (no goal conditioning), which is how
``prepare_data.py`` built the dataset. The public API below still accepts a
``goal`` argument for drop-in compatibility but ignores it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BACKBONE_PATH = None
DEFAULT_CKPT_PATH = None
DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_MAX_LEN = 1024

LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


class _HarmScoreModel(nn.Module):
    """Same architecture as training-time HarmScoreModel."""

    def __init__(self, backbone: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1]
        seq_lens = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[
            torch.arange(last_hidden.size(0), device=last_hidden.device),
            seq_lens,
        ]
        logit = self.score_head(pooled.float()).squeeze(-1)
        return torch.sigmoid(logit)


class HarmScoreEvaluator:
    """Batch / single harm-score inferencer (drop-in replacement for StrongREJECT).

    Score is in [0, 1]; higher = more harmful / jailbroken.
    """

    _cache: "dict[tuple, HarmScoreEvaluator]" = {}

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        backbone_path: Optional[str] = None,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        max_len: int = DEFAULT_MAX_LEN,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.ckpt_path = ckpt_path or os.environ.get("ECHO_EVALUATOR_CKPT")
        self.backbone_path = backbone_path or os.environ.get("ECHO_EVALUATOR_BACKBONE")
        if not self.ckpt_path:
            raise ValueError("Missing evaluator checkpoint path. Pass --evaluator_ckpt or set ECHO_EVALUATOR_CKPT.")
        if not self.backbone_path:
            raise ValueError("Missing evaluator backbone path. Pass --evaluator_backbone or set ECHO_EVALUATOR_BACKBONE.")
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.dtype = dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"Evaluator checkpoint not found: {self.ckpt_path}")
        if not Path(self.backbone_path).exists():
            raise FileNotFoundError(f"Evaluator backbone not found: {self.backbone_path}")

        print(f"[HarmScoreEvaluator] backbone = {self.backbone_path}")
        print(f"[HarmScoreEvaluator] ckpt     = {self.ckpt_path}")
        print(f"[HarmScoreEvaluator] device   = {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.backbone_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id or 0

        backbone = AutoModelForCausalLM.from_pretrained(
            self.backbone_path,
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation="sdpa",
        )
        if hasattr(backbone, "lm_head"):
            backbone.lm_head = nn.Identity()

        # Training used LoRA (not freeze_backbone), so we wrap a PEFT model so that
        # the saved state_dict keys match.
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        backbone = get_peft_model(backbone, lora_cfg)

        self.model = _HarmScoreModel(backbone, hidden_size=self.hidden_size)
        self.model.score_head = self.model.score_head.float()
        self.model.to(self.device)

        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[HarmScoreEvaluator] warn: {len(unexpected)} unexpected keys "
                  f"(first 3): {unexpected[:3]}")
        # Head weights should always be present; complain if not.
        head_missing = [k for k in missing if "score_head" in k]
        if head_missing:
            raise RuntimeError(
                f"score_head weights missing from checkpoint: {head_missing}"
            )
        print(f"[HarmScoreEvaluator] loaded state_dict "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @classmethod
    def get(
        cls,
        ckpt_path: Optional[str] = None,
        backbone_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "HarmScoreEvaluator":
        """Return a cached singleton for (ckpt, backbone, device)."""
        key = (
            ckpt_path or os.environ.get("ECHO_EVALUATOR_CKPT"),
            backbone_path or os.environ.get("ECHO_EVALUATOR_BACKBONE"),
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        inst = cls._cache.get(key)
        if inst is None:
            inst = cls(ckpt_path=key[0], backbone_path=key[1], device=key[2])
            cls._cache[key] = inst
        return inst

    def _encode(self, texts: List[str]) -> dict:
        encs = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        ids_list = encs["input_ids"]
        mask_list = encs["attention_mask"]
        # Ensure every sequence has at least one token (avoid seq_lens = -1).
        for i in range(len(ids_list)):
            if len(ids_list[i]) == 0:
                ids_list[i] = [self.pad_id]
                mask_list[i] = [1]
        max_len = max(len(x) for x in ids_list)
        padded_ids = []
        padded_mask = []
        for ids, msk in zip(ids_list, mask_list):
            pad = max_len - len(ids)
            padded_ids.append(ids + [self.pad_id] * pad)
            padded_mask.append(msk + [0] * pad)
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long, device=self.device),
        }

    @torch.inference_mode()
    def batch_score(self, responses: List[str], micro_batch: int = 8) -> List[float]:
        """Score a list of response texts. Returns floats in [0, 1]."""
        if not responses:
            return []
        scores: List[float] = []
        for i in range(0, len(responses), micro_batch):
            chunk = [r if r else " " for r in responses[i:i + micro_batch]]
            batch = self._encode(chunk)
            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
                preds = self.model(batch["input_ids"], batch["attention_mask"])
            preds = preds.float().clamp(0.0, 1.0).cpu().tolist()
            scores.extend(float(p) for p in preds)
        return scores

    def score(self, response: str) -> float:
        return self.batch_score([response])[0]

    def evaluate(self, goal: str, response: str) -> float:  # noqa: ARG002
        """Drop-in replacement: StrongREJECT used ``evaluate(goal, response)``.

        The fine-tuned model is response-only, so goal is unused.
        """
        return self.score(response)

    def batch_evaluate(
        self, goal: str, responses: List[str]  # noqa: ARG002
    ) -> List[float]:
        return self.batch_score(responses)


def load_default_evaluator(
    ckpt_path: Optional[str] = None,
    backbone_path: Optional[str] = None,
    device: Optional[str] = None,
) -> HarmScoreEvaluator:
    return HarmScoreEvaluator.get(
        ckpt_path=ckpt_path, backbone_path=backbone_path, device=device
    )

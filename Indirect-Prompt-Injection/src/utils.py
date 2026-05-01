"""
Utility functions: model loading, attention hook registration, config.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
RESULTS_DIR  = PROJECT_ROOT / "results"
for d in [DATA_DIR, CKPT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model configuration ──────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen3-8B"
NUM_HEADS = 32       # Q-heads in Qwen3-8B
NUM_KV_HEADS = 8     # KV-heads (GQA)
NUM_LAYERS = 36
K_LAYERS  = 4        # Top-K attack-sensitive layers selected for LSTP

# Mitigating Rectifier hyperparameter search space
GAMMA_GRID = [0.1, 0.2, 0.3, 0.5, 0.7]
TAU_GRID   = [0.05, 0.10, 0.15, 0.20, 0.30]


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str = MODEL_ID, device_map: str = "auto"):
    """Load Qwen3-8B in bf16 with eager attention (L40S has 46 GB VRAM, no quantization needed)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model {model_id} in bf16 (eager attention)…")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        attn_implementation="eager",   # exposes pre-softmax attention weights
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer


# ── Attention hook registration ──────────────────────────────────────────────

class AttentionCapture:
    """
    Registers forward hooks on every self_attn layer to capture attention
    weight matrices A_{l,h} ∈ R^{N_Y × N} before they are projected.

    Qwen3 eager attention returns `attn_weights` as the second element from
    the self_attn forward call; the hook intercepts this.
    """

    def __init__(self, model):
        self.model   = model
        self.hooks   = []
        # attn_store[layer_idx] = Tensor (num_heads, N_Y, N)
        self.attn_store: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output is (attn_output, attn_weights, past_key_value)
            # attn_weights shape: (batch, num_heads, seq_q, seq_k)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    N_Y = attn_weights.shape[2]
                    # Skip prefill (N_Y == N, can be 1000s of tokens → GBs of RAM).
                    # Only capture decode steps (N_Y == 1) where attention is small.
                    # FIS measures generation-time focus, so decode attention is correct.
                    if N_Y == 1:
                        self.attn_store[layer_idx] = (
                            attn_weights[0].detach().float().cpu()  # (H, 1, N)
                        )
        return hook_fn

    def register(self):
        """Register hooks on all self_attn modules."""
        self.remove()
        for idx, layer in enumerate(self.model.model.layers):
            h = layer.self_attn.register_forward_hook(self._make_hook(idx))
            self.hooks.append(h)
        logger.info(f"Registered attention hooks on {len(self.hooks)} layers.")

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.attn_store.clear()

    def get_attn(self) -> Dict[int, torch.Tensor]:
        """Return captured attention weights and clear the store."""
        store = dict(self.attn_store)
        self.attn_store.clear()
        return store


# ── Misc helpers ─────────────────────────────────────────────────────────────

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

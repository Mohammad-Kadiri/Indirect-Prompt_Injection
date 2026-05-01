"""
Focus Intensity Score (FIS) — ICON Section 4.5.

Given captured attention weight matrices A_{l,h} ∈ R^{N_Y × N},
computes per-head entropy, FIS, and the 3-dim descriptor v_{l,h}.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

EPS = 1e-9   # numerical stability


def compute_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    Generation-normalised entropy for every query token.

    Args:
        attn: (N_Y, N)  attention weights from a single head (post-softmax).
    Returns:
        entropy: (N_Y,)  normalised entropy per generated token.
    """
    N = attn.shape[1]
    log_N = torch.log(torch.tensor(N, dtype=torch.float32))
    ent = -(attn * torch.log(attn + EPS)).sum(dim=-1) / log_N   # (N_Y,)
    return ent.clamp(0.0, 1.0)


def compute_fis_head(attn: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """
    FIS for a single head.

    Args:
        attn: (N_Y, N)
    Returns:
        fis_score: scalar S_{l,h} = 1 - mean(E)
        entropy_seq: (N_Y,) per-token entropy
    """
    ent = compute_entropy(attn)
    fis = 1.0 - ent.mean().item()
    return fis, ent


def compute_fis_layer(attn_layer: torch.Tensor) -> Tuple[List[float], List[torch.Tensor]]:
    """
    FIS for all heads in a layer.

    Args:
        attn_layer: (H, N_Y, N)
    Returns:
        fis_heads: list of H FIS scores
        ent_heads:  list of H entropy sequences
    """
    H = attn_layer.shape[0]
    fis_heads, ent_heads = [], []
    for h in range(H):
        fis, ent = compute_fis_head(attn_layer[h])
        fis_heads.append(fis)
        ent_heads.append(ent)
    return fis_heads, ent_heads


def compute_descriptor(ent_seq: torch.Tensor) -> torch.Tensor:
    """
    3-dim descriptor v_{l,h} = [min_E, mean_E, std_E].

    Args:
        ent_seq: (N_Y,)
    Returns:
        v: (3,)
    """
    return torch.stack([
        ent_seq.min(),
        ent_seq.mean(),
        ent_seq.std() if ent_seq.numel() > 1 else torch.zeros(1, device=ent_seq.device).squeeze(),
    ])


def compute_all_fis(
    attn_store: Dict[int, torch.Tensor],
) -> Dict[str, object]:
    """
    Compute FIS for every layer and head.

    Args:
        attn_store: {layer_idx: (H, N_Y, N)}
    Returns:
        result dict with keys:
            'layer_fis'  : {layer_idx: float}     mean FIS over heads
            'head_fis'   : {layer_idx: [float]*H}
            'descriptors': {layer_idx: Tensor(H,3)}
    """
    layer_fis   = {}
    head_fis    = {}
    descriptors = {}

    for l_idx, attn in attn_store.items():
        fis_heads, ent_heads = compute_fis_layer(attn)
        layer_fis[l_idx]    = float(np.mean(fis_heads))
        head_fis[l_idx]     = fis_heads
        # descriptor per head
        desc = torch.stack([compute_descriptor(e) for e in ent_heads])  # (H, 3)
        descriptors[l_idx] = desc

    return {
        "layer_fis":   layer_fis,
        "head_fis":    head_fis,
        "descriptors": descriptors,
    }


def select_top_k_layers(layer_fis: Dict[int, float], k: int = 4) -> List[int]:
    """Return indices of K layers with highest mean FIS."""
    sorted_layers = sorted(layer_fis.items(), key=lambda x: x[1], reverse=True)
    return [l for l, _ in sorted_layers[:k]]


def build_feature_vector(
    descriptors: Dict[int, torch.Tensor],
    top_k_layers: List[int],
) -> torch.Tensor:
    """
    Concatenate per-head descriptors from top-K layers into global feature z.

    Shape: (3, H, K) → flattened: (3*H*K,) = (384,) for H=32, K=4.

    For 1D-CNN input: returned as (H*3, K) = (96, 4) where K is the
    'sequence length' dimension and H*3 is the channel dimension.
    """
    parts = []
    for l_idx in top_k_layers:
        if l_idx in descriptors:
            # descriptors[l_idx]: (H, 3)
            parts.append(descriptors[l_idx])   # (H, 3)
        else:
            H = 32
            parts.append(torch.zeros(H, 3))

    # Stack over K layers: (K, H, 3)
    z_raw = torch.stack(parts, dim=0)          # (K, H, 3)
    # Reshape for 1D-CNN: (channels=H*3, length=K)
    K, H, three = z_raw.shape
    z = z_raw.permute(1, 2, 0).reshape(H * three, K)  # (96, K)
    return z   # (96, K)

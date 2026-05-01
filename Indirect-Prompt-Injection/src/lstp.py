"""
Latent Space Trace Prober (LSTP) — ICON Section 4.6.

Architecture: 1D-CNN (2 conv layers) + AdaptiveMaxPool1d + 2-layer MLP.
Input shape: (batch, H*3, K) = (batch, 96, 4) for H=32, K=4.
Output: binary logit (attack / benign).

Parameter count target: ~31,553 (matching the paper).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class LSTP(nn.Module):
    """
    Lightweight probe for IPI detection.

    Conv layers capture inter-head/inter-layer dependencies.
    AdaptiveMaxPool1d ensures invariance to generation length.
    """

    def __init__(self, in_channels: int = 96, k_layers: int = 4):
        super().__init__()

        # Branch 1: convolutional feature extractor
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=2, padding=1),  # (B, 128, K+1)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=2, padding=1),            # (B, 64,  K+2)
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)   # (B, 64, 1)

        # Branch 2: global MLP
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 96, K)  feature tensor from build_feature_vector()
        Returns:
            logit: (batch,)
        """
        h = self.conv_block(x)     # (B, 64, L)
        h = self.pool(h).squeeze(-1)  # (B, 64)
        return self.mlp(h).squeeze(-1)  # (B,)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training helpers ─────────────────────────────────────────────────────────

def train_lstp(
    model: LSTP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    epochs:  int = 50,
    lr:      float = 1e-3,
    batch_size: int = 32,
    ckpt_path: Optional[Path] = None,
) -> dict:
    """
    Train LSTP with binary cross-entropy.

    Args:
        X_train/X_val: (N, 96, K)
        y_train/y_val: (N,)  float labels 0/1
    Returns:
        history dict with train/val losses and best val accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    X_tr = X_train.to(device); y_tr = y_train.to(device)
    X_v  = X_val.to(device);   y_v  = y_val.to(device)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            idx  = perm[i: i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss   = criterion(val_logits, y_v).item()
            preds      = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc    = (preds == y_v).float().mean().item()

        history["train_loss"].append(epoch_loss / len(X_tr))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train_loss={epoch_loss/len(X_tr):.4f} "
                  f"| val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    history["best_val_acc"] = best_val_acc
    return history


def predict_proba(model: LSTP, x: torch.Tensor) -> torch.Tensor:
    """Return P(attack) for a batch."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
    return torch.sigmoid(logits).cpu()


def is_attack(model: LSTP, x: torch.Tensor, threshold: float = 0.5) -> bool:
    """Single-sample attack decision."""
    prob = predict_proba(model, x.unsqueeze(0)).item()
    return prob >= threshold

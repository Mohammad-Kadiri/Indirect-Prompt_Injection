"""
Step 2: Train LSTP probe on the synthesized dataset.

Pipeline:
  1. Load Qwen3-8B with 4-bit NF4 quantization.
  2. Register attention capture hooks.
  3. Run each sample through the model, collecting FIS feature vectors.
  4. Select top-K attack-sensitive layers by mean FIS differential.
  5. Train LSTP 1D-CNN+MLP binary classifier.
  6. Grid-search γ and τ for the Mitigating Rectifier.
  7. Save LSTP checkpoint and config.

Output: checkpoints/lstp.pt, checkpoints/icon_config.json
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.utils        import (load_model_and_tokenizer, AttentionCapture,
                               DATA_DIR, CKPT_DIR, GAMMA_GRID, TAU_GRID,
                               K_LAYERS, NUM_HEADS, set_seed, save_json, load_json)
from src.fis          import compute_all_fis, select_top_k_layers, build_feature_vector
from src.lstp         import LSTP, train_lstp
from src.agent        import MockToolExecutor, run_react_agent, build_initial_scratchpad
from src.data_synthesis import BENIGN_SCENARIOS


def extract_features_for_dataset(
    dataset:     list,
    model,
    tokenizer,
    attn_capture: AttentionCapture,
    k_layers:    int = K_LAYERS,
    max_samples: int = 255,
) -> tuple:
    """
    Run each dataset sample through the model and extract FIS feature vectors.

    Returns:
        X: (N, 96, K) tensor
        y: (N,)       float labels
        top_k_layers: list of K layer indices
    """
    all_feats, all_labels = [], []
    layer_fis_attack, layer_fis_benign = [], []

    print(f"Extracting features for {min(len(dataset), max_samples)} samples...")
    for i, sample in enumerate(dataset[:max_samples]):
        if i % 20 == 0:
            print(f"  Processing sample {i}/{min(len(dataset), max_samples)}...")

        injection = sample["payload"] if sample["label"] == 1 else None
        executor  = MockToolExecutor(
            injection=injection,
            inject_tool=sample.get("inject_tool", "GmailReadEmail"),
        )

        # For attack samples, pre-populate the scratchpad so the model actually
        # sees the injected payload before generating. Without this, attack
        # attention patterns are indistinguishable from benign ones.
        if sample["label"] == 1:
            init_pad = build_initial_scratchpad(
                executor, sample.get("inject_tool", "GmailReadEmail")
            )
        else:
            init_pad = ""

        attn_store_cap = {}
        attn_capture.register()

        def cap_fn():
            attn_store_cap.update(attn_capture.get_attn())
            return attn_store_cap

        run_react_agent(
            user_task=sample["user_task"],
            model=model,
            tokenizer=tokenizer,
            executor=executor,
            capture_fn=cap_fn,
            initial_scratchpad=init_pad,
        )
        attn_capture.remove()
        attn_capture.attn_store.clear()

        if not attn_store_cap:
            print(f"  WARNING: No attention captured for sample {i}")
            continue

        fis_result = compute_all_fis(attn_store_cap)
        if sample["label"] == 1:
            layer_fis_attack.append(fis_result["layer_fis"])
        else:
            layer_fis_benign.append(fis_result["layer_fis"])

        all_feats.append(fis_result)
        all_labels.append(float(sample["label"]))

    # ── Select top-K attack-sensitive layers ─────────────────────────────────
    # Layers with largest mean FIS differential (attack vs benign)
    if layer_fis_attack and layer_fis_benign:
        attack_mean = {}
        benign_mean = {}
        for l_idx in layer_fis_attack[0]:
            attack_mean[l_idx] = float(np.mean([f[l_idx] for f in layer_fis_attack if l_idx in f]))
            benign_mean[l_idx] = float(np.mean([f[l_idx] for f in layer_fis_benign if l_idx in f]))

        differential = {l: attack_mean.get(l, 0) - benign_mean.get(l, 0)
                        for l in attack_mean}
        top_k_layers = sorted(differential, key=differential.get, reverse=True)[:k_layers]
        print(f"\nTop-{k_layers} attack-sensitive layers: {top_k_layers}")
        for l in top_k_layers:
            print(f"  Layer {l:2d}: attack_FIS={attack_mean[l]:.4f}, "
                  f"benign_FIS={benign_mean.get(l, 0):.4f}, "
                  f"diff={differential[l]:.4f}")
    else:
        top_k_layers = list(range(20, 20 + k_layers))
        print(f"Fallback: using layers {top_k_layers}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    X_list = []
    for fis_result in all_feats:
        feat = build_feature_vector(fis_result["descriptors"], top_k_layers)
        X_list.append(feat)

    X = torch.stack(X_list, dim=0)          # (N, 96, K)
    X = torch.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)  # guard NaN from short seqs
    y = torch.tensor(all_labels, dtype=torch.float32)  # (N,)

    print(f"\nFeature matrix: X={X.shape}, y={y.shape}")
    print(f"Class balance: {y.sum().int()} attack, {(1-y).sum().int()} benign")
    return X, y, top_k_layers


def main(args):
    set_seed(args.seed)

    print("=" * 60)
    print("Step 2: LSTP Training")
    print("=" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = DATA_DIR / "ipi_dataset.json"
    if not dataset_path.exists():
        print("Dataset not found — run step1_synthesize.py first.")
        sys.exit(1)
    dataset = load_json(dataset_path)
    print(f"Loaded dataset: {len(dataset)} samples")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer()
    attn_capture = AttentionCapture(model)

    # ── Extract features ──────────────────────────────────────────────────────
    X, y, top_k_layers = extract_features_for_dataset(
        dataset, model, tokenizer, attn_capture,
        k_layers=K_LAYERS, max_samples=args.max_samples,
    )

    # ── Train/val split (80/20) ───────────────────────────────────────────────
    indices = list(range(len(X)))
    tr_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed,
        stratify=y.numpy().astype(int),
    )
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_v,  y_v  = X[val_idx], y[val_idx]
    print(f"\nTrain: {len(X_tr)}, Val: {len(X_v)}")

    # ── Train LSTP ────────────────────────────────────────────────────────────
    lstp = LSTP(in_channels=NUM_HEADS * 3, k_layers=K_LAYERS)
    print(f"LSTP parameter count: {lstp.count_params():,}")

    ckpt_path = CKPT_DIR / "lstp.pt"
    history = train_lstp(
        lstp, X_tr, y_tr, X_v, y_v,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ckpt_path=ckpt_path,
    )

    # ── Save config ───────────────────────────────────────────────────────────
    icon_config = {
        "top_k_layers":  top_k_layers,
        "k_layers":      K_LAYERS,
        "num_heads":     NUM_HEADS,
        "gamma":         args.gamma,   # initial; refined by grid search in step 3
        "tau":           args.tau,
        "threshold":     args.threshold,
        "train_history": history,
        "lstp_params":   lstp.count_params(),
    }
    config_path = CKPT_DIR / "icon_config.json"
    save_json(icon_config, config_path)

    print(f"\nCheckpoint saved:    {ckpt_path}")
    print(f"Config saved:        {config_path}")
    print(f"Best val accuracy:   {history['best_val_acc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=255)
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--gamma",       type=float, default=0.3)
    parser.add_argument("--tau",         type=float, default=0.10)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--seed",        type=int, default=42)
    main(parser.parse_args())

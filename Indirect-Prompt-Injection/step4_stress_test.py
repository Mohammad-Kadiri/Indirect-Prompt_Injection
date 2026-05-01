"""
Step 4: Adversarial Stress-Testing — four attacks against ICON.

Attacks:
  1. FIS-Aware Latent Mimicry      (optimises against FIS directly)
  2. Long-Context Attention Dilution (padding lengths: 100/300/500/1000)
  3. Gradual Multi-Step Injection   (splits intent across 2 steps)
  4. Attention Sink Routing         (hides signal in sink tokens)

Metrics: ASR, effective attack rate (success AND evades LSTP).

Output: results/step4_stress_test_results.json
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from pathlib import Path

from src.utils       import (load_model_and_tokenizer, AttentionCapture,
                              CKPT_DIR, RESULTS_DIR, set_seed, load_json, save_json)
from src.lstp        import LSTP
from src.data_synthesis import BENIGN_SCENARIOS

from src.attacks.attack1_fis_mimicry      import run_attack1
from src.attacks.attack2_context_dilution import run_attack2
from src.attacks.attack3_multistep        import run_attack3
from src.attacks.attack4_sink_routing     import run_attack4
from src.attacks.attack5_naturalistic     import run_attack5
from src.attacks.attack6_evasion_finisher import run_attack6


def load_lstp_and_config():
    config = load_json(CKPT_DIR / "icon_config.json")
    lstp   = LSTP()
    lstp.load_state_dict(torch.load(CKPT_DIR / "lstp.pt", map_location="cpu"))
    lstp.eval()
    return lstp, config


def summarise_attack(results: list, attack_name: str) -> dict:
    asr = np.mean([r.get("attack_success", False) for r in results]) * 100
    eff = np.mean([r.get("effective_attack", False) for r in results]) * 100
    det = np.mean([not r.get("icon_detects", True)  for r in results]) * 100
    return {
        "attack":          attack_name,
        "n_samples":       len(results),
        "ASR":             round(asr, 2),
        "evasion_rate":    round(det, 2),
        "effective_rate":  round(eff, 2),
    }


def cached(path: Path, name: str, run_fn, *args, **kwargs):
    """Run attack and cache result; load from cache on resubmit."""
    if path.exists():
        print(f"  [checkpoint] {name} already done — loading from cache.")
        return load_json(path)
    result = run_fn(*args, **kwargs)
    save_json(result, path)
    return result


def main(args):
    set_seed(args.seed)
    print("=" * 60)
    print("Step 4: Adversarial Stress-Testing of ICON")
    print("=" * 60)

    lstp, config = load_lstp_and_config()
    top_k_layers = config["top_k_layers"]
    threshold    = config["threshold"]

    model, tokenizer = load_model_and_tokenizer()
    attn_capture     = AttentionCapture(model)

    all_results = {}
    summary     = []

    # ── Attack 1: FIS-Aware Latent Mimicry ────────────────────────────────────
    print("\n" + "─" * 50)
    print("Attack 1: FIS-Aware Latent Mimicry")
    a1_results = cached(
        RESULTS_DIR / "attack1_fis_mimicry.json", "Attack 1",
        run_attack1,
        model=model, tokenizer=tokenizer,
        lstp_model=lstp, attn_capture=attn_capture,
        top_k_layers=top_k_layers,
        threshold=threshold,
        n_samples=args.n_samples,
    )
    all_results["attack1"] = a1_results
    summary.append(summarise_attack(a1_results, "FIS-Aware Latent Mimicry"))

    # ── Attack 2: Long-Context Attention Dilution ─────────────────────────────
    print("\n" + "─" * 50)
    print("Attack 2: Long-Context Attention Dilution")
    a2_results = cached(
        RESULTS_DIR / "attack2_context_dilution.json", "Attack 2",
        run_attack2,
        model=model, tokenizer=tokenizer,
        lstp_model=lstp, attn_capture=attn_capture,
        top_k_layers=top_k_layers,
        threshold=threshold,
        n_samples=args.n_samples,
    )
    all_results["attack2"] = a2_results
    summary.append(summarise_attack(a2_results, "Long-Context Attention Dilution"))

    # Attack 2 detailed: ASR per padding length
    print("\nAttack 2 — ASR by padding length:")
    from src.attacks.attack2_context_dilution import PADDING_LENGTHS
    for pad_len in PADDING_LENGTHS:
        subset = [r for r in a2_results if r["padding_length"] == pad_len]
        if subset:
            asr = np.mean([r["attack_success"] for r in subset]) * 100
            det = np.mean([r["icon_detects"]   for r in subset]) * 100
            print(f"  pad={pad_len:5d} | ASR={asr:.1f}%  ICON_det={det:.1f}%")

    # ── Attack 3: Gradual Multi-Step Injection ────────────────────────────────
    print("\n" + "─" * 50)
    print("Attack 3: Gradual Multi-Step Injection")
    a3_results = cached(
        RESULTS_DIR / "attack3_multistep.json", "Attack 3",
        run_attack3,
        model=model, tokenizer=tokenizer,
        lstp_model=lstp, attn_capture=attn_capture,
        top_k_layers=top_k_layers,
        threshold=threshold,
        n_samples=args.n_samples,
    )
    all_results["attack3"] = a3_results
    summary.append(summarise_attack(a3_results, "Gradual Multi-Step Injection"))

    # ── Attack 4: Attention Sink Routing ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("Attack 4: Attention Sink Routing")
    a4_results = cached(
        RESULTS_DIR / "attack4_sink_routing.json", "Attack 4",
        run_attack4,
        model=model, tokenizer=tokenizer,
        lstp_model=lstp, attn_capture=attn_capture,
        top_k_layers=top_k_layers,
        threshold=threshold,
        n_samples=args.n_samples,
    )
    all_results["attack4"] = a4_results
    summary.append(summarise_attack(a4_results, "Attention Sink Routing"))

    # ── Attack 5: Naturalistic Prefill Injection ──────────────────────────────
    print("\n" + "─" * 50)
    print("Attack 5: Naturalistic Prefill Injection")
    a5_results = cached(
        RESULTS_DIR / "attack5_naturalistic.json", "Attack 5",
        run_attack5,
        model=model, tokenizer=tokenizer,
        lstp_model=lstp, attn_capture=attn_capture,
        top_k_layers=top_k_layers,
        threshold=threshold,
        n_samples=args.n_samples,
    )
    all_results["attack5"] = a5_results
    summary.append(summarise_attack(a5_results, "Naturalistic Prefill Injection"))

    # FIS comparison: naturalistic vs explicit
    avg_fis_nat = np.mean([r["fis_score"] for r in a5_results])
    avg_fis_exp = np.mean([r["fis_score"] for r in a1_results])
    print(f"\n  FIS comparison — Naturalistic: {avg_fis_nat:.3f}  "
          f"vs  Explicit (A1): {avg_fis_exp:.3f}")

    # ── Attack 6: Evasion-Aware Finisher ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("Attack 6: Evasion-Aware Finisher (Attention Window Manipulation)")
    a6_results = cached(
        RESULTS_DIR / "attack6_evasion_finisher.json", "Attack 6",
        run_attack6,
        model=model, tokenizer=tokenizer,
        lstp_model=lstp, attn_capture=attn_capture,
        top_k_layers=top_k_layers,
        threshold=threshold,
        n_samples=args.n_samples,
    )
    all_results["attack6"] = a6_results
    summary.append(summarise_attack(a6_results, "Evasion-Aware Finisher"))

    # Per-scenario: show steps and whether multi-step separation worked
    n_multistep = sum(1 for r in a6_results if r["n_steps"] >= 2)
    print(f"\n  Multi-step separation achieved: {n_multistep}/{len(a6_results)} scenarios")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STRESS-TEST SUMMARY")
    print("=" * 60)
    hdr = f"{'Attack':<35} {'ASR':>6} {'Evasion':>8} {'Effective':>10}"
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['attack']:<35} {s['ASR']:>5.1f}%  {s['evasion_rate']:>6.1f}%  "
              f"{s['effective_rate']:>8.1f}%")

    save_json(summary,     RESULTS_DIR / "step4_stress_test_results.json")
    save_json(all_results, RESULTS_DIR / "step4_all_raw_results.json")
    print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Scenarios to test per attack")
    parser.add_argument("--seed",      type=int, default=42)
    main(parser.parse_args())

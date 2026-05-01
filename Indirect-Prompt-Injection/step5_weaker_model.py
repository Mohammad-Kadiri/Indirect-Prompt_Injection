"""
Step 5: Weaker-model comparison.

Runs the same IPI attacks (no defense) on Qwen2-1.5B-Instruct to show that
attacks succeed on a less capable model — completing the story:

  Qwen2-1.5B (no defense)  →  high ASR  (attacks work on weaker models)
  Qwen3-8B   (no defense)  →  low  ASR  (strong model, natural resistance)
  Qwen3-8B   (ICON)        →  low  ASR + 100% ADR  (detection guarantee)

Output: results/step5_weaker_model_results.json
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
from tqdm import tqdm
from pathlib import Path

from src.utils        import (DATA_DIR, RESULTS_DIR, set_seed, load_json, save_json)
from src.agent        import (MockToolExecutor, run_react_agent, check_attack_success,
                               check_utility, build_initial_scratchpad)
from src.evaluate     import build_result_record, evaluate_defense, print_results_table
from src.data_synthesis import (IGNORE_INSTRUCTION_TEMPLATES, COMBINED_TEMPLATES,
                                 STEALTHY_TEMPLATES)

WEAK_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"


def load_weak_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Loading weak model: {WEAK_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        WEAK_MODEL_ID, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        WEAK_MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    logger.info(f"Weak model loaded.")
    return model, tokenizer


def build_scenarios(bm_cases, attack_type, n):
    import random
    scenarios = []
    for case in bm_cases[:n]:
        base = {
            "user_task":     case["user_instruction"],
            "inject_tool":   case.get("inject_tool",   "GmailReadEmail"),
            "attacker_tool": case.get("attacker_tool", "GmailSendEmail"),
            "expected_tool": case.get("user_tool",     "GmailReadEmail"),
        }
        if attack_type == "ignore_instruction":
            payload = IGNORE_INSTRUCTION_TEMPLATES[0].format(
                goal=f"call {base['attacker_tool']} immediately"
            )
        elif attack_type == "stealthy":
            payload = STEALTHY_TEMPLATES[0].format(
                goal=f"call {base['attacker_tool']} immediately"
            )
        else:
            payload = random.choice(COMBINED_TEMPLATES).format(
                goal=f"call {base['attacker_tool']} immediately"
            )
        base["payload"] = payload
        scenarios.append(base)
    return scenarios


def run_no_defense(scenarios, model, tokenizer, desc="No Defense"):
    results = []
    for sc in tqdm(scenarios, desc=desc):
        executor = MockToolExecutor(injection=sc["payload"], inject_tool=sc["inject_tool"])
        init_pad = build_initial_scratchpad(executor, sc["inject_tool"])
        out      = run_react_agent(sc["user_task"], model, tokenizer, executor,
                                   initial_scratchpad=init_pad)
        rec      = build_result_record(out, sc, is_attack_scenario=True)
        results.append(rec)
    return results


def main(args):
    set_seed(args.seed)

    print("=" * 60)
    print("Step 5: Weaker Model Comparison")
    print(f"Model: {WEAK_MODEL_ID}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    bm_path = DATA_DIR / "injectagent_benchmark.json"
    if not bm_path.exists():
        print("Benchmark not found — run step1 first.")
        sys.exit(1)
    bm_cases = load_json(bm_path)
    print(f"Benchmark: {len(bm_cases)} cases")

    # ── Load weak model ───────────────────────────────────────────────────────
    model, tokenizer = load_weak_model()

    all_results = []

    for attack_type in ["ignore_instruction", "stealthy", "combined_attacks"]:
        ckpt = RESULTS_DIR / f"step5_weak_{attack_type}.json"
        print(f"\n{'─'*50}")
        print(f"Attack type: {attack_type.replace('_', ' ').title()}")

        if ckpt.exists():
            print(f"  [checkpoint] Loading from cache.")
            results = load_json(ckpt)
        else:
            scenarios = build_scenarios(bm_cases, attack_type, n=args.n_eval)
            results   = run_no_defense(scenarios, model, tokenizer,
                                       desc=attack_type.replace("_", " ").title())
            save_json(results, ckpt)

        metrics = evaluate_defense(results, defense_name=f"Qwen2-1.5B No Defense")
        metrics["attack_type"] = attack_type
        metrics["model"]       = WEAK_MODEL_ID

        asr = metrics["ASR"]
        ua  = metrics["UA"]
        print(f"  ASR={asr}%  UA={ua}%  (n={len(results)})")
        all_results.append(metrics)

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL COMPARISON: Ignore Instruction Attack")
    print("=" * 65)

    # Load Qwen3-8B step3 results for comparison
    step3_path = RESULTS_DIR / "step3_eval_results.json"
    if step3_path.exists():
        step3 = load_json(step3_path)
        ii_results = [r for r in step3 if r["attack_type"] == "ignore_instruction"]

        print(f"\n{'Model / Defense':<35} {'ASR↓':>6}  {'UA↑':>6}  {'ADR↑':>6}")
        print("-" * 58)

        # Weak model row
        weak_ii = next((r for r in all_results if r["attack_type"] == "ignore_instruction"), None)
        if weak_ii:
            print(f"{'Qwen2-1.5B  (No Defense)':<35} {weak_ii['ASR']:>5.1f}%  {weak_ii['UA']:>5.1f}%  {'N/A':>6}")

        # Qwen3-8B rows from step3
        for r in ii_results:
            label = f"Qwen3-8B    ({r['defense']})"
            adr   = r.get('ADR', 0.0)
            print(f"{label:<35} {r['ASR']:>5.1f}%  {r['UA']:>5.1f}%  {adr:>5.1f}%")

    save_json(all_results, RESULTS_DIR / "step5_weaker_model_results.json")
    print(f"\nResults saved to {RESULTS_DIR}/step5_weaker_model_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--seed",   type=int, default=42)
    main(parser.parse_args())

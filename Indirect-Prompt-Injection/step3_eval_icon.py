"""
Step 3: Evaluate ICON on the InjectAgent benchmark.

Compares six conditions:
  1. No Defense
  2. Repeat Prompt
  3. Delimiting
  4. ICON (our replication)

For each condition and attack type (Ignore Instruction, Combined Attacks),
reports ASR, UA, ADR, URR.

Output: results/step3_eval_results.json
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path
from tqdm import tqdm

from src.utils         import (load_model_and_tokenizer, AttentionCapture,
                                DATA_DIR, CKPT_DIR, RESULTS_DIR,
                                set_seed, load_json, save_json)
from src.lstp          import LSTP
from src.icon          import ICONDefense
from src.agent         import (MockToolExecutor, run_react_agent, check_attack_success,
                                check_utility, build_initial_scratchpad)
from src.evaluate      import (build_result_record, evaluate_defense,
                                print_results_table, save_results)
from src.baselines.repeat_prompt import run_repeat_prompt
from src.baselines.delimiting    import run_delimiting, DelimitingExecutor
from src.data_synthesis          import (BENIGN_SCENARIOS, IGNORE_INSTRUCTION_TEMPLATES,
                                          STEALTHY_TEMPLATES, COMBINED_TEMPLATES,
                                          load_injectagent_benchmark)

CKPT_STEP3 = RESULTS_DIR / "step3_checkpoints"


def ckpt_path(attack_type: str, defense: str) -> Path:
    return CKPT_STEP3 / f"{defense}_{attack_type}.json"


def load_or_run(attack_type: str, defense: str, run_fn, *args):
    """Load cached results if they exist, otherwise run and cache."""
    p = ckpt_path(attack_type, defense)
    if p.exists():
        print(f"  [checkpoint] Loading {defense} / {attack_type} from cache.")
        return load_json(p)
    results = run_fn(*args)
    save_json(results, p)
    return results


def load_icon(model, tokenizer) -> ICONDefense:
    config    = load_json(CKPT_DIR / "icon_config.json")
    lstp      = LSTP()
    lstp.load_state_dict(torch.load(CKPT_DIR / "lstp.pt", map_location="cpu"))
    lstp.eval()

    icon = ICONDefense(
        model=model,
        tokenizer=tokenizer,
        gamma=config["gamma"],
        tau=config["tau"],
        threshold=config["threshold"],
        top_k_layers=config["top_k_layers"],
    )
    icon.lstp = lstp
    return icon


def build_attack_scenarios(bm_cases: list, attack_type: str, n: int = 50) -> list:
    scenarios = []
    for case in bm_cases[:n]:
        base = {
            "user_task":     case["user_instruction"],
            "inject_tool":   case.get("inject_tool", "GmailReadEmail"),
            "attacker_tool": case.get("attacker_tool", "GmailSendEmail"),
            "expected_tool": case.get("user_tool", "GmailReadEmail"),
        }

        if attack_type == "ignore_instruction":
            payload = IGNORE_INSTRUCTION_TEMPLATES[0].format(
                goal=f"call {base['attacker_tool']} immediately"
            )
        elif attack_type == "combined_attacks":
            import random
            payload = random.choice(COMBINED_TEMPLATES).format(
                goal=f"call {base['attacker_tool']} immediately"
            )
        else:
            payload = case.get("attacker_instruction", "")

        base["payload"] = payload
        scenarios.append(base)
    return scenarios


def run_no_defense(scenarios, model, tokenizer):
    results = []
    for sc in tqdm(scenarios, desc="No Defense"):
        executor  = MockToolExecutor(injection=sc["payload"], inject_tool=sc["inject_tool"])
        init_pad  = build_initial_scratchpad(executor, sc["inject_tool"])
        out       = run_react_agent(sc["user_task"], model, tokenizer, executor,
                                    initial_scratchpad=init_pad)
        rec       = build_result_record(out, sc, is_attack_scenario=True)
        results.append(rec)
    return results


def run_repeat_prompt_eval(scenarios, model, tokenizer):
    results = []
    for sc in tqdm(scenarios, desc="Repeat Prompt"):
        executor  = MockToolExecutor(injection=sc["payload"], inject_tool=sc["inject_tool"])
        init_pad  = build_initial_scratchpad(executor, sc["inject_tool"])
        out       = run_repeat_prompt(sc["user_task"], model, tokenizer, executor,
                                      initial_scratchpad=init_pad)
        rec       = build_result_record(out, sc, is_attack_scenario=True)
        results.append(rec)
    return results


def run_delimiting_eval(scenarios, model, tokenizer):
    from src.agent import INJECT_TOOL_THOUGHTS, INJECT_TOOL_INPUTS
    DELIM = "<<<TOOL_RESPONSE>>>"
    results = []
    for sc in tqdm(scenarios, desc="Delimiting"):
        executor = MockToolExecutor(injection=sc["payload"], inject_tool=sc["inject_tool"])
        # Get raw observation (with injection), then wrap in delimiters
        raw_obs  = executor.execute(sc["inject_tool"], {})
        executor.call_log.clear()
        delim_obs    = f"\n{DELIM}\n{raw_obs}\n{DELIM}\n"
        inject_tool  = sc["inject_tool"]
        thought      = INJECT_TOOL_THOUGHTS.get(inject_tool, f"I'll use {inject_tool} first.")
        action_input = INJECT_TOOL_INPUTS.get(inject_tool, "{}")
        init_pad     = (
            f"Thought: {thought}\n"
            f"Action: {inject_tool}\n"
            f"Action Input: {action_input}\n"
            f"Observation: {delim_obs}\n"
        )
        out = run_delimiting(sc["user_task"], model, tokenizer, executor,
                             initial_scratchpad=init_pad)
        rec = build_result_record(out, sc, is_attack_scenario=True)
        results.append(rec)
    return results


def run_icon_eval(scenarios, icon: ICONDefense):
    results = []
    for sc in tqdm(scenarios, desc="ICON"):
        executor  = MockToolExecutor(injection=sc["payload"], inject_tool=sc["inject_tool"])
        init_pad  = build_initial_scratchpad(executor, sc["inject_tool"])
        out       = icon.run_with_online_defense(sc["user_task"], executor,
                                                  initial_scratchpad=init_pad)
        rec       = build_result_record(out, sc, is_attack_scenario=True)
        results.append(rec)
    return results


def main(args):
    set_seed(args.seed)
    CKPT_STEP3.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 3: Evaluating ICON on InjectAgent Benchmark")
    print("=" * 60)

    # ── Load benchmark ────────────────────────────────────────────────────────
    bm_path = DATA_DIR / "injectagent_benchmark.json"
    if not bm_path.exists():
        print("Benchmark not found — run step1_synthesize.py first.")
        sys.exit(1)
    bm_cases = load_json(bm_path)
    print(f"Benchmark: {len(bm_cases)} cases")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer()

    # ── Load ICON ─────────────────────────────────────────────────────────────
    icon = load_icon(model, tokenizer)

    all_metrics = []

    for attack_type in ["ignore_instruction", "combined_attacks"]:
        print(f"\n{'─'*50}")
        print(f"Attack type: {attack_type.replace('_', ' ').title()}")
        scenarios = build_attack_scenarios(bm_cases, attack_type, n=args.n_eval)

        nd_res = load_or_run(attack_type, "no_defense",    run_no_defense,        scenarios, model, tokenizer)
        rp_res = load_or_run(attack_type, "repeat_prompt", run_repeat_prompt_eval, scenarios, model, tokenizer)
        dl_res = load_or_run(attack_type, "delimiting",    run_delimiting_eval,    scenarios, model, tokenizer)
        ic_res = load_or_run(attack_type, "icon",          run_icon_eval,          scenarios, icon)

        metrics = [
            evaluate_defense(nd_res,  defense_name="No Defense"),
            evaluate_defense(rp_res,  nd_res, "Repeat Prompt"),
            evaluate_defense(dl_res,  nd_res, "Delimiting"),
            evaluate_defense(ic_res,  nd_res, "ICON (Ours)"),
        ]
        for m in metrics:
            m["attack_type"] = attack_type

        print_results_table(metrics)
        all_metrics.extend(metrics)

        # Save raw results
        save_json(nd_res, RESULTS_DIR / f"no_defense_{attack_type}.json")
        save_json(ic_res, RESULTS_DIR / f"icon_{attack_type}.json")

    save_results(all_metrics, RESULTS_DIR / "step3_eval_results.json")
    print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval", type=int, default=50,
                        help="Number of benchmark cases per attack type")
    parser.add_argument("--seed",   type=int, default=42)
    main(parser.parse_args())

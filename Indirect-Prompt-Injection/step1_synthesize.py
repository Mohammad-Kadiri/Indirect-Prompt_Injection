"""
Step 1: Synthesize surrogate IPI training dataset (~255 samples).

Implements ICON Section 4.3 — LLM-as-Optimizer attack generation loop.
Falls back to rule-based payloads if OPENAI_API_KEY is not set.

Output: data/ipi_dataset.json
"""

import os
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import DATA_DIR, save_json, set_seed
from src.data_synthesis import synthesize_dataset, load_injectagent_benchmark


def main(args):
    set_seed(args.seed)

    print("=" * 60)
    print("Step 1: Adaptive Attack Data Synthesis")
    print("=" * 60)

    use_llm = bool(os.environ.get("OPENAI_API_KEY", ""))
    if use_llm:
        print("OpenAI API key found — using GPT-4o as LLM-as-Optimizer.")
    else:
        print("No OpenAI API key — using rule-based payload generator (fallback).")

    dataset = synthesize_dataset(
        n_samples=args.n_samples,
        max_rounds=args.max_rounds,
        use_llm=use_llm,
        api_key=os.environ.get("OPENAI_API_KEY"),
        seed=args.seed,
    )

    out_path = DATA_DIR / "ipi_dataset.json"
    save_json(dataset, out_path)
    print(f"\nDataset saved: {out_path}")
    print(f"Total samples: {len(dataset)}")
    print(f"  Attack:  {sum(d['label']==1 for d in dataset)}")
    print(f"  Benign:  {sum(d['label']==0 for d in dataset)}")

    # Also download / prepare InjectAgent benchmark
    print("\nLoading InjectAgent benchmark...")
    bm_cases = load_injectagent_benchmark(str(DATA_DIR))
    bm_path  = DATA_DIR / "injectagent_benchmark.json"
    save_json(bm_cases, bm_path)
    print(f"InjectAgent benchmark: {len(bm_cases)} test cases → {bm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int, default=255)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    main(parser.parse_args())

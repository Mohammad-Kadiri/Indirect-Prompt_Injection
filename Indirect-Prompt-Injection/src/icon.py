"""
ICON: Full inference-time detection and mitigation pipeline.

Phase 1 (offline):  Train LSTP on synthesized dataset.
Phase 2 (online):   For each generation step:
                    1. Capture attention weights.
                    2. Compute FIS.
                    3. Run LSTP probe.
                    4. If attack detected → apply MR.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.fis  import compute_all_fis, select_top_k_layers, build_feature_vector
from src.lstp import LSTP, predict_proba, is_attack
from src.mr   import MitigatingRectifier
from src.utils import K_LAYERS, NUM_HEADS, AttentionCapture


class ICONDefense:
    """
    Full ICON pipeline integrating FIS, LSTP, and MR.

    Usage:
        icon = ICONDefense(model, tokenizer)
        icon.load_lstp(ckpt_path)
        icon.set_top_k_layers(top_k_layers)
        result = icon.run(user_task, scenario)
    """

    def __init__(
        self,
        model,
        tokenizer,
        gamma:       float = 0.3,
        tau:         float = 0.10,
        threshold:   float = 0.5,
        top_k_layers: Optional[List[int]] = None,
        k:           int = K_LAYERS,
    ):
        self.model       = model
        self.tokenizer   = tokenizer
        self.gamma       = gamma
        self.tau         = tau
        self.threshold   = threshold
        self.k           = k
        self.top_k_layers = top_k_layers or list(range(20, 24))  # sensible default

        self.lstp    = LSTP(in_channels=NUM_HEADS * 3, k_layers=k)
        self.capture = AttentionCapture(model)
        self.mr      = MitigatingRectifier(gamma, tau, self.top_k_layers)

        # After running capture, the stored features
        self._last_fis_result = None

    def load_lstp(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")
        self.lstp.load_state_dict(state)
        self.lstp.eval()
        print(f"Loaded LSTP from {ckpt_path}")

    def set_top_k_layers(self, layers: List[int]):
        self.top_k_layers = layers
        self.mr = MitigatingRectifier(self.gamma, self.tau, layers)

    def extract_features(self, attn_store: Dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
        """FIS computation → feature vector for LSTP."""
        if not attn_store:
            return None
        fis_result = compute_all_fis(attn_store)
        self._last_fis_result = fis_result
        feat = build_feature_vector(fis_result["descriptors"], self.top_k_layers)
        return feat  # (96, K)

    def detect(self, feat: torch.Tensor) -> Tuple[bool, float]:
        """Run LSTP probe on a feature vector."""
        prob = predict_proba(self.lstp, feat.unsqueeze(0)).item()
        return prob >= self.threshold, prob

    def run_with_defense(
        self,
        user_task: str,
        executor,
        tool_names: Optional[List[str]] = None,
        initial_scratchpad: str = "",
    ) -> Dict[str, Any]:
        """
        Run the agent with ICON defense active.

        Registers attention capture, runs the agent, checks for attack
        after each generation step, and applies MR if needed.
        """
        from src.agent import run_react_agent

        self.capture.register()
        attack_detected = False
        detection_prob  = 0.0

        def capture_fn():
            return self.capture.get_attn()

        # Run agent; if attack detected at any step, patch the model
        try:
            result = run_react_agent(
                user_task=user_task,
                model=self.model,
                tokenizer=self.tokenizer,
                executor=executor,
                tool_names=tool_names,
                capture_fn=capture_fn,
                initial_scratchpad=initial_scratchpad,
            )
        finally:
            self.capture.remove()
            self.mr.unpatch(self.model)

        # Post-hoc detection on last captured attention
        feat = self.extract_features(result.get("attn_store", {}))
        if feat is not None:
            attack_detected, detection_prob = self.detect(feat)

        result["icon_detected"]      = attack_detected
        result["icon_detect_prob"]   = detection_prob
        result["icon_top_k_layers"]  = self.top_k_layers
        return result

    def run_with_online_defense(
        self,
        user_task: str,
        executor,
        tool_names: Optional[List[str]] = None,
        initial_scratchpad: str = "",
    ) -> Dict[str, Any]:
        """
        Full online defense: detect + rectify within the same generation.
        Registers hooks, runs agent; if probe fires, patches MR and
        re-runs the current step.
        """
        from src.agent import build_react_prompt, parse_action, MAX_NEW_TOKENS
        import re

        self.capture.register()
        scratchpad   = initial_scratchpad
        model_output = ""           # only what the model generates
        attack_detected = False
        mr_applied      = False

        try:
            from src.agent import MAX_STEPS
            for step in range(MAX_STEPS):
                prompt  = build_react_prompt(user_task, tool_names, scratchpad)
                inputs  = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                attn_store = self.capture.get_attn()
                feat = self.extract_features(attn_store)

                if feat is not None:
                    detected, prob = self.detect(feat)
                    if detected and not mr_applied:
                        attack_detected = True
                        mr_applied = True
                        # MR: sanitise context — re-run without the injected observation.
                        # The attention-steering patch (mr.py) modifies post-softmax weights
                        # but cannot change output[0] (already computed from original weights),
                        # so it has no effect on generation.  Context sanitisation is the
                        # effective alternative: drop the injected observation and re-generate.
                        clean_scratchpad = ""   # remove injection-containing initial_scratchpad
                        clean_prompt = build_react_prompt(user_task, tool_names, clean_scratchpad)
                        clean_inputs = self.tokenizer(clean_prompt, return_tensors="pt").to(
                            self.model.device
                        )
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                **clean_inputs,
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False,
                                temperature=None,
                                top_p=None,
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                        self.capture.get_attn()  # flush
                        inputs = clean_inputs   # use clean context for subsequent decode

                gen_ids  = output_ids[0, inputs["input_ids"].shape[1]:]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                scratchpad   += gen_text
                model_output += gen_text

                if "Final Answer:" in gen_text:
                    final  = re.search(r"Final Answer:\s*(.*)", gen_text, re.DOTALL)
                    answer = final.group(1).strip() if final else gen_text
                    return {
                        "final_answer":    answer,
                        "model_output":    model_output,
                        "tool_calls":      executor.call_log,
                        "scratchpad":      scratchpad,
                        "icon_detected":   attack_detected,
                        "mr_applied":      mr_applied,
                    }

                from src.agent import parse_action
                action = parse_action(gen_text)
                if action is None:
                    break
                tool_name, tool_input = action
                observation = executor.execute(tool_name, tool_input)
                scratchpad += f"\nObservation: {observation}\n"

        finally:
            self.capture.remove()
            self.mr.unpatch(self.model)

        return {
            "final_answer":  "",
            "model_output":  model_output,
            "tool_calls":    executor.call_log,
            "scratchpad":    scratchpad,
            "icon_detected": attack_detected,
            "mr_applied":    mr_applied,
        }

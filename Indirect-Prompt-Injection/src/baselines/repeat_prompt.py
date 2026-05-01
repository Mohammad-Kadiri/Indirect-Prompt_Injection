"""
Repeat Prompt baseline — appends the user instruction after each observation
to reinforce the model's original task intent (Debenedetti et al., 2024).
"""

from src.agent import build_react_prompt, parse_action, MAX_NEW_TOKENS, MAX_STEPS
import torch, re
from typing import Dict, Any, Optional, List


def run_repeat_prompt(
    user_task:  str,
    model,
    tokenizer,
    executor,
    tool_names: Optional[List[str]] = None,
    initial_scratchpad: str = "",
) -> Dict[str, Any]:
    scratchpad   = initial_scratchpad
    model_output = ""

    for step in range(MAX_STEPS):
        prompt  = build_react_prompt(user_task, tool_names, scratchpad)
        prompt += f"\n\n[Reminder] Your task is: {user_task}"
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids  = output_ids[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        scratchpad   += gen_text
        model_output += gen_text

        if "Final Answer:" in gen_text:
            final  = re.search(r"Final Answer:\s*(.*)", gen_text, re.DOTALL)
            answer = final.group(1).strip() if final else gen_text
            return {"final_answer": answer, "model_output": model_output,
                    "tool_calls": executor.call_log, "scratchpad": scratchpad}

        action = parse_action(gen_text)
        if action is None:
            break
        tool_name, tool_input = action
        observation = executor.execute(tool_name, tool_input)
        scratchpad += f"\nObservation: {observation}\n"

    return {"final_answer": "", "model_output": model_output,
            "tool_calls": executor.call_log, "scratchpad": scratchpad}

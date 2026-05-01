"""
Delimiting baseline — wraps tool responses in special delimiters to mark
untrusted content boundaries (Willison, 2023).
"""

from src.agent import build_react_prompt, parse_action, MAX_NEW_TOKENS, MAX_STEPS
import torch, re
from typing import Dict, Any, Optional, List


class DelimitingExecutor:
    """Wraps tool output in delimiters before returning."""

    def __init__(self, base_executor, delimiter: str = "<<<TOOL_RESPONSE>>>"):
        self.base = base_executor
        self.delim = delimiter

    def execute(self, tool_name, tool_input):
        raw = self.base.execute(tool_name, tool_input)
        return f"\n{self.delim}\n{raw}\n{self.delim}\n"

    @property
    def call_log(self):
        return self.base.call_log


def run_delimiting(
    user_task:  str,
    model,
    tokenizer,
    executor,
    tool_names: Optional[List[str]] = None,
    initial_scratchpad: str = "",
) -> Dict[str, Any]:
    delim_executor = DelimitingExecutor(executor)
    scratchpad   = initial_scratchpad
    model_output = ""

    for step in range(MAX_STEPS):
        prompt = build_react_prompt(user_task, tool_names, scratchpad)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
                    "tool_calls": delim_executor.call_log, "scratchpad": scratchpad}

        action = parse_action(gen_text)
        if action is None:
            break
        tool_name, tool_input = action
        observation = delim_executor.execute(tool_name, tool_input)
        scratchpad += f"\nObservation: {observation}\n"

    return {"final_answer": "", "model_output": model_output,
            "tool_calls": delim_executor.call_log, "scratchpad": scratchpad}

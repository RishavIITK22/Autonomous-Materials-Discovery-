from llmatdesign.modules.llms import AskLLM

def build_edit_prompt(base_prompt: str, gradient: str, n_edits: int) -> str:
    return f"""
My original prompt:
"{base_prompt}"

Flaw described:
{gradient}

Generate {n_edits} new prompts that fix this flaw.
Wrap each candidate in <PROMPT>...<PROMPT>.
"""

class PromptEditor:
    def __init__(self, llm: AskLLM, config: dict):
        self.llm = llm
        self.edits_per_grad = config['n_edits_per_gradient']

    def edit(self, base_prompt: str, gradient: str) -> list[str]:
        e_prompt = build_edit_prompt(base_prompt, gradient, self.edits_per_grad)
        out = self.llm.ask(e_prompt)
        # parse <PROMPT> tags
        import re
        edits = re.findall(r'<PROMPT>(.*?)<PROMPT>', out, re.DOTALL)
        return [e.strip() for e in edits]

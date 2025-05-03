import random
from llmatdesign.modules.llms import AskLLM
def build_gradient_prompt(base_prompt: str, errors: list[str]) -> str:
    err_str = "\n".join(f"- Input: {x}, True: {y}" for x,y in errors)
    return f"""
My current modification prompt is:
"{base_prompt}"

It made the following errors:
{err_str}

List {len(errors)} specific reasons why the prompt might 
have failed, each wrapped in <START>...<END>."""
class GradientGenerator:
    def __init__(self, llm: AskLLM, config: dict):
        self.llm = llm
        self.batch_size = config['gradient_batch_size']
        self.n_grad = config['n_gradients']

    def get_gradients(self, base_prompt: str, error_list: list[tuple]) -> list[str]:
        # sample up to batch_size error cases
        errs = random.sample(error_list, min(len(error_list), self.batch_size))
        g_prompt = build_gradient_prompt(base_prompt, errs)
        # ask for n_grad reasons in one shot
        out = self.llm.ask(g_prompt)
        # parse each <START>...<END>
        import re
        grads = re.findall(r'<START>(.*?)<END>', out, re.DOTALL)
        return grads[:self.n_grad]
    
    
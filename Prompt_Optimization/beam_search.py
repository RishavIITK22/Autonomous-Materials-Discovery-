from gradient_generator import GradientGenerator
from prompt_editor import PromptEditor
import random

def expand_beam(prompts: list[str], errors: list[tuple], gg: GradientGenerator, pe: PromptEditor) -> list[str]:
    new_cands = []
    for p in prompts:
        grads = gg.get_gradients(p, errors)
        for g in grads:
            edits = pe.edit(p, g)
            new_cands.extend(edits)
    # optional: random paraphrase or shuffle
    return new_cands
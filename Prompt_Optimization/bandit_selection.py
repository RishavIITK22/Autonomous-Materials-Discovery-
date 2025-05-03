import numpy as np

def successive_rejects(scores: dict[str, float], b: int) -> list[str]:
    # keep top-b by score
    sorted_prompts = sorted(scores, key=lambda p: scores[p], reverse=True)
    return sorted_prompts[:b]
# Prompt‑Optimization (ProTeGi) for both modification and reflection
from beam_search import expand_beam
from bandit_selection import successive_rejects
import ast
import ase
import matdeeplearn

def optimize_template(
    agent,
    base_prompt: str,
    error_collector: callable,
    expand_fn: callable,
    score_fn: callable,
    config: dict,
    extra_context: dict = None
) -> list[str]:
    """
    Generic PTGI optimizer for a single prompt template.
    - error_collector(prompt) -> list of error cases
    - expand_fn(prompts, errors) -> list of new prompt candidates
    - score_fn(prompt, errors) -> float score
    - extra_context: optional shared state (e.g. best mod prompt)
    Returns the final beam of optimized prompts.
    """
    beam = [base_prompt]
    for _ in range(config['n_iterations']):
        scores = {}
        errors_map = {}
        # evaluate each candidate in the beam
        for p in beam:
            errs = error_collector(p, extra_context)
            errors_map[p] = errs
            scores[p] = score_fn(p, errs)
        # select top-k prompts
        beam = successive_rejects(scores, config['beam_width'])
        # expand each selected prompt
        new_beam = []
        for p in beam:
            errs = errors_map[p]
            new_beam.extend(expand_fn([p], errs))
        beam = new_beam
    return beam


def run_ptgi(agent, formula, target, config):
    # load base templates
    from llmatdesign.prompts.gpt import base_template_bandgap, base_template_reflection
    from llmatdesign.prompts.utils import format_prompt, get_action, get_reflection_prompt_A, get_reflection
    from gradient_generator import GradientGenerator
    from prompt_editor import PromptEditor

    # instantiate gradient & editor once
    gg = GradientGenerator(agent.llm, config)
    pe = PromptEditor(agent.llm, config)

    # ----- OPTIMIZE MODIFICATION PROMPT -----
    def collect_mod_errors(prompt_tpl, _ctx=None):
        errs = []
        for f in config.get('validation_set', []):
            ok, struct = agent.query_materials_project(f, 'structure')
            ok2, init_gap = agent.query_materials_project(f, 'band_gap')
            if not ok or not ok2:
                continue
            single = format_prompt(
                prompt_tpl,
                [None], [struct], [init_gap], [None],
                property_type='band_gap',
                target_property=target
            )
            try:
                out = get_action(agent.llm, single)
                mod = ast.literal_eval(out)
                new_structure, new_gap, band_gap_uncertainty, real_structure= agent.perform_modification(
                    struct, mod['Modification'], calculation_type='band_gap'
                )
                if not agent.is_within_threshold(new_gap, target):
                    errs.append((f"{f}:{init_gap:.2f}", target))
            except:
                errs.append((f"{f}:{init_gap:.2f}", target))
        return errs

    optimized_mods = optimize_template(
        agent,
        base_template_bandgap,
        collect_mod_errors,
        lambda ps, errs: expand_beam(ps, errs, gg, pe),
        lambda p, errs: -len(errs),
        config
    )
    best_mod = optimized_mods[0]

    # ----- OPTIMIZE REFLECTION PROMPT -----
    def collect_ref_errors(prompt_tpl, ctx):
        errs = []
        # ctx passes best_mod
        for f in config.get('validation_set', []):
            ok, struct = agent.query_materials_project(f, 'structure')
            ok2, init_gap = agent.query_materials_project(f, 'band_gap')
            if not ok or not ok2:
                continue
            # generate one modification using best_mod
            single = format_prompt(
                best_mod,
                [None], [struct], [init_gap], [None],
                property_type='band_gap',
                target_property=target
            )
            try:
                out = get_action(agent.llm, single)
                mod = ast.literal_eval(out)
                new_struct, new_gap, band_gap_uncertainty, real_structure = agent.perform_modification(
                    struct, mod['Modification'], calculation_type='band_gap'
                )
                # build reflection prompt input
                prev_val = init_gap
                new_val = new_gap
                r_text = get_reflection_prompt_A(
                     struct.get_chemical_formula('metal'),
                     new_struct.get_chemical_formula('metal'), 
                     out, 
                     target, 
                     prev_val,
                     new_val)
                # ask reflection
                refl = get_reflection(agent.llm, r_text)
                # basic check: must use "successful" or "failure"
                if 'success' not in refl.lower() and 'failure' not in refl.lower():
                    errs.append((f"{f}:{new_val:.2f}", refl))
            except:
                errs.append((f"{f}", ''))
        return errs

    optimized_refs = optimize_template(
        agent,
        base_template_reflection,
        collect_ref_errors,
        lambda ps, errs: expand_beam(ps, errs, gg, pe),
        lambda p, errs: -len(errs),
        config,
        extra_context={'best_mod':best_mod}
    )

    return optimized_mods, optimized_refs
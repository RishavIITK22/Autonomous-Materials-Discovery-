import yaml
import argparse
from llmatdesign.core.agent import Agent
from llmatdesign.modules.llms import AskLLM
from optimizer import run_ptgi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Prompt_Optimization\ptgi_config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # set up LLM and Agent (example with GPT-4)
    llm = AskLLM('Llama-3.3-70B-Instruct', api_key=cfg.get('api_key'), base_url="https://cloud.olakrutrim.com/v1")
    agent = Agent(llm,
                  save_path=cfg.get('save_path'),
                  forcefield_config_path=cfg.get('forcefield_config_path'),
                  bandgap_config_path=cfg.get('bandgap_config_path'),
                  formation_energy_config_path=cfg.get('formation_energy_config_path'),
                  mp_api_key=cfg.get('mp_api_key'))

    beam = run_ptgi(agent,
                    cfg['material_formula'],
                    cfg['target_value'],
                    cfg)
    print("Optimized prompts:")
    for p in beam:
        print(p)

if __name__ == '__main__':
    main()

# Autonomous-Materials-Discovery-
This repo showcases the of developing of a  autonomous materials discovery framework based web app , leveraging the power of LLMs.
This work is inspired and referenced originally from the following two research works:
   -> https://github.com/Fung-Lab/MatDeepLearn_dev.git ( paper- https://doi.org/10.1038/s41524-021-00554-0)
   -> https://github.com/Fung-Lab/LLMatDesign.git (paper- https://arxiv.org/abs/2406.13163)

This work attempts to develop a web interface that enable users to input a starting material and a target band gap value, selecting LLM engine and the corresponding api key and the underlying algorithm will discover a material with given target band gap value, within some maximum iterations, as set by user.

To set up this locally(preferred), in order to further develop, follow the given steps:
  1. Follow all steps given in the repo: https://github.com/RishavIITK22/MatDeepLearn_dev.git, and prepare the required python environment
  2. Next git clone https://github.com/RishavIITK22/LLMatDesign.git
      ->cd LLMatDesign
      ->pip install -r requirements.txt
      ->pip install -e .
  4. Next run the web_app.py or web_app_updated.py in the .notebooks/ directory as mentioned in this repo to run the web app script.

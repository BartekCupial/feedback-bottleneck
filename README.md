# Feedback-bottleneck
This codebase is reference implementation of the paper [What Drives Interactive Improvement from Feedback?](https://arxiv.org/abs/2606.30774). See also our [project page](https://j-lojek.github.io/feedback-generation-is-a-bottleneck/).



## Installation
Clone and install:
```bash
git clone git@github.com:BartekCupial/feedback-bottleneck.git
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Running

Each environment has a ready-to-run script under `scripts/`. Activate the
virtualenv and launch the one you want:

```bash
source .venv/bin/activate

bash scripts/run_math.sh      # Omni-MATH
bash scripts/run_code.sh      # Codeforces
bash scripts/run_bbeh.sh      # Linguini (BBEH)
bash scripts/run_arc_agi.sh   # ARC-AGI
```

Shared settings (model, sampling, agent) live in `scripts/_common.sh`; per-environment
settings live in each `run_*.sh`. Edit those files to change a run, or point at a
different model on the fly with `MODEL_ID=...`:

```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct bash scripts/run_math.sh
```

## Code Structure
```text
feedback_bottleneck/
  config/                    CLI / configuration helpers.
  envs/                      Environment implementations and registrations.
    arc_agi/                 ARC-AGI grid-transformation environment.
    bbeh/                    BIG-Bench Extra Hard (Linguini) environment.
    countdown/               Countdown task environment.
    crafter/                 Crafter environment.
    enterprise_ops           Enterprise Ops environment.
    nle/                     NetHack Learning Environment wrapper.
    omni_code/               Codeforces competitive-programming environment.
    omni_math/               Omni-MATH mathematical reasoning environment.
    sciknoweval/             SciKnowEval science reasoning environment.
    wrappers/                Shared Gymnasium wrappers (gym_compatibility, video_recorder).
  llm/                       LLM interaction layer.
    actor/                   Actor implementations: client, dummy, HF, and LLM actors.
    agent/                   Per-environment agent logic (arc_agi, bbeh, code, math) plus evaluator,
                             guidance, and vllm config patch.
    utils/                   Shared utilities: math eval metrics, metrics tracker,
                             model inputs, typing helpers, W&B utilities.
  dataset_env.py             Dataset environment entry point.
  dataset_utils.py           Dataset construction helpers.
  env_wrapper.py             Top-level environment wrapper.
  spaces.py                  Observation / action space definitions.
  utils.py                   Shared project-level utilities.
scripts/                     Per-environment run scripts (run_*.sh, _common.sh)
                             and the collect_episodes entry point.
```

## Citation

If you use **Feedback-bottleneck**, please cite the project:

```bibtex
@misc{cupial2026what,
  title         = {What Drives Interactive Improvement from Feedback?},
  author        = {Bart{\l}omiej Cupia{\l} and Jan {\L}ojek and Miko{\l}aj Garstecki and Szymon Pob{\l}ocki and Alicja Ziarko and Piotr Mi{\l}o{\'s}},
  journal={arXiv preprint arXiv:2606.30774},
  year={2026}
}
```

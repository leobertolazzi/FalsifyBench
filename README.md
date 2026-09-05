<h1 align="center">FALSIFYBENCH: Evaluating Inductive Reasoning in LLMs with Rule Discovery Games</h1>

This repository contains the code for the paper: Leonardo Bertolazzi, Katya Tentori, Raffaella Bernardi (2026). [*FALSIFYBENCH: Evaluating Inductive Reasoning in LLMs with Rule Discovery Games*](https://arxiv.org/abs/2606.04751).

> **Abstract:** Large language models (LLMs) are increasingly deployed as autonomous agents in scientific tasks. Yet whether these systems can effectively engage in forms of inductive reasoning relevant to scientific discovery remains an open question. In this work, we introduce FALSIFYBENCH, an evaluation framework for hypothesis-driven reasoning inspired by the classic Wason 2-4-6 task, in which agents must discover hidden semantic properties by iteratively proposing examples and receiving feedback. This task captures key elements of scientific reasoning: hypothesis generation, evidence gathering, and belief revision in response to both confirming and disconfirming evidence. Our evaluation of 12 LLMs across model families and scales shows that reasoning models are generally stronger scientific reasoners than instruction-tuned models, although no model comes close to optimal performance. The primary driver of success is the capacity for negative testing: models that actively seek to falsify their hypotheses consistently outperform those that primarily seek confirmation. Moreover, a fine-grained turn-level analysis, neglected in previous work, reveals that failure is tied to identifiable patterns in how models navigate the hypothesis space.

## Table of Contents
- [Setup](#setup)
- [Repository structure](#repository-structure)
- [Data](#data)
- [Run the benchmark](#run-the-benchmark)
- [Reproduce the paper analyses](#reproduce-the-paper-analyses)
- [Turn-level relation annotation](#turn-level-relation-annotation)
- [Tests](#tests)  
- [License](#license)
- [Citation](#citation)

## Setup

Python 3.11 or later and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync
cp .env.example .env
```

Add credentials only for the providers you use:

- `OPENAI_API_KEY`: OpenAI hosted GPT models
- `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`: OpenAI hosted GPT models
- `TOGETHER_API_KEY`: all other models evaluated in the paper

## Repository structure

- `raw_data/`: source WordNet triples.
- `data/`: processed games and the curated 100 games used in the benchmark.
- `src/benchmark/`: dataset construction, game engine, and evaluation.
- `src/annotation/`: oracle-validation and hypothesis-target relation annotation.
- `src/analysis/`: mixed-effects regression, error-analysis, and plots.
- `src/scripts/`: deterministic benchmark-construction script.
- `results/`: saved game records, annotations, tables, and paper figures.
- `tests/`: unit and regression tests.

## Data

The paper evaluates a curated set of 100 games with the target rules `animal`, `artifact`, `body part`, `food`, and `plant`. Sampling categories form two groups based on their taxonomic distance from the target:

- `close`: at most four WordNet edges from the target;
- `deep`: more than four edges from the target.

The ready-to-use benchmark is `data/curated_games.json`, with 50 games from each distance group. To reconstruct the processed and curated datasets:

```bash
uv run python -m src.main --mode prepare
uv run python -m src.scripts.build_curated_games --per-group 50 --seed 42
```

## Run the benchmark

Run one model as both player and oracle on the curated set of games:

```bash
uv run python -m src.main --mode play --curated --player-model gpt-5-mini
```

Use `--resume` to continue an interrupted run. The 12 identifiers used in the paper are:

```text
gpt-5.2-chat
gpt-5-mini
gpt-5-nano
openai/gpt-oss-120b
openai/gpt-oss-20b
deepseek-ai/DeepSeek-V3.1
Qwen/Qwen3.5-9B
MiniMaxAI/MiniMax-M2.5
meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
mistralai/Mistral-Small-24B-Instruct-2501
moonshotai/Kimi-K2.5
zai-org/GLM-5
```

Note that API generation is inherently nondeterministic. The fixed game set, prompts, provider settings, retry behavior, and saved records are included for reproducibility.

## Reproduce the paper analyses

The repository includes all saved game records, so no API calls are needed to reproduce the experiments and figures from the paper.

```bash
# Per-game and aggregate metrics
uv run python -m src.benchmark.evaluation --results-dir results

# Turn-level relation and linguistic failure analyses
uv run python -m src.analysis.failure --results-dir results

# Oracle accuracy and Cohen’s kappa against human annotations
uv run python -m src.annotation.agreement \
  --input-dir results/annotation_pairs_oracle \
  --output results/annotation_pairs_oracle/agreement_summary.json

# Game-level mixed-effects logistic regression
uv run python -m src.analysis.mixed_effects \
  --results-dir results \
  --annotation-dir results/annotation_pairs_oracle

# Figures
uv run python -m src.annotation.plot_agreement \
  --input-dir results/annotation_pairs_oracle \
  --summary-path results/annotation_pairs_oracle/agreement_summary.json
uv run python -m src.analysis.plots --results-dir results
```

The resulting figures are written to `results/plots/`, `results/annotation_plots/`, and `results/mixed_effect_analysis/`.

## Turn-level relation annotation

GPT-5-Mini annotations of the set relation between each current hypothesis and target rule can be regenerated with:

```bash
uv run python -m src.annotation.annotate_relations \
  --input results \
  --model gpt-5-mini \
  --progress
```

## Tests

```bash
uv run --extra dev pytest -q
```

## License

[![CC-BY-SA](https://img.shields.io/badge/License-Creative%20Commons%20Attribution--ShareAlike%204.0%20International%20Public%20License-green.svg)](https://creativecommons.org/licenses/by-sa/4.0)

This work is licensed under a [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation

```bibtex
@misc{bertolazzi2026falsifybench,
  title         = {FALSIFYBENCH: Evaluating Inductive Reasoning in LLMs with Rule Discovery Games},
  author        = {Leonardo Bertolazzi and Katya Tentori and Raffaella Bernardi},
  year          = {2026},
  eprint        = {2606.04751},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.04751}
}
```



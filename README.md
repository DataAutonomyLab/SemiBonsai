# SemiBonsai (revised reproducible version)

This repository contains a cleaned, runnable version of SemiBonsai for numerical question answering over semi-structured tables. It includes source code and benchmark data, but excludes generated results, caches, logs, and credentials.

The pipeline consists of a table structurer, uncertainty resolver, numerical reasoner, and evaluator.

## Requirements

- macOS or Linux
- Python **3.11** (recommended; several optional ML packages do not support Python 3.13)
- An OpenAI-compatible API key for VLM/LLM stages
- `wkhtmltoimage` only when HTML-to-image conversion is needed by `imgkit`

CUDA and `cupy-cuda12x` are not required for the core pipeline.

## Installation

```bash
git clone git@github.com:JrJessyLuo/SemiBonsai-revised.git
cd SemiBonsai-revised
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## API configuration

Never place an API key in a tracked file. Set it in the current terminal:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LLM_PROVIDER="openai"
source bash/environment.sh
```

For another OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="your-provider-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"
export VLM_API_URL="$OPENAI_BASE_URL"
export VLM_API_KEY="$OPENAI_API_KEY"
source bash/environment.sh
```

`bash/environment.sh` resolves the repository root automatically and contains no credentials. `codes/config.yaml` is configured for the included `datasets/` directory and an ignored `result/` directory.

## Quick smoke test

Run every command from the directory shown. The first two steps call the configured VLM.

```bash
cd codes/table_structurer
python vlm_identification.py --dataset test_dataset
python vlm_identification.py --dataset test_dataset --multiple
python convert_table_structural_model.py --dataset test_dataset

cd ../uncertainty_resolver
python question_rewriting.py --dataset test_dataset --model GPT-4.1

cd ../reasoner
python query_plan.py \
  --dataset test_dataset \
  --model GPT-4.1 \
  --rewrite_model GPT-4.1

cd ../utils
python evaluate_utils.py --dataset test_dataset --model GPT-4.1
```

Generated files are written beneath `result/` and are not tracked.

## Running HiTab-Num

```bash
cd codes/table_structurer
python vlm_identification.py --dataset hitab_num
python vlm_identification.py --dataset hitab_num --multiple
python convert_table_structural_model.py --dataset hitab_num

cd ../uncertainty_resolver
python question_rewriting.py --dataset hitab_num --model GPT-4.1

cd ../reasoner
python query_plan.py \
  --dataset hitab_num \
  --model GPT-4.1 \
  --rewrite_model GPT-4.1

cd ../utils
python evaluate_utils.py --dataset hitab_num --model GPT-4.1
```

For a targeted reasoner run, use comma-separated QA IDs:

```bash
python query_plan.py \
  --dataset hitab_num \
  --model GPT-4.1 \
  --rewrite_model GPT-4.1 \
  --qa_rids 29d75fddfb7895be35e1e80ae7739b9b
```

## Repository layout

```text
codes/table_structurer/       table identification and structural conversion
codes/uncertainty_resolver/  ambiguity detection, pruning, and rewriting
codes/reasoner/              schema linking, query planning, and execution
codes/router/                optional LLM routing experiments
codes/utils/                 API, table, and evaluation utilities
datasets/                    included benchmark and example data
bash/environment.sh          credential-free environment setup
result/                      generated locally; never committed
```

## Reproducibility notes

- Model outputs can vary. Record model names and command-line arguments with experiment reports.
- The evaluator accepts fraction/percentage equivalents such as `0.304` and `30.4`.
- Some original `range` questions contain scalarized labels that appear to retain one endpoint; treat those records carefully.
- The revised converter preserves repeated hierarchical rows by full paths, such as `basic research | higher education`.

## Security and privacy

- Do not commit `.env`, API keys, local environment scripts, generated prompts, logs, or results.
- Run a secret scan before publishing.
- If a credential has ever been exposed, revoke it at the provider and create a new one.

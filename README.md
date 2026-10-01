# Cost-Effective Numerical QA over Semi-Structured Table: Structuring, Resolution, Planning

Accepted at **IEEE ICDE 2027**.

This repository provides the implementation and datasets for **SemiBonsai**, a numerical question-answering framework for semi-structured tables. SemiBonsai organizes the workflow into three principal stages: table structuring, uncertainty resolution, and query planning.

## Datasets

The repository includes the following numerical table question-answering benchmarks under `datasets/`:

- **HiTab-Num** (`hitab_num`)
- **MultiHiertt-Num** (`multihiertt_num`)
- **RealHiTBench-Num** (`realhitbench_num`)
- **Complex subsets** (`complex_subsets`), containing the Complex-Q, Complex-T, and Complex-QT evaluation subsets
- **Test dataset** (`test_dataset`), a compact example intended for pipeline verification

Each benchmark directory contains the question-answer records and source tables required by the corresponding experiment. Generated predictions and evaluation outputs are intentionally excluded from the repository.

## Baselines

The `codes/baselines/` directory contains a direct GPT baseline that answers each question from the original table without applying the SemiBonsai structuring, rewriting, or planning stages.

```bash
cd codes/baselines
python gpt.py --dataset hitab_num --model gpt-4.1
```

Baseline predictions are written to `result/<dataset>/baselines/<model>/predictions.jsonl`. Published performance summaries are not included because they are generated experimental results rather than source artifacts.

## File Overview

1. `codes/table_structurer/vlm_identification.py`: Identifies hierarchical table structure and subtable boundaries using a vision-language model.
2. `codes/table_structurer/convert_table_structural_model.py`: Converts identified table structure into the layered structural representation used by downstream modules.
3. `codes/uncertainty_resolver/uncertainty_detection.py`: Detects under-specified expressions and associates them with candidate table headers.
4. `codes/uncertainty_resolver/pruning.py`: Prunes incompatible grounding candidates using the layered table structure.
5. `codes/uncertainty_resolver/question_rewriting.py`: Produces grounded question rewrites for ambiguous questions.
6. `codes/reasoner/query_plan.py`: Performs decomposition, schema linking, query planning, execution, and answer selection.
7. `codes/reasoner/operation.py`: Implements numerical and table operations used by generated query plans.
8. `codes/router/llm_routing.py`: Provides the optional routing logic for selecting an answering path.
9. `codes/baselines/gpt.py`: Implements the direct GPT baseline over the original tables.
10. `codes/utils/evaluate_utils.py`: Evaluates generated answers against benchmark labels.
11. `codes/utils/`: Contains shared API, table-processing, prompt, and data utilities.
12. `codes/config.yaml`: Defines dataset locations, output locations, and pipeline configuration.
13. `bash/environment.sh`: Initializes repository-relative paths and model-service settings without storing credentials.
14. `datasets/`: Contains the benchmark questions, labels, and source tables.

## Installation

### Requirements

- **Python 3.11**
- **macOS or Linux**
- An **OpenAI API key**, or access to an OpenAI-compatible endpoint
- `wkhtmltoimage` when HTML-to-image conversion is required by `imgkit`

CUDA and `cupy-cuda12x` are not required for the core SemiBonsai pipeline.

### Setup

Clone the repository and create an isolated Python environment:

```bash
git clone git@github.com:JrJessyLuo/SemiBonsai-revised.git
cd SemiBonsai-revised

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Setup & Configuration

### 1. API Configuration

Credentials must be supplied through environment variables and must not be written into tracked files:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export LLM_PROVIDER="openai"
source bash/environment.sh
```

For an OpenAI-compatible service, additionally configure its base URL:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://your-provider.example/v1"
export VLM_API_KEY="$OPENAI_API_KEY"
export VLM_API_URL="$OPENAI_BASE_URL"
source bash/environment.sh
```

### 2. Repository Structure

The pipeline expects the following project layout:

```text
.
├── bash/
│   └── environment.sh
├── codes/
│   ├── baselines/
│   ├── reasoner/
│   ├── router/
│   ├── table_structurer/
│   ├── uncertainty_resolver/
│   ├── utils/
│   └── config.yaml
├── datasets/
│   ├── hitab_num/
│   ├── multihiertt_num/
│   ├── realhitbench_num/
│   ├── complex_subsets/
│   └── test_dataset/
└── requirements.txt
```

`codes/config.yaml` uses repository-relative dataset and output paths. By default, generated artifacts are stored under `result/`, which is excluded from version control.

### 3. Dataset Selection

Commands use `--dataset <dataset_name>`. Supported primary dataset names are:

- `test_dataset`
- `hitab_num`
- `multihiertt_num`
- `realhitbench_num`

## Usage

The following commands execute the complete SemiBonsai pipeline. Replace `test_dataset` with another supported dataset name for a full benchmark run.

### 1. Table Structuring

Identify the table structure, identify multiple subtables when present, and convert the outputs into the layered structural model:

```bash
cd codes/table_structurer

python vlm_identification.py --dataset test_dataset
python vlm_identification.py --dataset test_dataset --multiple
python convert_table_structural_model.py --dataset test_dataset
```

### 2. Uncertainty Resolution

Detect ambiguous expressions, ground them to the table schema, prune incompatible candidates, and generate rewritten questions:

```bash
cd ../uncertainty_resolver

python question_rewriting.py \
  --dataset test_dataset \
  --model GPT-4.1
```

### 3. Query Planning and Execution

Generate and execute query plans for the rewritten questions:

```bash
cd ../reasoner

python query_plan.py \
  --dataset test_dataset \
  --model GPT-4.1 \
  --rewrite_model GPT-4.1
```

To run a selected set of questions, provide comma-separated question identifiers:

```bash
python query_plan.py \
  --dataset hitab_num \
  --model GPT-4.1 \
  --rewrite_model GPT-4.1 \
  --qa_rids <QUESTION_ID_1>,<QUESTION_ID_2>
```

### 4. Evaluation

Evaluate the generated numerical answers:

```bash
cd ../utils

python evaluate_utils.py \
  --dataset test_dataset \
  --model GPT-4.1
```

An optional LLM-based fallback judge can be enabled for answer pairs that cannot be resolved by deterministic normalization rules:

```bash
python evaluate_utils.py \
  --dataset test_dataset \
  --model GPT-4.1 \
  --use-llm-fallback \
  --judge-model <JUDGE_MODEL>
```

## Output

All generated table metadata, rewrites, query plans, predictions, and evaluation files are written beneath `result/`. This directory is ignored by Git so that the repository contains only source code, configuration, and benchmark data.

Model-based stages may produce different outputs across models or repeated runs. For reproducibility, report the exact model identifiers, command-line arguments, and service configuration used for each experiment.

## Security

Do not commit API keys, `.env` files, personal filesystem paths, generated prompts, logs, or result files. If a credential has previously been exposed, revoke it and issue a replacement before running the code.

## Contact

For questions or issues, contact [feng.luo@student.rmit.edu.au](mailto:feng.luo@student.rmit.edu.au).

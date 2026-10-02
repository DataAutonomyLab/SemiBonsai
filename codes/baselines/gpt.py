"""Direct GPT baseline for numerical QA over the original tables."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CODES_ROOT = REPOSITORY_ROOT / "codes"
os.environ.setdefault("SemiBonsai_BASE_DIR", str(REPOSITORY_ROOT))
if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from utils.api_utils import llm_generate_setup  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct GPT baseline over the original benchmark tables."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name under datasets/.")
    parser.add_argument("--model", default="gpt-4.1", help="Model identifier.")
    parser.add_argument(
        "--qa_rids",
        default=None,
        help="Optional comma-separated question IDs for a targeted run.",
    )
    parser.add_argument("--output", default=None, help="Optional output JSONL path.")
    return parser.parse_args()


def normalize_table_id(value: Any) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"The direct baseline requires one table, received: {value}")
        value = value[0]
    return str(value)


def load_workbook_as_text(path: Path) -> str:
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    rendered = []
    for sheet_name, frame in sheets.items():
        frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        rendered.append(f"Sheet: {sheet_name}\n{frame.to_csv(index=False, header=False)}")
    return "\n\n".join(rendered)


def build_prompt(question: str, table_text: str) -> str:
    return (
        "Answer the numerical question using only the supplied table. "
        "Perform any necessary arithmetic and return only the final answer, "
        "without an explanation or unit.\n\n"
        f"Table:\n{table_text}\n\nQuestion: {question}\nAnswer:"
    )


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def main() -> None:
    args = parse_args()
    dataset_dir = REPOSITORY_ROOT / "datasets" / args.dataset / "data"
    qa_path = dataset_dir / "single_tab_qa.jsonl"
    table_dir = dataset_dir / "table"

    if not qa_path.exists():
        raise FileNotFoundError(f"Question file not found: {qa_path}")
    if not table_dir.exists():
        raise FileNotFoundError(f"Table directory not found: {table_dir}")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running the baseline.")

    selected_ids = None
    if args.qa_rids:
        selected_ids = {value.strip() for value in args.qa_rids.split(",") if value.strip()}

    with qa_path.open("r", encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle if line.strip()]

    if selected_ids is not None:
        questions = [
            item
            for item in questions
            if str(item.get("qa_id") or item.get("question_id") or item.get("id"))
            in selected_ids
        ]

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else REPOSITORY_ROOT
        / "result"
        / args.dataset
        / "baselines"
        / safe_model_name(args.model)
        / "predictions.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_handle:
        for item in tqdm(questions, desc="Direct GPT baseline"):
            question_id = item.get("qa_id") or item.get("question_id") or item.get("id")
            question = item.get("query") or item.get("question")
            table_id = normalize_table_id(item.get("table_id"))
            table_path = table_dir / f"{table_id}.xlsx"
            if not table_path.exists():
                raise FileNotFoundError(f"Table file not found: {table_path}")

            table_text = load_workbook_as_text(table_path)
            started = time.time()
            response = llm_generate_setup(
                build_prompt(str(question), table_text),
                args.model,
                temperature=0.0,
                json_format=False,
            )
            record = {
                "question_id": question_id,
                "table_id": table_id,
                "question": question,
                "label": item.get("label"),
                "final_answer": response["text"].strip(),
                "time_cost": {"total": time.time() - started},
                "token_cost": {
                    "input": response["input_tokens"],
                    "output": response["output_tokens"],
                    "total": response.get(
                        "total_tokens",
                        response["input_tokens"] + response["output_tokens"],
                    ),
                },
            }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()

    print(f"[DONE] Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()

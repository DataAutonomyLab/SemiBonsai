#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import yaml

# Support the README invocation: `cd codes/utils && python evaluate_utils.py ...`.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.dirname(SCRIPT_DIR)
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

from utils.api_utils import llm_generate
from utils.constants import MODEL_MAP


# ============================================================
# 0) Prompt (LLM fallback)
# ============================================================
evaluation_prompt_num = """
You are a strict numeric comparator. Output ONLY "T" or "F".

Input:
A: {a}
B: {b}

Procedure:
1) Extract the first numeric value from A and B. If either is missing, output F.
   - Ignore surrounding text, commas, currency symbols, and percent signs.

2) Unit handling (optional):
   - If both contain a recognizable unit, convert to the same base unit BEFORE comparing.
   - If units are incompatible (different physical dimension), output F.
   - If one or both have no unit, compare as plain numbers.

   Supported conversions (examples, not exhaustive):
   - percent: X% = X * 0.01
   - thousand/million/billion/trillion: *1e3 / *1e6 / *1e9 / *1e12
   - 千/万/亿: *1e3 / *1e4 / *1e8
   - length: mm/cm/m/km -> *1e-3 / *1e-2 / *1 / *1e3
   - mass: mg/g/kg -> *1e-6 / *1e-3 / *1
   - time: ms/s/min/h -> *1e-3 / *1 / *60 / *3600

3) Let nA, nB be the normalized numbers.
   Output T if:
     |nA - nB| <= max(1e-6, 0.02 * max(|nA|, |nB|, 1.0))

4) If not equal by (3), allow common scale-factor mismatches:
   If nB/nA (or nA/nB) is within 2% of any factor in:
     {{0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e6, 1e8, 1e9, 1e12}}
   then output T.

Otherwise output F.
""".strip()


# ============================================================
# 1) Rule-based fair numeric compare (from your snippet, bug-fixed)
# ============================================================

def normalize_num_str(s: Any) -> Optional[float]:
    if s is None:
        return None

    s = str(s).strip()
    if not s:
        return None

    is_percent = "%" in s
    s = s.replace("%", "")

    # Keep only digits, separators, sign, exponent
    s = re.sub(r"[^0-9.,eE+\-]", "", s)
    if not s:
        return None

    # Decide decimal vs thousand separators
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            # comma = decimal, dots = thousands
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            # dot = decimal, commas = thousands
            s = s.replace(",", "")
    elif "," in s:
        # only comma -> treat as decimal
        s = s.replace(",", ".")

    try:
        val = float(Decimal(s))
    except (InvalidOperation, ValueError):
        return None

    if is_percent:
        val /= 100.0
    return val


def nearly_equal(
    a: float,
    b: float,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-8,
    allow_opposite_sign: bool = True,
) -> bool:
    # Standard near-equality
    diff = abs(a - b)
    scale = max(abs(a), abs(b), 1.0)
    if diff <= max(abs_tol, rel_tol * scale):
        return True

    # Optional: magnitude-equality even if sign differs
    if allow_opposite_sign:
        mag_diff = abs(abs(a) - abs(b))
        mag_scale = max(abs(a), abs(b), 1.0)
        if mag_diff <= max(abs_tol, rel_tol * mag_scale):
            return True

    return False


def _digits_signature(s: Any) -> str:
    ds = re.sub(r"\D", "", str(s))
    ds = ds.lstrip("0") or "0"
    return ds


def fair_compare_num_str(
    sa: Any,
    sb: Any,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-8,
    max_exp: int = 3,
) -> Tuple[bool, Optional[float], Optional[float], Optional[int]]:
    """
    Fair comparison between two numeric strings/values, allowing:
      - tolerance-based equality
      - percent/fraction (x100)
      - 10^k scaling for |k| <= max_exp, only if digit-signature matches

    Returns: (is_equal, va, vb, used_exp)
      used_exp:
        0 for direct
        +/-2 for percent-style (x100)
        k for 10^k scaling
        None for not equal
    """
    va = normalize_num_str(sa)
    vb = normalize_num_str(sb)
    if va is None or vb is None:
        return False, va, vb, None

    # 1) direct
    if nearly_equal(va, vb, rel_tol=rel_tol, abs_tol=abs_tol):
        return True, va, vb, 0

    # 2) percent/fraction (x100), no digit-signature constraint
    if nearly_equal(va * 100.0, vb, rel_tol=rel_tol, abs_tol=abs_tol):
        return True, va, vb, +2
    if nearly_equal(vb * 100.0, va, rel_tol=rel_tol, abs_tol=abs_tol):
        return True, va, vb, -2

    # 3) general 10^k scaling but only if digit patterns match
    sig_a = _digits_signature(sa)
    sig_b = _digits_signature(sb)
    if sig_a != sig_b:
        return False, va, vb, None

    for k in range(-max_exp, max_exp + 1):
        if k == 0:
            continue
        factor = 10.0 ** k
        if nearly_equal(va * factor, vb, rel_tol=rel_tol, abs_tol=abs_tol):
            return True, va, vb, k
        if nearly_equal(vb * factor, va, rel_tol=rel_tol, abs_tol=abs_tol):
            return True, va, vb, -k

    return False, va, vb, None


# ============================================================
# 2) LLM fallback comparator (your required logic)
# ============================================================

LLMGenerateFn = Callable[..., Any]  # expected signature: llm_generate(prompt, model=..., **kwargs)

def _normalize_llm_result(x: Any) -> str:
    """
    Your llm_generate sometimes returns dict with 'text'.
    Normalize to "T"/"F".
    """
    if isinstance(x, dict) and "text" in x:
        x = x["text"]
    s = str(x).strip().upper()
    return "T" if s.startswith("T") else "F"


def is_equal_num(
    a: Any,
    b: Any,
    *,
    llm_generate: Optional[LLMGenerateFn],
    llm_model: str = "gpt-5.1",
) -> str:
    """
    Two-stage:
      1) fair_compare_num_str
      2) LLM fallback if not matched
    Returns: "T" or "F"
    """
    is_correct, _, _, _ = fair_compare_num_str(a, b)
    if is_correct:
        return "T"

    if llm_generate is None:
        # If user didn't provide LLM, we cannot fallback
        return "F"

    prompt = evaluation_prompt_num.format(a=str(a), b=str(b))
    out = llm_generate(prompt, model=llm_model)
    return _normalize_llm_result(out)


# ============================================================
# 3) Public API: evaluate one pair
# ============================================================

@dataclass
class EvalResult:
    is_correct: bool
    method: str                 # "rule" or "llm"
    pred_val: Optional[float]
    gt_val: Optional[float]
    used_exp: Optional[int]
    llm_judge: Optional[str]    # "T"/"F"/None


def evaluate_pair(
    prediction: Any,
    ground_truth: Any,
    *,
    llm_generate: Optional[LLMGenerateFn] = None,
    llm_model: str = "gpt-5.1",
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-8,
    max_exp: int = 3,
    use_llm_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate one (prediction, ground_truth) pair.
    Returns a JSON-serializable dict with:
      is_correct, method, pred_val, gt_val, used_exp, llm_judge
    """
    is_correct, va, vb, used_exp = fair_compare_num_str(
        prediction, ground_truth, rel_tol=rel_tol, abs_tol=abs_tol, max_exp=max_exp
    )
    if is_correct:
        r = EvalResult(
            is_correct=True, method="rule",
            pred_val=va, gt_val=vb, used_exp=used_exp,
            llm_judge=None
        )
        return r.__dict__

    if not use_llm_fallback:
        r = EvalResult(
            is_correct=False, method="rule",
            pred_val=va, gt_val=vb, used_exp=used_exp,
            llm_judge=None
        )
        return r.__dict__

    judge = is_equal_num(
        prediction, ground_truth, llm_generate=llm_generate, llm_model=llm_model
    )
    r = EvalResult(
        is_correct=(judge == "T"),
        method="llm",
        pred_val=va,
        gt_val=vb,
        used_exp=used_exp,
        llm_judge=judge
    )
    return r.__dict__


# ============================================================
# 4) Optional: batch evaluate JSONL (resume-safe)
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(output_jsonl: str, id_field: str) -> set:
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if id_field in obj:
                    done.add(str(obj[id_field]))
            except Exception:
                continue
    return done



# ============================================================
# 5) CLI
# ============================================================
def _load_prediction(result_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(result_path):
        return None

    try:
        payload = json.load(open(result_path, "r", encoding="utf-8"))
    except Exception:
        return None

    if isinstance(payload, list):
        return payload[0] if payload else None
    if isinstance(payload, dict):
        return payload
    return None


def evaluate_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "..", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.dataset not in cfg.get("datasets", {}):
        available = ", ".join(sorted(cfg.get("datasets", {}).keys()))
        raise KeyError(f"Unknown dataset {args.dataset!r}. Available: {available}")

    ds_cfg = cfg["datasets"][args.dataset]
    model_name = MODEL_MAP.get(args.model, args.model)

    qa_path = os.path.join(
        cfg["base_folder"], args.dataset, ds_cfg["qa_file"]
    )
    prediction_root = os.path.join(
        cfg["out_folder"],
        args.dataset,
        ds_cfg["result_dir"],
        f"runs_out_{args.mode}_{model_name}",
    )

    output_jsonl = args.output or os.path.join(
        cfg["out_folder"],
        args.dataset,
        "our",
        f"{args.mode}_{model_name}_predictions_with_eval.jsonl",
    )
    summary_json = os.path.splitext(output_jsonl)[0] + "_summary.json"

    qa_rows = read_jsonl(qa_path)
    evaluated_rows: List[Dict[str, Any]] = []
    missing_ids: List[str] = []

    for qa in qa_rows:
        qa_id = f"single_tab_{qa['id']}"
        result_path = os.path.join(prediction_root, qa_id, "result.json")
        prediction_record = _load_prediction(result_path)

        table_id = qa.get("table_id")
        if isinstance(table_id, list) and len(table_id) == 1:
            table_id = table_id[0]

        row: Dict[str, Any] = {
            "qa_id": qa_id,
            "id": qa.get("id"),
            "dataset_name": args.dataset,
            "method_name": "SemiBonsai",
            "table_id": table_id,
            "query": qa.get("query"),
            "label": qa.get("label"),
            "final_answer": None,
            "ev_res": 0,
            "status": "missing_prediction",
            "evaluation": None,
        }

        if prediction_record is None:
            missing_ids.append(qa_id)
            evaluated_rows.append(row)
            continue

        prediction = prediction_record.get("final_answer")
        evaluation = evaluate_pair(
            prediction,
            qa.get("label"),
            llm_generate=llm_generate if args.use_llm_fallback else None,
            llm_model=args.judge_model,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
            max_exp=args.max_exp,
            use_llm_fallback=args.use_llm_fallback,
        )

        row.update(
            {
                "final_answer": prediction,
                "ev_res": int(bool(evaluation["is_correct"])),
                "status": "evaluated",
                "evaluation": evaluation,
            }
        )
        evaluated_rows.append(row)

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    tmp_output = output_jsonl + ".tmp"
    with open(tmp_output, "w", encoding="utf-8") as f:
        for row in evaluated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_output, output_jsonl)

    found_count = len(evaluated_rows) - len(missing_ids)
    correct_count = sum(row["ev_res"] for row in evaluated_rows)
    total_count = len(evaluated_rows)
    method_counts: Dict[str, int] = {}
    for row in evaluated_rows:
        evaluation = row.get("evaluation") or {}
        method = evaluation.get("method", row["status"])
        method_counts[method] = method_counts.get(method, 0) + 1

    summary = {
        "dataset_name": args.dataset,
        "model": model_name,
        "mode": args.mode,
        "total_questions": total_count,
        "predictions_found": found_count,
        "missing_predictions": len(missing_ids),
        "correct": correct_count,
        "accuracy_all_questions": correct_count / total_count if total_count else 0.0,
        "accuracy_found_predictions": correct_count / found_count if found_count else 0.0,
        "evaluation_method_counts": method_counts,
        "missing_qa_ids": missing_ids,
        "predictions_with_eval": output_jsonl,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] Per-question evaluation saved to: {output_jsonl}")
    print(f"[DONE] Summary saved to: {summary_json}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate query_plan.py predictions for one configured dataset."
    )
    parser.add_argument("--dataset", required=True, help="Dataset key in codes/config.yaml")
    parser.add_argument("--mode", default="raw", choices=["raw"])
    parser.add_argument("--model", default="GPT-4o", help="Display name in MODEL_MAP or backend model id")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    parser.add_argument("--output", default=None, help="Optional output JSONL path")
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument("--abs-tol", type=float, default=1e-8)
    parser.add_argument("--max-exp", type=int, default=3)
    parser.add_argument("--use-llm-fallback", action="store_true")
    parser.add_argument("--judge-model", default="gpt-4o")
    evaluate_dataset(parser.parse_args())

if __name__ == "__main__":
    main()

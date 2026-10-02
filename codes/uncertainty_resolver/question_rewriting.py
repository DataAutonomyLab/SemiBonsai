from uncertainty_detection import *

import argparse
import glob
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List
import yaml
from loguru import logger
from tqdm import tqdm
from pruning import prune_phrase_groundings_with_layered_tree

from utils.basic_utils import (
    load_table_meta_from_layered_tree,
    build_union_meta_for_raw,
    extract_json_from_text,
    read_jsonl
)
from utils.api_utils import llm_generate_setup as llm_generate
from utils.constants import MODEL_MAP


def _clean_structural_elements(elements):
    """Remove empty/null path components, including legacy strings like 'None x'."""
    cleaned = []
    seen = set()
    for element in elements or []:
        parts = [
            part.strip()
            for part in str(element).split()
            if part.strip().lower() not in {"none", "null", "nan"}
        ]
        value = " ".join(parts).strip()
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)
    return cleaned


def _rewrite_preserves_constraints(original, rewritten):
    """Reject rewrites that introduce null paths or drop explicit constraints."""
    if not rewritten or re.search(r"\b(?:none|null|nan)\b", rewritten, re.IGNORECASE):
        return False

    def normalized(text):
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    rewritten_norm = normalized(rewritten)

    # Parenthetical text usually defines an entity (for example, "250 or more
    # employees"). It must remain present; losing it can change the calculation.
    for constraint in re.findall(r"\(([^()]*)\)", original):
        constraint_norm = normalized(constraint)
        if constraint_norm and constraint_norm not in rewritten_norm:
            return False

    # Preserve all explicit numbers, years and ranges from the original query.
    original_numbers = re.findall(r"(?<![a-z])[-+]?\d[\d,]*(?:\.\d+)?%?", original.lower())
    for number in original_numbers:
        if normalized(number) not in rewritten_norm:
            return False
    return True


def _canonical_schema(raw_schema, table_meta):
    """Keep only exact, axis-correct metadata paths from the model output."""
    raw_schema = raw_schema if isinstance(raw_schema, dict) else {}
    available_rows = set(table_meta.get("row_headers", []) or [])
    available_cols = set(table_meta.get("column_headers", []) or [])

    def valid(values, available):
        out = []
        for value in values or []:
            value = str(value).strip()
            if value in available and value not in out:
                out.append(value)
        return out

    return {
        "row_headers": valid(raw_schema.get("row_headers", []), available_rows),
        "column_headers": valid(raw_schema.get("column_headers", []), available_cols),
    }


def _suppress_generic_competing_phrases(question, grounding_by_phrase, table_meta=None):
    """An explicit `by X` axis outranks vague alternatives such as `kind of people`."""
    axis_match = re.search(r"\bby\s+([^,?]+)", str(question or ""), re.I)
    if not axis_match:
        return grounding_by_phrase
    axis = " ".join(re.findall(r"[a-z0-9]+", axis_match.group(1).casefold()))
    explicit_paths = []
    for path in (table_meta or {}).get("row_headers", []) or []:
        root = str(path).split("|")[0].strip()
        root_norm = " ".join(re.findall(r"[a-z0-9]+", root.casefold()))
        if axis and (axis in root_norm or root_norm in axis):
            explicit_paths.append(str(path))
    # If metadata directly resolves the requested axis, it is the complete
    # admissible grounding set. This also works when Stage 1 failed to emit a
    # separate `by X` phrase.
    if explicit_paths:
        return {f"by {axis_match.group(1).strip()}": list(dict.fromkeys(explicit_paths))}

    has_explicit_axis = any(
        re.match(r"^\s*by\b", str(phrase), re.I)
        for phrase in grounding_by_phrase
    )
    if not has_explicit_axis:
        return grounding_by_phrase
    generic = re.compile(
        r"\b(?:kind|type|group|category)\s+of\s+(?:people|person|persons|respondents?)\b",
        re.I,
    )
    return {
        phrase: paths
        for phrase, paths in grounding_by_phrase.items()
        if not generic.search(str(phrase))
    }


def _ensure_required_scope(question, schema, table_meta):
    """Add an explicit total/all denominator when ratio language requires it."""
    q = str(question or "").casefold()
    if not re.search(r"\b(?:account(?:ed)? for|share of)\b", q):
        return schema
    if not re.search(r"\b(?:total|all)\b", q):
        return schema

    out = {
        "row_headers": list(schema.get("row_headers", []) or []),
        "column_headers": list(schema.get("column_headers", []) or []),
    }

    def is_total_path(path):
        parts = [p.strip().casefold() for p in str(path).split("|") if p.strip()]
        leaf = parts[-1] if parts else ""
        return leaf in {"total", "all", "all companies", "overall", "national total"}

    for axis in ("row_headers", "column_headers"):
        available = table_meta.get(axis, []) or []
        totals = [str(path) for path in available if is_total_path(path)]
        # Prefer adding a denominator on an axis already used by this
        # interpretation; otherwise add it only when there is a single clear
        # total path on that axis.
        if totals and (out[axis] or len(totals) == 1):
            for path in totals:
                if path not in out[axis]:
                    out[axis].append(path)
    return out


def _explicit_axis_paths(question, table_meta):
    match = re.search(r"\bby\s+([^,?]+)", str(question or ""), re.I)
    if not match:
        return []
    axis = " ".join(re.findall(r"[a-z0-9]+", match.group(1).casefold()))
    matches = []
    for path in table_meta.get("row_headers", []) or []:
        root = str(path).split("|")[0].strip()
        root_norm = " ".join(re.findall(r"[a-z0-9]+", root.casefold()))
        if axis and (axis in root_norm or root_norm in axis):
            matches.append(str(path))
    # Prefer leaves. Keep the parent only if no leaf path exists.
    leaves = [path for path in matches if "|" in path]
    return leaves or matches


def _infer_operator_hint(question, proposed=""):
    allowed = {
        "direct", "sum", "average", "count", "min", "max", "difference",
        "decline_magnitude", "difference_rate", "proportion", "opposite",
        "argmin", "argmax",
    }
    proposed = str(proposed or "").strip().lower()
    q = str(question or "").lower()
    # Deterministic language rules override an inconsistent model suggestion.
    if re.search(r"\b(?:how many times|times (?:as )?(?:high|large|many|much)|multiple)\b", q):
        return "proportion"
    if re.search(r"\b(?:account(?:ed)? for|share of|relative to (?:the )?total)\b", q):
        return "proportion"
    # Check change semantics before generic superlatives.  Otherwise a question
    # such as "which category had the largest decrease" is incorrectly reduced
    # to argmax(raw value), rather than old-minus-new followed by argmax.
    if re.search(r"\b(?:declin|decreas|fell|fall|drop|lower)\w*\b", q):
        if "percentage point" in q or re.search(
            r"\b(?:largest|greatest|biggest|most|how much|amount)\b", q
        ):
            return "decline_magnitude"
    if re.search(r"\b(?:which|what)\b.*\b(?:highest|lowest|largest|smallest|more likely|less likely)\b", q):
        if re.search(r"\b(?:lowest|smallest|less likely)\b", q):
            return "argmin"
        return "argmax"
    if "percentage point" in q:
        return "difference"
    if re.search(r"\bpercent(?:age)?\s+(?:increase|decrease|decline|change|growth)\b", q):
        return "difference_rate"
    return proposed if proposed in allowed else "direct"


def _needs_semantic_rewrite(question, grounding_by_phrase):
    """Use rewriting only when grounding must resolve a genuine ambiguity.

    Exact mentions of a row/column label are schema-linking evidence, not an
    ambiguity by themselves.  Rewriting those questions used to paraphrase
    otherwise clear operator language (for example ``how much higher``), and
    hard pruning then made the paraphrase irreversible.
    """
    q = str(question or "").casefold()
    ambiguity_markers = re.compile(
        r"\b(?:this|these|those|such|former|latter|above|below|aforementioned|"
        r"respectively|specific|certain|the rest|the remainder)\b"
    )
    if ambiguity_markers.search(q):
        return True

    # A single phrase expanding to several structural paths is the main case
    # where the layered tree adds information (parent/category expansion).
    for paths in (grounding_by_phrase or {}).values():
        if len(list(dict.fromkeys(paths or []))) > 1:
            return True
    return False


def load_processed_ids(out_dir: str) -> set:
    processed = set()
    for fp in glob.glob(os.path.join(out_dir, "*.jsonl")):
        stem = os.path.splitext(os.path.basename(fp))[0]
        processed.add(stem)
    return processed


def process_single_pair(qa_pair, table_meta_infos, layered_tree, table_id, model_name, raw2subtab=None, invoke=False):
    cur_question = qa_pair["query"]

    # NEW table metadata format (labels only)
    # cur_table_meta = {
    #     "subtable_titles": table_meta_infos[table_id][0].get("subtable_titles", []),
    #     "column_headers": table_meta_infos[table_id][0].get("column_headers", []),
    #     "row_headers": table_meta_infos[table_id][0].get("row_headers", []),
    # }
    cur_table_meta = build_union_meta_for_raw(table_id, raw2subtab, table_meta_infos)

    start_time = time.time()

    # ---------- Stage 1: identify underspecified phrases + grounding ----------
    prompt1 = UncertaintyIdentifyAndGround_prompt.format(
        question=cur_question,
        table_metadata=json.dumps(cur_table_meta, ensure_ascii=False),
    )
    out1 = llm_generate(prompt1, model=model_name, json_format=True)

    if not str(out1.get("text", "") or "").strip():
        raise RuntimeError(
            "Uncertainty detection returned no model output; refusing to "
            "overwrite an existing rewriting record with an empty fallback."
        )

    input_tokens_1 = out1.get("input_tokens", 0)
    output_tokens_1 = out1.get("output_tokens", 0)

    try:
        res1 = extract_json_from_text(out1["text"])
    except Exception:
        res1 = {"phrase_groundings": []}

    raw_phrase_groundings = res1.get("phrase_groundings", [])
    if not isinstance(raw_phrase_groundings, list):
        raw_phrase_groundings = []

    stage1_time = time.time() - start_time

    output = prune_phrase_groundings_with_layered_tree(layered_tree, raw_phrase_groundings)

    underspecified_phrases = output[0]
    structural_elements = _clean_structural_elements(output[1])
    grounding_by_phrase = {
        phrase: _clean_structural_elements(paths)
        for phrase, paths in output[2].items()
    }
    grounding_by_phrase = {
        phrase: paths for phrase, paths in grounding_by_phrase.items() if paths
    }
    grounding_by_phrase = _suppress_generic_competing_phrases(
        cur_question, grounding_by_phrase, cur_table_meta
    )
    # The explicit-axis guard may replace a vague Stage-1 phrase with a new
    # canonical phrase such as `by household size`; keep those replacement
    # keys instead of intersecting them away.
    underspecified_phrases = list(grounding_by_phrase.keys())
    structural_elements = list(dict.fromkeys(
        path
        for paths in grounding_by_phrase.values()
        for path in paths
    ))

    use_rewrite = bool(
        underspecified_phrases
        and structural_elements
        and _needs_semantic_rewrite(cur_question, grounding_by_phrase)
    )

    # Clear questions keep their original wording and the full schema.  The
    # pruned paths are retained as diagnostic/ranking evidence, but are not a
    # hard boundary for query planning.
    if not use_rewrite:
        direct_candidate = {
            "rewritten_question": cur_question,
            "relevant_schema": {"row_headers": [], "column_headers": []},
            "scope_decisions": {},
            "operator_hint": _infer_operator_hint(cur_question),
            "candidate_source": "raw_full_schema",
        }
        return {
            "ambiguity": False,
            "use_rewrite": False,
            "time_cost": stage1_time,
            "token_cost": [input_tokens_1, output_tokens_1],
            "raw_phrase_groundings": raw_phrase_groundings,
            "underspecified_phrases": underspecified_phrases,
            "selected_groundings": structural_elements,
            "grounding_by_phrase": grounding_by_phrase,
            **direct_candidate,
            "candidates": [direct_candidate],
        }

    # ---------- Stage 2: rewrite question using pruned groundings ----------
    prompt2 = UncertaintyRewrite_prompt.format(
        question=cur_question,
        key_phrases=underspecified_phrases,
        structural_elements=structural_elements,
        grounding_by_phrase=json.dumps(grounding_by_phrase, ensure_ascii=False),
        table_metadata=json.dumps(cur_table_meta, ensure_ascii=False),
    )
    out2 = llm_generate(prompt2, model=model_name, json_format=True)
    res2 = extract_json_from_text(out2["text"])

    input_tokens_2 = out2.get("input_tokens", 0)
    output_tokens_2 = out2.get("output_tokens", 0)

    raw_candidates = res2.get("candidates", []) if isinstance(res2, dict) else []
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raw_candidates = [res2 if isinstance(res2, dict) else {}]

    candidates = []
    seen = set()
    explicit_axis_paths = _explicit_axis_paths(cur_question, cur_table_meta)
    for raw_candidate in raw_candidates[:3]:
        if not isinstance(raw_candidate, dict):
            continue
        candidate_question = str(raw_candidate.get("rewritten_question", cur_question)).strip()
        if not _rewrite_preserves_constraints(cur_question, candidate_question):
            candidate_question = cur_question
        relevant_schema = _canonical_schema(raw_candidate.get("relevant_schema"), cur_table_meta)
        relevant_schema = _ensure_required_scope(
            cur_question, relevant_schema, cur_table_meta
        )
        if explicit_axis_paths:
            # This is a hard query constraint, not an interpretation. The LLM
            # may see the full metadata and invent age/sex/employment axes even
            # when pruning supplied only household size, so enforce it here.
            relevant_schema["row_headers"] = list(explicit_axis_paths)
            candidate_question = cur_question
        scope_decisions = raw_candidate.get("scope_decisions", {})
        if not isinstance(scope_decisions, dict):
            scope_decisions = {}
        if explicit_axis_paths:
            locked_axis = explicit_axis_paths[0].split("|")[0].strip()
            scope_decisions = {"explicit_axis": locked_axis}
        operator_hint = _infer_operator_hint(cur_question, raw_candidate.get("operator_hint"))
        signature = (
            candidate_question.casefold(),
            tuple(relevant_schema["row_headers"]),
            tuple(relevant_schema["column_headers"]),
            operator_hint,
        )
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append({
            "rewritten_question": candidate_question,
            "relevant_schema": relevant_schema,
            "scope_decisions": scope_decisions,
            "operator_hint": operator_hint,
            "candidate_source": "rewritten_pruned_schema",
        })

    if not candidates:
        candidates = [{
            "rewritten_question": cur_question,
            "relevant_schema": {"row_headers": [], "column_headers": []},
            "scope_decisions": {},
            "operator_hint": _infer_operator_hint(cur_question),
            "candidate_source": "raw_full_schema",
        }]
    primary = candidates[0]

    total_time = time.time() - start_time
    total_in = input_tokens_1 + input_tokens_2
    total_out = output_tokens_1 + output_tokens_2

    return {
        "ambiguity": True,
        "use_rewrite": True,
        "time_cost": total_time,
        "token_cost": [total_in, total_out],
        # Stage-1 outputs
        "raw_phrase_groundings": raw_phrase_groundings,
        "underspecified_phrases": underspecified_phrases,
        "selected_groundings": structural_elements,
        "grounding_by_phrase": grounding_by_phrase,
        # Stage-2 outputs
        "rewritten_question": primary["rewritten_question"],
        "relevant_schema": primary["relevant_schema"],
        "scope_decisions": primary["scope_decisions"],
        "operator_hint": primary["operator_hint"],
        "candidates": candidates,
    }


def run_multihiertt_benchmark(mode, model_name, cfg, args, aug_mode=False, overwrite_existing=True, qa_rids=[]):
    base_dir = cfg["base_folder"]
    out_folder = cfg["out_folder"]
    ds_cfg = cfg["datasets"][args.dataset]

    question_type = "single_tab"
    vlm_dir = os.path.join(out_folder, args.dataset, "our/table_processed/meta_infos")

    if not aug_mode:
        qa_path = os.path.join(base_dir, args.dataset, ds_cfg["qa_file"])
    else:
        qa_path = os.path.join(base_dir, args.dataset, ds_cfg["aug_qa_file"])

    out_dir = os.path.join(out_folder, args.dataset, ds_cfg["diamb_dir"], model_name)
    os.makedirs(out_dir, exist_ok=True)

    processed_ids = load_processed_ids(out_dir)

    print(f"🚀 [INFO] Starting Benchmark: {args.dataset}")

    # ---- load layered-tree label meta (keyed by SUBTABLE_ID) ----
    value_index_root = os.path.join(out_folder, args.dataset, "our/value_index")
    table_meta_infos, layered_trees = load_table_meta_from_layered_tree(
        value_index_root,
        return_tree_map=True,
    )
    print(f"[INFO] Loaded layered-tree meta for {len(table_meta_infos)} subtables")


    # ---- build RAW -> SUBTAB fallback mapping ----
    raw2subtab = {}

    multitab_path = os.path.join(out_folder, args.dataset, "our/table_processed/multitab_mapping.jsonl")
    if os.path.exists(multitab_path):
        with open(multitab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_id = record.get("raw_table_id")
                subtab_ids = record.get("subtab_ids", []) or []
                # keep only those we actually have in table_meta_infos
                subtab_ids = [s for s in subtab_ids if s in table_meta_infos]
                if raw_id and subtab_ids:
                    raw2subtab[raw_id] = subtab_ids

    # also add a weak fallback from subtab_id naming: infer raw_id -> one subtab
    for sid in table_meta_infos.keys():
        if isinstance(sid, str) and sid.startswith("subtab_"):
            # infer raw from "subtab_{raw}_..."
            parts = sid.split("_", 2)
            if len(parts) >= 2:
                raw = parts[1]
                raw2subtab.setdefault(raw, [])
                if sid not in raw2subtab[raw]:
                    raw2subtab[raw].append(sid)

    def normalize_table_ids(tids):
        """
        Convert QA table_id(s) to subtab ids that exist in table_meta_infos.
        Strategy:
          - if tid is already a subtab id and exists -> keep
          - else if tid is a raw id and we have mapping -> replace with first mapped subtab
          - else drop
        Returns: list[subtab_id]
        """
        if isinstance(tids, str):
            tids = [tids]
        if not isinstance(tids, list):
            return []
        

        out = []
        for tid in tids:
            if tid in table_meta_infos:
                out.append(tid)
            elif tid in raw2subtab and raw2subtab[tid]:
                # pick first as default representative
                out.append(raw2subtab[tid][0])
            else:
                # unknown
                pass
        return out

    print(f"[INFO] Loading QA pairs from: {qa_path}")
    qa_pairs = read_jsonl(qa_path)

    if qa_rids:
        requested_ids = {str(x) for x in qa_rids}
        qa_pairs = [qa for qa in qa_pairs if str(qa.get("id")) in requested_ids]
        print(f"[Filtering] Selected {len(qa_pairs)} QA pairs by --qa_rids")

    if aug_mode:
        print(f"[INFO] Loaded {len(qa_pairs)} QA pairs")
        qa_pairs = [_ for _ in qa_pairs if _["llm"] == model_name]
        print(f"[Filtering] Resulting {len(qa_pairs)} QA pairs")

    print(f"[INFO] Loaded {len(qa_pairs)} QA pairs")

    valid_qa_pairs: List[Dict[str, Any]] = []
    print("[INFO] Filtering QA pairs...")

    dropped_no_meta = 0
    dropped_non_numeric = 0
    dropped_multi_table = 0

    for qa in tqdm(qa_pairs, total=len(qa_pairs)):
        tids_raw = qa.get("table_id")
        tids = normalize_table_ids(tids_raw)

        if not tids:
            dropped_no_meta += 1
            continue

        # Preserve the original raw table id(s) for union-metadata lookup.
        qa["_raw_table_ids"] = tids_raw if isinstance(tids_raw, list) else [tids_raw]

        # update qa in-place so later code uses normalized subtab id
        qa["table_id"] = tids

        # numeric answer only
        try:
            float(qa["label"])
        except Exception:
            dropped_non_numeric += 1
            continue

        # single table only
        if question_type == "single_tab" and len(tids) > 1:
            dropped_multi_table += 1
            continue

        # VLM meta check is optional now; do NOT fail if missing
        # (and you currently don't even use single_subtable filter)
        try:
            _ = [json.load(open(os.path.join(vlm_dir, f"{tid}.json"), "r")) for tid in tids]
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[Filter Error] during VLM check: {e}")
            pass

        valid_qa_pairs.append(qa)

    print(f"[INFO] Dropped (no meta after normalize): {dropped_no_meta}")
    print(f"[INFO] Dropped (non-numeric label): {dropped_non_numeric}")
    print(f"[INFO] Dropped (multi-table for single_tab): {dropped_multi_table}")

    if not valid_qa_pairs:
        # provide actionable debugging info
        some_keys = list(table_meta_infos.keys())[:5]
        raise RuntimeError(
            "No valid QA pairs found after filtering.\n"
            f"- table_meta_infos size: {len(table_meta_infos)}\n"
            f"- example meta keys: {some_keys}\n"
            f"- dropped_no_meta={dropped_no_meta}, dropped_non_numeric={dropped_non_numeric}, dropped_multi_table={dropped_multi_table}\n"
            "Likely cause: QA table_id uses raw ids but value_index/layered_tree was generated only for subtables.\n"
        )

    print(f"✅ [INFO] Found {len(valid_qa_pairs)} valid QA pairs after filtering.")
    failed_cases = []

    for qa_pair in tqdm(valid_qa_pairs, total=len(valid_qa_pairs)):
        table_id = qa_pair["table_id"][0]  # now normalized to subtab id

        raw_table_ids = qa_pair.get("_raw_table_ids", [table_id])
        table_ref = raw_table_ids[0] if len(raw_table_ids) == 1 else raw_table_ids

        # Prefer the raw-table tree (needed for one raw table with multiple
        # subtables), then fall back to the normalized subtable/table id.
        tree_key = raw_table_ids[0] if raw_table_ids else table_id
        current_layered_tree = layered_trees.get(tree_key) or layered_trees.get(table_id)
        if current_layered_tree is None:
            raise KeyError(
                f"No layered_tree found for raw table {tree_key!r} "
                f"or normalized table {table_id!r}"
            )

        if args.dataset == "realhitbench_num":
            processed_record = process_single_pair(qa_pair, table_meta_infos, current_layered_tree, table_ref, model_name, raw2subtab, invoke=True)
        else:
            processed_record = process_single_pair(qa_pair, table_meta_infos, current_layered_tree, table_ref, model_name, raw2subtab)

        save_fpath = os.path.join(out_dir, f'{qa_pair["id"]}.jsonl')
        with open(save_fpath, "w", encoding="utf-8") as f:
            f.write(json.dumps(processed_record, ensure_ascii=False, indent=2) + "\n")

    
    print(failed_cases)


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument(
        "--mode",
        type=str,
        default="raw"
    )
    parser.add_argument("--model", type=str, default="GPT-4o")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument(
        "--qa_rids",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=[],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()

    model_name = MODEL_MAP[args.model]

    with open("../config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    if args.dataset == "multihiertt_num":
        args.qa_rids = [int(_) for _ in args.qa_rids]

    run_multihiertt_benchmark(
        args.mode,
        model_name,
        cfg,
        args,
        aug_mode=args.aug,
        overwrite_existing=True,
        qa_rids=args.qa_rids,
    )

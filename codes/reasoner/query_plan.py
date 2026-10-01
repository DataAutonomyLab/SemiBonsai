# -*- coding: utf-8 -*-

import os
import json
import time
import pickle
import re
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from utils.basic_utils import (
    analyze_llm_output,
    load_multitab_mapping_jsonl,
    extract_python_code,
    run_extracted_code,
    load_table_meta,
    read_jsonl,
    extract_json_from_text,
    parse_llm_dict,
    safe_json_loads,
    load_table_meta_from_layered_tree,
    embed_fn,   # optional
)

from utils.api_utils import llm_generate_setup as llm_generate
from utils.constants import MODEL_MAP

from operation import (
    action_generation_prompt,
    program_generation_prompt,
    schema_linking_prompt,
)

LLM_MODEL = "gpt-4.1"
TEMPERATURE = 0


def _usable_rewritten_question(original: str, candidate: str) -> str:
    """Use a rewrite only when it keeps the original's explicit constraints."""
    candidate = str(candidate or "").strip()
    if not candidate or re.search(r"\b(?:none|null|nan)\b", candidate, re.IGNORECASE):
        return original

    def normalized(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    candidate_norm = normalized(candidate)
    for constraint in re.findall(r"\(([^()]*)\)", original):
        constraint_norm = normalized(constraint)
        if constraint_norm and constraint_norm not in candidate_norm:
            return original
    for number in re.findall(r"(?<![a-z])[-+]?\d[\d,]*(?:\.\d+)?%?", original.lower()):
        if normalized(number) not in candidate_norm:
            return original
    return candidate


def _infer_raw_operator_hint(question: str) -> str:
    """Conservative operator hints for the unmodified-question candidate."""
    q = str(question or "").casefold()
    if re.search(r"\b(?:account(?:ed)?\s+(?:for|of)|share of|how many times|ratio of)\b", q):
        return "proportion"
    if (
        re.search(r"\bpercentage of\b.*\b(?:among|of all)\b", q)
        or re.search(r"\bamong\b.*\bpercentage of\b", q)
    ):
        return "proportion"
    if re.search(r"\b(?:higher|lower|more likely|less likely)\b.*\bthan\b", q):
        return "difference"
    if re.search(r"\b(?:how much|by what percentage|how many percentage points)\b.*\b(?:declin|decreas|fell|drop)\w*\b", q):
        return "decline_magnitude"
    if re.search(r"\bhow much\b.*\b(?:higher|lower|more likely|less likely)\b", q):
        return "difference"
    if re.search(r"\bdifference\b", q):
        return "difference"
    # In HiTab, `range` often denotes a multi-valued answer whose numeric
    # conversion retained only one endpoint.  Do not silently reinterpret it
    # as max-min; let the table evidence and aggregation metadata drive it.
    if re.search(r"\branges?\b", q):
        return "range"
    if re.search(r"\b(?:averaged annually|average annual|average pace)\b", q):
        return "period_average"
    if re.search(r"\b(?:average|averaged annually|mean)\b", q):
        return "average"
    return ""


def _norm_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _explicit_matching_rows(question: str, table_meta: Dict[str, Any]) -> List[str]:
    """Return complete row paths whose leaf is explicitly named."""
    q_norm = _norm_text(question)
    matched = []
    for row in (table_meta or {}).get("row_headers", []) or []:
        row = str(row)
        leaf = _norm_text(row.split("|")[-1])
        if len(leaf) >= 3 and re.search(
            rf"(?<![a-z0-9]){re.escape(leaf)}(?![a-z0-9])", q_norm
        ):
            matched.append(row)
    return list(dict.fromkeys(matched))


def _exact_period_candidate(question: str, table_meta: Dict[str, Any]):
    """Prefer a table-provided period value over rebuilding it from annuals."""
    q = str(question or "").casefold()
    if not re.search(r"\b(?:averaged annually|average annual|average pace)\b", q):
        return None
    period = re.search(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)?\d{2})\b", q)
    if not period:
        return None
    start, end = period.groups()
    if len(end) == 2:
        end = start[:2] + end
    columns = [str(x) for x in (table_meta or {}).get("column_headers", []) or []]

    def has_period(col):
        years = _explicit_years(col)
        if start in years and end in years:
            return True
        compact = _norm_text(col)
        return _norm_text(f"{start}-{end[-2:]}") in compact

    matches = [col for col in columns if has_period(col)]
    if len(matches) != 1:
        return None
    rows = _explicit_matching_rows(question, table_meta)
    # Growth/pace refers to real (constant-dollar) values when the table
    # offers nominal and real versions of the same metric.
    if re.search(r"\b(?:growth|pace)\b", q):
        constant_rows = [
            str(row) for row in (table_meta or {}).get("row_headers", []) or []
            if "constant" in str(row).casefold()
            and any(
                token in _norm_text(row)
                for token in re.findall(r"[a-z0-9]+", q)
                if token in {"gdp", "domestic", "product", "total", "r", "d"}
            )
        ]
        if len(constant_rows) == 1:
            rows = constant_rows
    return {
        "rewritten_question": question,
        "relevant_schema": {
            "row_headers": rows,
            "column_headers": matches,
        },
        "scope_decisions": {
            "exact_period_column": matches[0],
            "period_semantics": "Use the table-provided aggregate for the exact requested period.",
        },
        "operator_hint": "direct",
        "candidate_source": "exact_period_schema",
        "answer_transform": "identity",
    }


def _percent_distribution_candidate(question: str, table_meta: Dict[str, Any]):
    """Use an explicit distribution column when it already encodes the share."""
    q = str(question or "").casefold()
    if not re.search(r"\b(?:account(?:ed)?\s+(?:for|of)|share of)\b", q):
        return None
    columns = [str(x) for x in (table_meta or {}).get("column_headers", []) or []]
    matches = [c for c in columns if re.search(r"\bpercent(?:age)? distribution\b", c.casefold())]
    rows = _explicit_matching_rows(question, table_meta)
    if len(matches) != 1 or not rows:
        return None
    return {
        "rewritten_question": question,
        "relevant_schema": {"row_headers": rows, "column_headers": matches},
        "scope_decisions": {"precomputed_share_column": matches[0]},
        "operator_hint": "direct",
        "candidate_source": "precomputed_share_schema",
        "answer_transform": "identity",
    }


def _matrix_share_candidate(question: str, table_meta: Dict[str, Any]):
    """Resolve `column entity accounts for row total` matrix questions."""
    q = str(question or "").casefold()
    marker = re.search(r"\baccount(?:ed)?\s+(?:for|of)\b", q)
    if not marker:
        return None
    before, after = q[:marker.start()], q[marker.end():]
    rows = [str(x) for x in (table_meta or {}).get("row_headers", []) or []]
    cols = [str(x) for x in (table_meta or {}).get("column_headers", []) or []]
    total_cols = [c for c in cols if _norm_text(c.split("|")[-1]) == "total"]

    # A hierarchical row matrix may encode the requested performer beneath
    # the requested work type, e.g. `basic research | higher education`, with
    # `basic research | total` as its denominator.
    nested_numerators = []
    for row in rows:
        parts = [part.strip() for part in row.split("|")]
        if len(parts) < 2:
            continue
        if (
            _norm_text(parts[-1]) in _norm_text(before)
            and any(_norm_text(parent) in _norm_text(after) for parent in parts[:-1])
        ):
            nested_numerators.append(row)
    if len(nested_numerators) == 1 and len(total_cols) == 1:
        parent_path = nested_numerators[0].split("|")[:-1]
        nested_totals = [
            row for row in rows
            if row.split("|")[:-1] == parent_path
            and _norm_text(row.split("|")[-1]) == "total"
        ]
        if len(nested_totals) == 1:
            return {
                "rewritten_question": question,
                "relevant_schema": {
                    "row_headers": nested_numerators + nested_totals,
                    "column_headers": total_cols,
                },
                "scope_decisions": {
                    "numerator_row": nested_numerators[0],
                    "denominator_row": nested_totals[0],
                    "metric_column": total_cols[0],
                },
                "operator_hint": "proportion",
                "candidate_source": "hierarchical_row_share_schema",
                "answer_transform": "identity",
            }

    subject_cols = [c for c in cols if _norm_text(c.split("|")[-1]) in _norm_text(before)]
    object_rows = [r for r in rows if _norm_text(r.split("|")[-1]) in _norm_text(after)]
    if len(subject_cols) != 1 or len(object_rows) != 1 or len(total_cols) != 1:
        return None
    return {
        "rewritten_question": question,
        "relevant_schema": {
            "row_headers": object_rows,
            "column_headers": subject_cols + total_cols,
        },
        "scope_decisions": {
            "numerator_column": subject_cols[0],
            "denominator_column": total_cols[0],
            "metric_row": object_rows[0],
        },
        "operator_hint": "proportion",
        "candidate_source": "matrix_share_schema",
        "answer_transform": "identity",
    }


def _risk_reference_candidate(question: str, table_meta: Dict[str, Any]):
    """Make an implicit unexposed risk-ratio reference explicit."""
    q = str(question or "").casefold()
    if not re.search(r"\b(?:did not|without|no)\b", q):
        return None
    cols = [str(x) for x in (table_meta or {}).get("column_headers", []) or []]
    risk_cols = [c for c in cols if "risk ratio" in c.casefold()]
    if not risk_cols:
        return None
    gendered = [
        c for c in risk_cols
        if (" men " in f" {q} " and "men" in c.casefold())
        or (" women " in f" {q} " and "women" in c.casefold())
    ]
    selected_cols = gendered or risk_cols
    if len(selected_cols) != 1:
        return None
    clarified = (
        question.rstrip(" ?")
        + "? Use the exposed group's risk ratio and the conventional unexposed "
          "reference risk ratio of 1.0; do not substitute another treatment row."
    )
    return {
        "rewritten_question": clarified,
        "relevant_schema": {"row_headers": [], "column_headers": selected_cols},
        "scope_decisions": {"implicit_reference_risk_ratio": 1.0},
        "operator_hint": "difference",
        "candidate_source": "risk_reference_schema",
        "answer_transform": "absolute_value",
    }


def _interval_share_candidate(question: str, table_meta: Dict[str, Any]):
    """Expand a numeric entity interval such as 50-249 into all covered rows."""
    q = str(question or "").casefold()
    if not re.search(r"\b(?:account(?:ed)?\s+(?:for|of)|share of)\b", q):
        return None
    interval = re.search(r"\(?\b(\d[\d,]*)\s*[-–]\s*(\d[\d,]*)\b", q)
    if not interval:
        return None
    low, high = (int(x.replace(",", "")) for x in interval.groups())
    rows = [str(x) for x in (table_meta or {}).get("row_headers", []) or []]
    covered = []
    for row in rows:
        leaf = row.split("|")[-1]
        bounds = re.search(r"\b(\d[\d,]*)\s*[-–]\s*(\d[\d,]*)\b", leaf)
        if not bounds:
            continue
        row_low, row_high = (int(x.replace(",", "")) for x in bounds.groups())
        if low <= row_low and row_high <= high:
            covered.append(row)
    totals = [
        row for row in rows
        if _norm_text(row.split("|")[-1]) in {"total", "all", "all companies", "overall"}
    ]
    if not covered or len(totals) != 1:
        return None
    return {
        "rewritten_question": question,
        "relevant_schema": {
            "row_headers": covered + totals,
            "column_headers": [],
        },
        "scope_decisions": {
            "interval": [low, high],
            "numerator_rows": covered,
            "denominator_row": totals[0],
        },
        "operator_hint": "proportion",
        "candidate_source": "interval_share_schema",
        "answer_transform": "identity",
    }


def _infer_answer_transform(question: str) -> str:
    """Return a label-free normalization implied by comparative wording."""
    q = str(question or "").casefold()
    comparative_magnitude = bool(
        re.search(r"\b(?:higher|lower|more likely|less likely)\b.*\bthan\b", q)
    )
    asks_magnitude = comparative_magnitude or bool(
        re.search(r"\b(?:how much|how many percentage points|amount|largest|greatest|biggest)\b", q)
        and re.search(r"\b(?:declin|decreas|fell|drop|reced|lower|less likely)\w*\b", q)
    )
    return "absolute_value" if asks_magnitude else "identity"


def _derived_metric_candidate(question: str, table_meta: Dict[str, Any]):
    """Return a direct-lookup candidate when the table already has the metric.

    This prevents a percent-change/annual-growth column from being replaced by
    an unnecessary reconstruction from raw values.
    """
    q = str(question or "").casefold()
    wants_change = bool(re.search(r"\b(?:change|growth|declin|decreas|fell|drop|reced)\w*\b", q))
    if not wants_change:
        return None

    columns = [str(x) for x in (table_meta or {}).get("column_headers", []) or []]
    derived = [
        col for col in columns
        if re.search(r"\b(?:percent(?:age)? change|change in rate|growth rate|annual growth)\b", col.casefold())
    ]
    if not derived:
        return None

    stop = {
        "what", "was", "were", "the", "how", "much", "many", "rate", "of",
        "in", "on", "to", "from", "compared", "with", "and", "percent",
        "percentage", "change", "decreased", "decrease", "declined", "decline",
    }
    q_tokens = set(re.findall(r"[a-z0-9]+", q)) - stop

    def score(col):
        c_tokens = set(re.findall(r"[a-z0-9]+", col.casefold())) - stop
        return len(q_tokens & c_tokens)

    best_score = max(score(col) for col in derived)
    best = [col for col in derived if score(col) == best_score]
    # Avoid injecting a broad alternative when several unrelated metrics tie.
    if len(best) != 1:
        return None

    rows = [str(x) for x in (table_meta or {}).get("row_headers", []) or []]

    def norm(value):
        return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))

    q_norm = norm(question)
    explicit_rows = []
    for row in rows:
        leaf = str(row).split("|")[-1].strip()
        leaf_norm = norm(leaf)
        if len(leaf_norm) >= 3 and re.search(
            rf"(?<![a-z0-9]){re.escape(leaf_norm)}(?![a-z0-9])", q_norm
        ):
            explicit_rows.append(row)

    selected_rows = list(dict.fromkeys(explicit_rows))
    if not selected_rows:
        total_names = {
            "total", "all", "overall", "canada", "national total",
            "total provinces and territories", "total all jurisdictions",
        }
        total_rows = [
            row for row in rows
            if norm(str(row).split("|")[-1]) in total_names
        ]
        if len(total_rows) == 1:
            selected_rows = total_rows

    return {
        "rewritten_question": question,
        "relevant_schema": {"row_headers": selected_rows, "column_headers": best},
        "scope_decisions": {
            "derived_metric": best[0],
            "derived_metric_rows": selected_rows,
        },
        "operator_hint": "direct",
        "candidate_source": "derived_metric_schema",
        "answer_transform": _infer_answer_transform(question),
    }


# -----------------------------------------------------------------------------
# Value-index helpers (NEW)
# -----------------------------------------------------------------------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _load_layered_tree(value_index_dir: str) -> Dict[str, Any]:
    p = os.path.join(value_index_dir, "layered_tree.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing layered_tree.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_subtables_from_tree(layered_tree: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Return list of {"subtable_id":..., "title":...}
    """
    out = []
    for n in layered_tree.get("nodes", []):
        if n.get("type") == "subtable":
            sid = str(n.get("subtable_id", "")).strip()
            title = str(n.get("title", "")).strip()
            if sid:
                out.append({"subtable_id": sid, "title": title})
    # stable sort by title then id
    out.sort(key=lambda x: (x.get("title", ""), x.get("subtable_id", "")))
    return out


def _normalize_text(s: str) -> List[str]:
    import re
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def choose_best_subtable(
    query: str,
    subtables: List[Dict[str, str]],
    *,
    use_embed: bool = False,
    embed_fn_callable=None,
) -> Optional[Dict[str, str]]:
    """
    Pick best subtable by:
      - if use_embed and embed_fn_callable: cosine similarity between query and title
      - else token-overlap score between query tokens and title tokens
    """
    if not subtables:
        return None
    if len(subtables) == 1:
        return subtables[0]

    # ---- embedding route ----
    if use_embed and embed_fn_callable is not None:
        try:
            q_emb = embed_fn_callable([query])[0]
            titles = [st.get("title", "") or st.get("subtable_id", "") for st in subtables]
            t_embs = embed_fn_callable(titles)

            # cosine
            def cos(a, b):
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                if denom == 0:
                    return -1.0
                return float(np.dot(a, b) / denom)

            best_idx, best_score = 0, -1e9
            for i, emb in enumerate(t_embs):
                score = cos(q_emb, emb)
                if score > best_score:
                    best_idx, best_score = i, score
            return subtables[best_idx]
        except Exception:
            # fallback to token overlap
            pass

    # ---- token overlap fallback ----
    q_toks = set(_normalize_text(query))
    best = None
    best_score = -1
    for st in subtables:
        title = st.get("title", "") or st.get("subtable_id", "")
        t_toks = set(_normalize_text(title))
        if not t_toks:
            score = 0
        else:
            score = len(q_toks & t_toks)
        if score > best_score:
            best_score = score
            best = st
    return best


def build_df_for_subtable(
    value_index_dir: str,
    subtable_id: str,
    *,
    relevant_row_headers: Optional[List[str]] = None,
    relevant_column_headers: Optional[List[str]] = None,
    na_value: Any = "",
) -> pd.DataFrame:
    """
    Construct a dataframe for ONE subtable_id from raw-level:
      mapping_table.csv: col_path_id,row_path_id,value_id
      value_table.csv: value_id,value

    IMPORTANT:
    - We first filter mapping_table to only rows whose col_path_id and row_path_id belong to `subtable_id`.
    - Header selection uses the layered tree: a parent match expands to all of
      its descendant leaves, and either labels or complete paths may match.
    - A non-empty header request which resolves to no leaf is an error.  It
      must never silently fall back to the complete table.

    Output index/columns are "pretty labels" built from layered_tree:
      index:  "<title> | <row-path...>"
      col:    "<title> | <col-path...>"
    """
    layered_tree = _load_layered_tree(value_index_dir)
    nodes = layered_tree.get("nodes", [])
    # path_id -> (type, path(list), label, subtable_id, subtable_title)
    pid2info: Dict[str, Dict[str, Any]] = {}
    node_by_id = {
        str(n.get("id")): n for n in nodes
        if isinstance(n, dict) and n.get("id")
    }

    # build subtable node mapping
    stid2title: Dict[str, str] = {}
    for n in nodes:
        if n.get("type") == "subtable":
            sid = str(n.get("subtable_id", "")).strip()
            stid2title[sid] = str(n.get("title") or "").strip()

    for n in nodes:
        if n.get("type") in ("colhdr", "rowhdr") and n.get("is_leaf", False):
            pid = str(n.get("path_id", "")).strip()
            if not pid:
                continue
            path = n.get("path")
            if not isinstance(path, list) or not path:
                # fallback to single label
                lab = str(n.get("label", "")).strip()
                path = [lab] if lab else []
            # pid contains prefix: subtab_xxx::C::... or subtab_xxx::R::...
            # we can parse subtable_id from pid directly (safe)
            sid = pid.split("::", 1)[0]
            pid2info[pid] = {
                "type": n.get("type"),
                "path": [str(x).strip() for x in path if str(x).strip()],
                "label": str(n.get("label", "")).strip(),
                "subtable_id": sid,
                "subtable_title": stid2title.get(sid, ""),
            }

    def normalize_header(value: Any) -> str:
        value = (
            str(value or "")
            .casefold()
            .replace("%", " percent ")
            .replace("$", " dollar ")
            .replace("|", " ")
        )
        return " ".join(re.findall(r"[a-z0-9]+", value))

    def node_path(node: Dict[str, Any]) -> List[str]:
        path = node.get("path")
        if isinstance(path, list) and path:
            return [str(x).strip() for x in path if str(x).strip()]
        label = str(node.get("label", "")).strip()
        return [label] if label else []

    def descendant_leaf_pids(node_id: str, header_type: str) -> List[str]:
        """Return descendant leaf path ids in layered-tree/table order."""
        result: List[str] = []
        stack = [node_id]
        while stack:
            current_id = stack.pop(0)
            current = node_by_id.get(current_id, {})
            if current.get("type") != header_type:
                continue
            path_id = str(current.get("path_id", "")).strip()
            if current.get("is_leaf", False) and path_id:
                result.append(path_id)
                continue
            stack[0:0] = [str(x) for x in current.get("children_ids", [])]
        return result

    def resolve_headers(
        requested: Optional[List[str]],
        header_type: str,
        available_pids: List[str],
    ) -> Optional[List[str]]:
        wanted = [str(x).strip() for x in (requested or []) if str(x).strip()]
        if not wanted:
            return None

        matched = set()
        for raw_wanted in wanted:
            needle = normalize_header(raw_wanted)
            if not needle:
                continue
            candidates = []
            for node_id, node in node_by_id.items():
                if node.get("type") != header_type:
                    continue
                node_sid = str(node_id).split("::", 2)[1] if "::" in node_id else ""
                if node_sid != subtable_id:
                    continue
                label = normalize_header(node.get("label", ""))
                path = normalize_header(" | ".join(node_path(node)))
                # Prefer the most specific match.  This prevents a complete
                # leaf path such as "large companies | 250-499" from also
                # selecting its shorter parent "large companies".
                if path and needle == path:
                    score = (4, len(path))
                elif label and needle == label:
                    score = (3, len(label))
                elif path and path in needle:
                    score = (2, len(path))
                elif label and label in needle:
                    score = (1, len(label))
                elif path and needle in path:
                    score = (0, len(needle))
                elif label and needle in label:
                    score = (0, len(needle))
                else:
                    continue
                candidates.append((score, node_id))

            if candidates:
                best_score = max(score for score, _ in candidates)
                for score, node_id in candidates:
                    if score == best_score:
                        matched.update(descendant_leaf_pids(node_id, header_type))

        # mapping_table is the source of truth for stable physical table order.
        ordered = [pid for pid in available_pids if pid in matched]
        if not ordered:
            axis = "row" if header_type == "rowhdr" else "column"
            raise ValueError(
                f"No {axis} header matched for subtable {subtable_id}: {wanted}"
            )
        return ordered

    mapping_path = os.path.join(value_index_dir, "mapping_table.csv")
    value_path = os.path.join(value_index_dir, "value_table.csv")
    mp = _safe_read_csv(mapping_path)
    vt = _safe_read_csv(value_path)

    # expected columns
    for col in ["col_path_id", "row_path_id", "value_id"]:
        if col not in mp.columns:
            raise ValueError(f"mapping_table.csv missing column {col}. columns={list(mp.columns)}")
    for col in ["value_id", "value"]:
        if col not in vt.columns:
            raise ValueError(f"value_table.csv missing column {col}. columns={list(vt.columns)}")

    mp["col_path_id"] = mp["col_path_id"].astype(str)
    mp["row_path_id"] = mp["row_path_id"].astype(str)

    # 1) restrict to this subtable_id
    # path_id prefix is subtable_id
    mp_sub = mp[
        mp["col_path_id"].str.startswith(subtable_id + "::")
        & mp["row_path_id"].str.startswith(subtable_id + "::")
    ].copy()

    # 2) resolve requested nodes, expanding parent nodes to descendant leaves.
    row_order = mp_sub["row_path_id"].drop_duplicates().tolist()
    col_order = mp_sub["col_path_id"].drop_duplicates().tolist()
    keep_row_pids = resolve_headers(relevant_row_headers, "rowhdr", row_order)
    keep_col_pids = resolve_headers(relevant_column_headers, "colhdr", col_order)
    if keep_row_pids is not None:
        mp_sub = mp_sub[mp_sub["row_path_id"].isin(keep_row_pids)]
        row_order = keep_row_pids
    if keep_col_pids is not None:
        mp_sub = mp_sub[mp_sub["col_path_id"].isin(keep_col_pids)]
        col_order = keep_col_pids

    # join values
    merged = mp_sub.merge(vt, on="value_id", how="left")
    merged["value"] = merged["value"].fillna(na_value).astype(str)

    # pivot
    pivot = merged.pivot_table(
        index="row_path_id",
        columns="col_path_id",
        values="value",
        aggfunc="first",
        dropna=False,
        sort=False,
    )
    pivot = pivot.reindex(index=row_order, columns=col_order)

    # pretty labels
    def pretty(pid: str) -> str:
        info = pid2info.get(pid)
        if not info:
            return pid
        st_title = info.get("subtable_title", "")
        chain = info.get("path") or []
        parts = []
        if st_title:
            parts.append(st_title)
        parts.extend(chain)
        return " | ".join([p for p in parts if p])

    pivot.index = [pretty(x) for x in pivot.index.tolist()]
    pivot.columns = [pretty(x) for x in pivot.columns.tolist()]
    pivot = pivot.fillna(na_value)
    return pivot


# -----------------------------------------------------------------------------
# Original pipeline code (mostly unchanged)
# -----------------------------------------------------------------------------
@dataclass
class TableCtx:
    question: str = ""
    table_meta: Dict[str, Any] = field(default_factory=lambda: {"column_headers": [], "row_headers": []})
    table_values: Optional[pd.DataFrame] = None

    subquestions: Dict[str, Any] = field(default_factory=dict)
    atomic_subquestions: Dict[str, Any] = field(default_factory=dict)
    atomic_subquestion_mapping: Dict[str, Any] = field(default_factory=dict)

    operation_history: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_history: List[Dict[str, Any]] = field(default_factory=list)
    last_operation: str = ""

    atomic_subquestion_programs: Dict[str, Any] = field(default_factory=dict)


def init_context(question: str, table_meta: Dict[str, Any], df: pd.DataFrame):
    ctx = TableCtx()
    ctx.question = question
    ctx.table_meta = table_meta
    ctx.table_values = df

    ctx.subquestions = {"raw": question}
    ctx.atomic_subquestions = {"raw": question}
    ctx.atomic_subquestion_mapping = {}

    ctx.last_operation = "init"
    ctx.operation_history = []
    ctx.reasoning_history = []
    return ctx


def _deterministic_total_share_context(question, table_meta, df, canonical_schema, operator_hint):
    """Build a complete ratio plan for explicit `items account for total` queries."""
    q = str(question or "").lower()
    if operator_hint != "proportion":
        return None
    if not re.search(r"\b(?:account(?:ed)? for|share of|share .* total)\b", q):
        return None
    rows = [str(x).strip() for x in (canonical_schema or {}).get("row_headers", []) if str(x).strip()]
    if len(rows) < 2:
        return None

    total_leaves = {"total", "all", "all companies", "overall", "national total"}
    totals = [r for r in rows if r.split("|")[-1].strip().casefold() in total_leaves]
    numerators = [r for r in rows if r not in totals]
    if not totals or not numerators:
        return None

    ctx = init_context(question, table_meta, df)
    subquestions = {}
    for i, row in enumerate(numerators + [totals[0]], start=1):
        subquestions[f"#{i}"] = row
    numerator_ids = list(subquestions)[:-1]
    denominator_id = list(subquestions)[-1]
    numerator_expr = numerator_ids[0] if len(numerator_ids) == 1 else f"sum({', '.join(numerator_ids)})"
    formula = f"proportion({numerator_expr}, {denominator_id})"

    ctx.subquestions.update(subquestions)
    ctx.atomic_subquestions = dict(subquestions)
    ctx.last_operation = "infer_calculation_formula"
    mapping_json = json.dumps(subquestions, ensure_ascii=False)
    ctx.operation_history = [
        f'infer_calculation_formula("raw", "{formula}", {mapping_json})'
    ]
    ctx.reasoning_history = [
        f"To answer raw, you should utilize the formula: {formula}."
    ]
    return ctx


def _deterministic_matrix_share_context(question, table_meta, df, canonical_schema, operator_hint):
    """Build numerator-column / total-column for a single matrix row."""
    if operator_hint != "proportion":
        return None
    rows = [str(x).strip() for x in (canonical_schema or {}).get("row_headers", []) if str(x).strip()]
    cols = [str(x).strip() for x in (canonical_schema or {}).get("column_headers", []) if str(x).strip()]
    totals = [c for c in cols if _norm_text(c.split("|")[-1]) == "total"]
    numerators = [c for c in cols if c not in totals]
    if len(rows) != 1 or len(totals) != 1 or len(numerators) != 1:
        return None
    ctx = init_context(question, table_meta, df)
    subquestions = {
        "#1": f"{rows[0]} at {numerators[0]}",
        "#2": f"{rows[0]} at {totals[0]}",
    }
    ctx.subquestions.update(subquestions)
    ctx.atomic_subquestions = dict(subquestions)
    ctx.last_operation = "infer_calculation_formula"
    ctx.operation_history = [
        f'infer_calculation_formula("raw", "proportion(#1, #2)", {json.dumps(subquestions, ensure_ascii=False)})'
    ]
    ctx.reasoning_history = ["To answer raw, use proportion(#1, #2)."]
    return ctx


def _explicit_years(text):
    """Return explicit four-digit years in textual order, without duplicates."""
    return list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", str(text or ""))))


def _schema_and_linking_paths(candidate, ctx_dict):
    """Collect paths actually proposed and linked for label-free validation."""
    schema = candidate.get("relevant_schema", {}) or {}
    paths = list(schema.get("row_headers", []) or [])
    paths += list(schema.get("column_headers", []) or [])
    for linked in (ctx_dict.get("schema_linking", {}) or {}).values():
        if not isinstance(linked, dict):
            continue
        paths += list(linked.get("relevant_row_headers", []) or [])
        paths += list(linked.get("relevant_column_headers", []) or [])
    return list(dict.fromkeys(str(path) for path in paths if str(path).strip()))


def update_context(ctx, action_output, function):
    source_qs = action_output["parameters"]["subq_id"]
    target_qs = action_output["parameters"]["subquestions"]

    ctx.subquestions.update(target_qs)

    if source_qs in ctx.atomic_subquestions:
        del ctx.atomic_subquestions[source_qs]
    ctx.atomic_subquestions.update(target_qs)

    ctx.last_operation = action_output["action"]
    ctx.operation_history.append(function)

    if ctx.last_operation == "infer_calculation_formula":
        formula = action_output["parameters"]["formula"]
        ctx.reasoning_history.append(f"To answer {source_qs}, you should utilize the formula: {formula}.")
    elif ctx.last_operation == "multihop_question_decomposition":
        order = action_output["parameters"]["order"]
        ctx.reasoning_history.append(
            f"To answer the complex question {source_qs}, you should answer simpler subquestions in this sequence: {order}."
        )
    return ctx


possible_next_operation_dict = {
    "init": ["infer_calculation_formula", "multihop_question_decomposition", "generate_execute_program"],
    "infer_calculation_formula": ["multihop_question_decomposition", "generate_execute_program"],
    "multihop_question_decomposition": ["infer_calculation_formula", "generate_execute_program"],
}


def _is_direct_schema_fetch(question: str, table_meta: Dict[str, Any]) -> bool:
    """Return True when a subquestion already identifies a table cell/slice."""
    q_norm = " ".join(_normalize_text(question))

    def axis_matches(headers):
        if not headers:
            return True
        for header in headers:
            full = " ".join(_normalize_text(str(header)))
            parts = [
                " ".join(_normalize_text(part))
                for part in str(header).split("|")
            ]
            if (full and full in q_norm) or any(part and part in q_norm for part in parts):
                return True
        return False

    return axis_matches(table_meta.get("row_headers", [])) and axis_matches(
        table_meta.get("column_headers", [])
    )


def question_decomposition(
    opeartion_set, q, meta, df, model_name, mode="raw", max_steps=6,
    skip=False, operator_hint="",
):
    ctx = init_context(q, meta, df)
    total_input_tokens, total_output_tokens = 0, 0

    if skip:
        return ctx, total_input_tokens, total_output_tokens

    steps = 1
    invalid_action_attempts = 0
    action_repair_feedback = ""
    while steps < max_steps:
        # Once a calculation formula has produced operands that already map
        # directly to table headers, decomposition is complete. Asking the LLM
        # again can turn a scalar fetch into redundant sibling questions (for
        # example, splitting an already atomic `yes, rate` operand into both
        # yes and no). Stop deterministically and proceed to schema linking.
        if (
            ctx.last_operation == "infer_calculation_formula"
            and ctx.atomic_subquestions
            and all(
                _is_direct_schema_fetch(text, ctx.table_meta)
                for text in ctx.atomic_subquestions.values()
            )
        ):
            break

        if mode in ["raw"]:
            prompt = action_generation_prompt.format(
                operator_set=opeartion_set[0],
                dataset_specific_example=opeartion_set[1],
                subquestions=json.dumps(ctx.atomic_subquestions, ensure_ascii=False),
                table_metadata=json.dumps(ctx.table_meta, ensure_ascii=False),
                operation_history=json.dumps(ctx.operation_history, ensure_ascii=False),
                possible_actions=possible_next_operation_dict[ctx.last_operation],
            )
            if operator_hint and operator_hint != "direct":
                hint_text = {
                    "proportion": (
                        "Use numerator / matching population denominator. For wording "
                        "'among population X, what percentage were Y', compute Y within X; "
                        "do not use a percentage column that instead describes X within Y."
                    ),
                    "difference": (
                        "Use a signed/end-minus-start difference for percentage-point difference. "
                        "For a risk-ratio question comparing an exposed group with an "
                        "unstated 'did not/without' reference group, use the conventional "
                        "reference risk ratio 1.0 unless the table explicitly supplies that "
                        "counterpart; never substitute an unrelated treatment row."
                    ),
                    "decline_magnitude": (
                        "Return the positive decline magnitude: old value minus new value. "
                        "For 'largest/greatest decrease', compute old-minus-new for every "
                        "eligible entity over the exact requested endpoints, then select "
                        "the largest positive decline; do not select the smallest raw value."
                    ),
                    "difference_rate": "Use difference_rate for percent increase/decrease, not for 'how many times'.",
                    "period_average": (
                        "If the table provides a column for the exact multi-year period, "
                        "read that aggregate directly. Otherwise average only the annual "
                        "values; never divide that average by another trend value."
                    ),
                    "range": (
                        "Range can request endpoints rather than max-minus-min. Preserve "
                        "explicit percentile endpoints and the dataset aggregation semantics; "
                        "do not invent subtraction unless the question asks for a difference."
                    ),
                }.get(operator_hint, f"Prefer the operator semantics: {operator_hint}.")
                prompt += (
                    "\n\nCANONICAL OPERATOR HINT (treat as a hard semantic constraint): "
                    + operator_hint + ". " + hint_text
                )
            if action_repair_feedback:
                prompt += "\n\nPREVIOUS OUTPUT ERROR: " + action_repair_feedback

        if "deepseek" in model_name.lower():
            all_output = llm_generate(prompt, model=model_name, json_format=True)
            try:
                output = extract_json_from_text(all_output["text"])
            except Exception:
                print("-------------------------")
                print(all_output["text"])
                return None, 0, 0
        else:
            for _ in range(3):
                all_output = llm_generate(prompt, model=model_name, json_format=True)
                try:
                    output = parse_llm_dict(all_output["text"])
                    break
                except Exception:
                    prompt = prompt + "\n\nReturn ONLY a complete JSON object (no markdown fences)."

        input_tokens, output_tokens = all_output["input_tokens"], all_output["output_tokens"]
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        if "function" not in output:
            break

        function = output["function"]
        if "generate_execute_program" in function:
            break

        prev_ctx = deepcopy(ctx)
        try:
            action_output = analyze_llm_output(function)
            if not isinstance(action_output, dict) or not isinstance(action_output.get("parameters"), dict):
                raise ValueError(f"Unparseable action function: {function}")
            ctx = update_context(ctx, action_output, function)
            steps += 1
            invalid_action_attempts = 0
            action_repair_feedback = ""
        except Exception as e:
            invalid_action_attempts += 1
            action_repair_feedback = (
                f"{e}. Return a complete function call, not only the function "
                "name. For infer_calculation_formula include subq_id, formula, "
                "and the placeholder-to-subquestion mapping."
            )
            print(
                f"[WARNING] Invalid decomposition action "
                f"({invalid_action_attempts}/3): {e}"
            )
            ctx = prev_ctx
            if invalid_action_attempts >= 3:
                # Let process_single_res perform a clean outer retry instead
                # of treating a partially decomposed context as successful.
                return None, total_input_tokens, total_output_tokens

    return ctx, total_input_tokens, total_output_tokens


def _merge_grounding_hints(ctx, grounding_by_phrase, prefer_latest_column=False):
    """Deterministically apply valid rewrite groundings to linked leaf headers."""
    available_rows = {str(x).strip() for x in ctx.table_meta.get("row_headers", []) if str(x).strip()}
    available_cols = {str(x).strip() for x in ctx.table_meta.get("column_headers", []) if str(x).strip()}

    def norm(value):
        return " ".join(_normalize_text(str(value)))

    for subq_id, subq_text in ctx.atomic_subquestions.items():
        linked = ctx.atomic_subquestion_mapping.setdefault(
            subq_id, {"relevant_row_headers": [], "relevant_column_headers": []}
        )
        linked_rows = list(linked.get("relevant_row_headers", []) or [])
        linked_cols = list(linked.get("relevant_column_headers", []) or [])
        subq_norm = norm(subq_text)

        for phrase, hints in (grounding_by_phrase or {}).items():
            phrase_norm = norm(phrase)
            phrase_matches_subq = bool(phrase_norm and phrase_norm in subq_norm)
            matched_hints = [h for h in hints if norm(h) and norm(h) in subq_norm]
            if not phrase_matches_subq and not matched_hints:
                continue
            # A phrase-level match intentionally carries all of that phrase's
            # grounded paths. Otherwise add only the paths which themselves
            # occur in this atomic subquestion. This prevents one matching
            # leaf from pulling every sibling leaf into every operand.
            applicable_hints = hints if phrase_matches_subq else matched_hints
            for hint in applicable_hints:
                hint = str(hint).strip()
                # Path strings may include parent labels. Prefer the leaf label
                # when it is an exact member of the metadata label pool.
                candidates = [hint] + [part.strip() for part in hint.split("|") if part.strip()]
                for candidate in candidates:
                    if candidate in available_rows and candidate not in linked_rows:
                        linked_rows.append(candidate)
                    if candidate in available_cols and candidate not in linked_cols:
                        linked_cols.append(candidate)

        linked["relevant_row_headers"] = linked_rows
        linked["relevant_column_headers"] = linked_cols

    # HiTab numerical questions without an explicit year conventionally target
    # the latest period. Resolve a parent/all-years link to one concrete leaf.
    question = ctx.question or ""
    has_explicit_year = bool(re.search(r"\b(?:19|20)\d{2}\b", question))
    asks_for_period = bool(re.search(r"\b(?:between|from|over the period|trend|each year|all years)\b", question, re.I))
    year_headers = []
    for header in available_cols:
        match = re.search(r"(?:^|\|\s*)((?:19|20)\d{2})\s*$", header)
        if match:
            year_headers.append((int(match.group(1)), header))
    year_headers.sort()
    if prefer_latest_column and not has_explicit_year and not asks_for_period and year_headers:
        latest = year_headers[-1][1]
        known_year_headers = {header for _, header in year_headers}
        for linked in ctx.atomic_subquestion_mapping.values():
            cols = list(linked.get("relevant_column_headers", []) or [])
            if not cols or len(cols) > 1 or any(c not in known_year_headers for c in cols):
                linked["relevant_column_headers"] = [latest]


def _apply_canonical_schema(ctx, canonical_schema):
    """Apply candidate-specific schema as a hard scope, not a soft hint."""
    canonical_schema = canonical_schema if isinstance(canonical_schema, dict) else {}
    rows = [str(x).strip() for x in canonical_schema.get("row_headers", []) if str(x).strip()]
    cols = [str(x).strip() for x in canonical_schema.get("column_headers", []) if str(x).strip()]
    available_rows = set(ctx.table_meta.get("row_headers", []) or [])
    available_cols = set(ctx.table_meta.get("column_headers", []) or [])
    rows = [x for x in rows if x in available_rows]
    cols = [x for x in cols if x in available_cols]
    if not rows and not cols:
        return

    def norm(value):
        return " ".join(_normalize_text(str(value)))

    def best_path_matches(headers, subq_norm):
        """Choose the most specific canonical paths mentioned by a subquestion."""
        scored = []
        for header in headers:
            parts = [norm(part) for part in header.split("|")]
            parts = [part for part in parts if part]
            matched = [part for part in parts if part in subq_norm]
            # Prefer more matched path components, then more matched text.
            score = (len(matched), sum(len(part) for part in matched))
            if score[0] > 0:
                scored.append((score, header))
        if not scored:
            return []
        best_score = max(score for score, _ in scored)
        return [header for score, header in scored if score == best_score]

    only_one_subq = len(ctx.atomic_subquestions) == 1
    for subq_id, subq_text in ctx.atomic_subquestions.items():
        linked = ctx.atomic_subquestion_mapping.setdefault(
            subq_id, {"relevant_row_headers": [], "relevant_column_headers": []}
        )
        subq_norm = norm(subq_text)

        matching_rows = best_path_matches(rows, subq_norm)
        matching_cols = best_path_matches(cols, subq_norm)

        # Row paths usually identify separate operands, so distribute them by
        # subquestion text. Column paths usually describe the common measure
        # (e.g. rate | total) and should constrain every operand.
        if rows and (matching_rows or only_one_subq):
            linked["relevant_row_headers"] = matching_rows or list(rows)
        if cols:
            linked["relevant_column_headers"] = matching_cols or list(cols)


def schema_linking(
    ctx, model_name, grounding_by_phrase=None, prefer_latest_column=False,
    canonical_schema=None,
):
    prompt = schema_linking_prompt.format(
        subquestions=json.dumps(ctx.atomic_subquestions, ensure_ascii=False),
        table_metadata=json.dumps(ctx.table_meta, ensure_ascii=False),
        grounding_hints=json.dumps(grounding_by_phrase or {}, ensure_ascii=False),
    )

    max_retries = 3
    input_tokens, output_tokens = 0, 0
    output = ""

    for attempt in range(max_retries):
        try:
            print(f"[INFO] Attempt {attempt + 1}/{max_retries} for schema linking.")
            all_output = llm_generate(prompt, model=model_name, json_format=True)
            output = all_output["text"]
            input_tokens, output_tokens = all_output["input_tokens"], all_output["output_tokens"]

            subq_matched_schema = safe_json_loads(output)
            ctx.atomic_subquestion_mapping = subq_matched_schema
            break
        except Exception as e:
            print(output)
            print("--errr", e)

    # Merge soft hints first, then let the candidate-specific canonical schema
    # define the hard scope. HiTab's no-year convention is applied last so a
    # canonical parent such as `current $millions` cannot re-expand all years.
    _merge_grounding_hints(ctx, grounding_by_phrase, False)
    _apply_canonical_schema(ctx, canonical_schema)
    if prefer_latest_column:
        _merge_grounding_hints(ctx, {}, True)
    return input_tokens, output_tokens


def program_compose(ctx, subq_relevant_data, model_name, numerical_reasoning_context=""):
    prompt = program_generation_prompt.format(
        table=ctx.table_values,
        atomic_subquestions=json.dumps(ctx.atomic_subquestions, ensure_ascii=False),
        atomic_subquestions_subdata=json.dumps(subq_relevant_data, ensure_ascii=False),
        subquestions=json.dumps(
            {k: v for k, v in ctx.subquestions.items() if k not in ctx.atomic_subquestions.keys()}, ensure_ascii=False
        ),
        reasoning_history=ctx.reasoning_history,
        numerical_reasoning_context=numerical_reasoning_context,
    )

    all_output = llm_generate(prompt, model=model_name)

    output = all_output["text"]
    input_tokens, output_tokens = all_output["input_tokens"], all_output["output_tokens"]

    generated_python_code = extract_python_code(output)
    final_ans = run_extracted_code(generated_python_code)
    return generated_python_code, final_ans, input_tokens, output_tokens


def make_json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj.item()
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    return str(obj)


def _candidate_rule_score(candidate, ctx_dict, result_data, original_question=""):
    """Label-free sanity score used before/fallback for candidate verification."""
    score = 0
    reasons = []
    answer = result_data.get("final_answer")
    if isinstance(answer, bool) or isinstance(answer, (dict, list, tuple, set)):
        reasons.append("hard_invalid_numeric_answer_type")
    elif answer is not None and not (
        isinstance(answer, str) and ("error" in answer.lower() or not answer.strip())
    ):
        score += 4
    else:
        reasons.append("missing_or_execution_error")

    schema = candidate.get("relevant_schema", {}) or {}
    if schema.get("row_headers"):
        score += 1
    if schema.get("column_headers"):
        score += 1

    q_norm = " ".join(_normalize_text(original_question))
    evidence_paths = _schema_and_linking_paths(candidate, ctx_dict)

    # Explicit endpoints are hard query constraints.  A candidate may leave its
    # proposed schema empty, so validate both proposed schema and actual linking.
    # We deliberately do not infer an answer here; this only checks coverage.
    requested_years = _explicit_years(original_question)
    if requested_years:
        missing_years = [
            year for year in requested_years
            if not any(re.search(rf"\b{re.escape(year)}\b", path) for path in evidence_paths)
        ]
        if missing_years:
            reasons.append(f"hard_missing_explicit_period:{missing_years}")
    by_match = re.search(r"\bby\s+([^,?]+)", str(original_question or ""), re.I)
    if by_match:
        axis_norm = " ".join(_normalize_text(by_match.group(1)))
        row_roots = {
            " ".join(_normalize_text(str(path).split("|")[0]))
            for path in schema.get("row_headers", []) or []
        }
        if axis_norm and row_roots:
            matching_roots = {root for root in row_roots if axis_norm in root or root in axis_norm}
            if not matching_roots:
                reasons.append(f"hard_missing_explicit_axis:{axis_norm}")
            elif row_roots - matching_roots:
                reasons.append(f"hard_competing_axis:{sorted(row_roots - matching_roots)}")

    if (
        re.search(r"\b(?:account(?:ed)? for|share of)\b", q_norm)
        and re.search(r"\b(?:total|all)\b", q_norm)
    ):
        paths = evidence_paths
        has_total = any(
            (str(path).split("|")[-1].strip().casefold()
             in {"total", "all", "all companies", "overall", "national total"})
            for path in paths
        )
        if not has_total:
            reasons.append("hard_missing_total_denominator")

    hint = candidate.get("operator_hint", "direct")
    reasoning = " ".join(ctx_dict.get("reasoning_history", [])).lower()
    program = str(result_data.get("generated_python_code", "") or "").lower()
    expected = {
        "proportion": ("proportion(", "division("),
        "difference": ("difference(",),
        "decline_magnitude": ("difference(", "opposite("),
        "difference_rate": ("difference_rate(",),
        "argmax": ("argmax(", "max("),
        "argmin": ("argmin(", "min("),
        "sum": ("sum(",),
        "average": ("average(",),
    }.get(hint)
    if expected:
        if any(token in reasoning or token in program for token in expected):
            score += 2
        else:
            reasons.append(f"operator_mismatch:{hint}")

    if hint == "decline_magnitude":
        asks_for_largest = bool(re.search(
            r"\b(?:largest|greatest|biggest|most)\b.*\b(?:declin|decreas|drop|fall)",
            str(original_question or ""), re.I,
        ))
        has_old_minus_new = bool(
            "difference(" in reasoning
            or re.search(r"(?:old|start|earlier)\s*[-–]\s*(?:new|end|later)", reasoning + program)
            or re.search(r"-\s*(?:new|end|later)", program)
        )
        if not has_old_minus_new:
            reasons.append("hard_wrong_decline_direction_or_missing_change")
        if asks_for_largest and not any(
            token in reasoning or token in program
            for token in ("argmax(", "idxmax(", ".max(", "max(")
        ):
            reasons.append("hard_missing_largest_decline_selection")
    return score, reasons


def _select_candidate_without_label(original_question, candidates, contexts, results, model_name):
    """Choose one executed candidate without exposing the ground-truth label."""
    scored = []
    for i, (candidate, ctx, result) in enumerate(zip(candidates, contexts, results)):
        score, reasons = _candidate_rule_score(
            candidate, ctx, result, original_question=original_question
        )
        scored.append({"index": i, "score": score, "reasons": reasons})

    # Prefer a table-provided change metric over reconstruction from rounded
    # raw rates when the question explicitly asks for decline magnitude.
    derived_magnitude = [
        i for i, (candidate, result) in enumerate(zip(candidates, results))
        if candidate.get("candidate_source") == "derived_metric_schema"
        and candidate.get("answer_transform") == "absolute_value"
        and isinstance(result.get("final_answer"), (int, float))
        and not isinstance(result.get("final_answer"), bool)
    ]
    if len(derived_magnitude) == 1:
        return derived_magnitude[0], {
            "method": "deterministic_derived_metric",
            "scores": scored,
            "reason": "Prefer the table's explicit derived change metric for a requested decline magnitude.",
        }

    # These candidates correspond to explicit table encodings and therefore
    # dominate a reconstructed answer from lower-level or reverse-conditioned
    # values. This preference is label-free.
    explicit_metric = [
        i for i, (candidate, result) in enumerate(zip(candidates, results))
        if candidate.get("candidate_source") in {
            "exact_period_schema", "interval_share_schema", "matrix_share_schema",
            "hierarchical_row_share_schema", "risk_reference_schema",
        }
        and isinstance(result.get("final_answer"), (int, float))
        and not isinstance(result.get("final_answer"), bool)
    ]
    if len(explicit_metric) == 1:
        return explicit_metric[0], {
            "method": "deterministic_explicit_metric",
            "scores": scored,
            "reason": "Prefer an explicitly encoded period/share interpretation over reconstruction.",
        }

    # `Among X, what percentage were Y` fixes the denominator to X. A direct
    # percentage column may encode the reverse conditioning (X among Y), so
    # prefer the candidate that actually computes Y-count / X-total.
    if re.search(r"\bamong\b.*\bpercentage of\b", str(original_question or ""), re.I):
        proportion_candidates = [
            i for i, (candidate, result) in enumerate(zip(candidates, results))
            if candidate.get("operator_hint") == "proportion"
            and isinstance(result.get("final_answer"), (int, float))
            and not isinstance(result.get("final_answer"), bool)
        ]
        if len(proportion_candidates) == 1:
            return proportion_candidates[0], {
                "method": "deterministic_conditioned_proportion",
                "scores": scored,
                "reason": "Use the explicitly conditioned population as the denominator.",
            }

    eligible = [
        x for x in scored
        if "missing_or_execution_error" not in x["reasons"]
        and "hard_invalid_numeric_answer_type" not in x["reasons"]
    ]
    viable = [
        x for x in eligible
        if not any(reason.startswith("hard_") for reason in x["reasons"])
    ]
    pool = viable or eligible or scored
    fallback_index = max(pool, key=lambda x: (x["score"], -x["index"]))["index"]
    if len(pool) <= 1:
        return fallback_index, {"method": "deterministic", "scores": scored}

    evidence = []
    for item in pool:
        i = item["index"]
        ctx = contexts[i]
        result = results[i]
        evidence.append({
            "index": i,
            "rewritten_question": candidates[i].get("rewritten_question"),
            "relevant_schema": candidates[i].get("relevant_schema"),
            "scope_decisions": candidates[i].get("scope_decisions"),
            "operator_hint": candidates[i].get("operator_hint"),
            "reasoning_history": ctx.get("reasoning_history", []),
            "schema_linking": ctx.get("schema_linking", {}),
            "final_answer": result.get("final_answer"),
            "generated_python_code": str(result.get("generated_python_code", ""))[:3000],
            "rule_score": item["score"],
            "rule_warnings": item["reasons"],
        })

    prompt = f"""You are a strict candidate verifier for table numerical reasoning.
Select exactly ONE candidate that best answers the original question. You must
not use or infer any ground-truth answer. Check preservation of explicit scope,
axis, entities, periods and units; exact schema relevance; operator semantics;
operand direction; and whether the program actually uses the linked evidence.
Prefer explicit wording in the original question over interpretations of vague
phrases. Reject candidates with irrelevant dimensions, missing operands, empty
values, execution errors, or formulas inconsistent with the question.
Every explicit year or range endpoint must be present in the linked evidence and
actually used by the program. For shares of a total, require a justified
numerator and the matching total denominator; never trust a percentage-looking
column merely because it is already normalized. For largest/greatest decrease,
require old-minus-new for every eligible entity followed by selection of the
largest positive decline; this is not the minimum raw value.

Original question: {original_question}
Candidates: {json.dumps(evidence, ensure_ascii=False)}

Return JSON only:
{{"selected_index": 0, "reason": "concise evidence-based reason"}}"""
    try:
        output = llm_generate(prompt, model=model_name, json_format=True)
        decision = safe_json_loads(output["text"])
        selected = int(decision.get("selected_index", fallback_index))
        allowed = {x["index"] for x in pool}
        if selected not in allowed:
            selected = fallback_index
        return selected, {
            "method": "llm_verifier",
            "scores": scored,
            "reason": decision.get("reason", ""),
        }
    except Exception as exc:
        return fallback_index, {
            "method": "deterministic_fallback",
            "scores": scored,
            "error": f"{type(exc).__name__}: {exc}",
        }


def process_qa_pair(
    opeartion_set,
    qa_pair: Dict[str, Any],
    table_meta_infos: Dict[str, Any],
    value_index_root: str,   # NEW
    out_dir: str,
    diamb_dir: str,
    overwrite_existing: bool,
    question_type: str,
    mode: str,
    model_name: str,
    rewrite_model_name: Optional[str] = None,
    use_embed_for_subtab: bool = False,
):
    qa_id = f"{question_type}_{qa_pair['id']}"
    case_dir = os.path.join(out_dir, qa_id)
    os.makedirs(case_dir, exist_ok=True)

    result_path = os.path.join(case_dir, "result.json")
    action_path = os.path.join(case_dir, "query_plan.txt")

    if (not overwrite_existing) and os.path.isfile(result_path):
        try:
            old = json.load(open(result_path, "r", encoding="utf-8"))
            if isinstance(old, list) and len(old) > 0:
                print(f"[SKIP] {qa_id} already processed → {result_path}")
                return
        except Exception:
            pass

    # raw table id (value_index is stored at raw level)
    raw_table_id = qa_pair["table_id"][0] if type(qa_pair["table_id"])==list else qa_pair["table_id"]

    value_index_dir = os.path.join(value_index_root, raw_table_id)
    if not os.path.isdir(value_index_dir):
        print(f"[ERROR] value_index_dir not found: {value_index_dir}")
        return

    question = qa_pair["query"]

    # ---- pick best subtable from layered_tree.json ----
    try:
        layered_tree = _load_layered_tree(value_index_dir)
        subtables = _list_subtables_from_tree(layered_tree)
        best = choose_best_subtable(
            question,
            subtables,
            use_embed=use_embed_for_subtab,
            embed_fn_callable=embed_fn if use_embed_for_subtab else None,
        )
        if best is None:
            print(f"[ERROR] No subtable found in layered_tree for raw={raw_table_id}")
            return
        chosen_subtab_id = best["subtable_id"]
        chosen_subtab_title = best.get("title", "")
    except Exception as e:
        print(f"[ERROR] Failed to select subtable for {qa_id}: {e}")
        return

    # ---- load meta for the chosen subtable id ----
    if chosen_subtab_id not in table_meta_infos:
        print(f"[ERROR] Missing meta for chosen subtable: {chosen_subtab_id}")
        return
    meta = table_meta_infos[chosen_subtab_id]
    meta_for_ctx = {
        "column_headers": meta[0].get("column_headers", []),
        "row_headers": meta[0].get("row_headers", []),
    }

    # ---- Load full df for that chosen subtable ----
    try:
        df = build_df_for_subtable(value_index_dir, chosen_subtab_id)

        print(f"[START] Processing {qa_id} (raw={raw_table_id}, chosen_subtab={chosen_subtab_id}, title={chosen_subtab_title})")
    except Exception as e:
        print(f"[ERROR] Failed to build df for {qa_id}: {e}")
        return

    # ---- Candidate rewrites (new format) with single-candidate fallback ----
    rewrite_model_name = rewrite_model_name or model_name
    disamb_path = os.path.join(diamb_dir, rewrite_model_name, f"{qa_pair['id']}.jsonl")
    grounding_by_phrase = {}
    rewrite_candidates = []
    use_rewrite = False
    if os.path.exists(disamb_path):
        try:
            with open(disamb_path, "r", encoding="utf-8") as f:
                disamb_candidates = json.load(f)
            if isinstance(disamb_candidates, dict):
                grounding_by_phrase = disamb_candidates.get("grounding_by_phrase", {}) or {}
                use_rewrite = bool(disamb_candidates.get(
                    "use_rewrite", disamb_candidates.get("ambiguity", False)
                ))
                disamb_rewrites = disamb_candidates.get("candidates", []) or []
                if not disamb_rewrites:
                    disamb_rewrites = [{
                        "rewritten_question": disamb_candidates.get("rewritten_question", ""),
                        "relevant_schema": disamb_candidates.get("relevant_schema", {}) or {},
                        "scope_decisions": disamb_candidates.get("scope_decisions", {}) or {},
                        "operator_hint": disamb_candidates.get("operator_hint", ""),
                    }]
                if use_rewrite:
                    rewrite_candidates = disamb_rewrites
        except Exception:
            rewrite_candidates = []

    # The unmodified question over the full schema is always candidate 0.
    # Rewriting/pruning supplies optional alternatives rather than replacing
    # the raw interpretation.  Keep at most two rewritten alternatives so the
    # existing three-candidate execution budget is unchanged.
    raw_candidate = {
        "rewritten_question": question,
        "relevant_schema": {},
        "scope_decisions": {},
        "operator_hint": _infer_raw_operator_hint(question),
        "candidate_source": "raw_full_schema",
        "answer_transform": _infer_answer_transform(question),
    }
    optional_rewrites = []
    seen_rewrites = {(question.casefold().strip(), (), (), "")}
    for candidate in rewrite_candidates:
        if not isinstance(candidate, dict):
            continue
        schema = candidate.get("relevant_schema", {}) or {}
        signature = (
            str(candidate.get("rewritten_question", "") or "").casefold().strip(),
            tuple(schema.get("row_headers", []) or []),
            tuple(schema.get("column_headers", []) or []),
            str(candidate.get("operator_hint", "") or ""),
        )
        if signature in seen_rewrites:
            continue
        seen_rewrites.add(signature)
        normalized = dict(candidate)
        normalized.setdefault("candidate_source", "rewritten_pruned_schema")
        optional_rewrites.append(normalized)
    semantic_candidates = [
        _exact_period_candidate(question, meta_for_ctx),
        _matrix_share_candidate(question, meta_for_ctx),
        _interval_share_candidate(question, meta_for_ctx),
        _risk_reference_candidate(question, meta_for_ctx),
        _derived_metric_candidate(question, meta_for_ctx),
    ]
    alternatives = [c for c in semantic_candidates if c] + optional_rewrites
    rewrite_candidates = [raw_candidate] + alternatives[:2]

    # ---- Operation set ----
    current_operation_set = opeartion_set.copy()
    if len(opeartion_set[0].strip()) == 0:
        current_operation_set[0] = qa_pair.get("aggregation", "")

    # ---- Execute each interpretation independently, then select without label ----
    executed_candidates, contexts, results = [], [], []
    for candidate_index, candidate in enumerate(rewrite_candidates[:3]):
        qa_pair_local = dict(qa_pair)
        rewritten_q = str(candidate.get("rewritten_question", "") or "").strip()
        if rewritten_q:
            qa_pair_local["query"] = _usable_rewritten_question(question, rewritten_q)
        is_raw_candidate = candidate.get("candidate_source") == "raw_full_schema"
        qa_pair_local["_grounding_by_phrase"] = {} if is_raw_candidate else grounding_by_phrase
        qa_pair_local["_canonical_schema"] = candidate.get("relevant_schema", {}) or {}
        qa_pair_local["_scope_decisions"] = candidate.get("scope_decisions", {}) or {}
        qa_pair_local["_operator_hint"] = str(candidate.get("operator_hint", "") or "")
        normalized_candidate = dict(candidate)
        normalized_candidate["rewritten_question"] = qa_pair_local["query"]
        try:
            ctx_dict, result_data = process_single_res(
                current_operation_set, qa_pair_local, qa_id, meta_for_ctx, df,
                value_index_dir, chosen_subtab_id, mode, model_name, max_retries=3,
            )
            if ctx_dict is None or result_data is None:
                raise RuntimeError("candidate execution returned no result")
            if candidate.get("answer_transform") == "absolute_value":
                answer = result_data.get("final_answer")
                if isinstance(answer, (int, float)) and not isinstance(answer, bool):
                    result_data["raw_final_answer"] = answer
                    result_data["final_answer"] = abs(answer)
        except Exception as exc:
            ctx_dict = {"error": f"{type(exc).__name__}: {exc}"}
            result_data = {
                "question_id": qa_id,
                "question": qa_pair_local["query"],
                "final_answer": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        normalized_candidate["candidate_index"] = candidate_index
        executed_candidates.append(normalized_candidate)
        contexts.append(ctx_dict)
        results.append(result_data)

    selected_index, verification = _select_candidate_without_label(
        question, executed_candidates, contexts, results, model_name
    )
    selected_result = dict(results[selected_index])
    selected_result["selected_candidate_index"] = selected_index
    selected_result["verification"] = verification

    with open(action_path, "w", encoding="utf-8") as f:
        json.dump(contexts, f, indent=4, ensure_ascii=False)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump([selected_result], f, indent=4, ensure_ascii=False)

    with open(os.path.join(case_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump({
            "original_question": question,
            "selected_index": selected_index,
            "verification": verification,
            "candidates": executed_candidates,
            "contexts": contexts,
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"[DONE] Results saved to {result_path}")


def process_single_res(
    opeartion_set,
    qa_pair,
    qa_id,
    meta_for_ctx,
    df,
    value_index_dir: str,      # NEW
    chosen_subtab_id: str,     # NEW
    mode,
    model_name,
    max_retries=3,
):
    step1_in, step1_out = 0, 0
    question = qa_pair["query"]

    # ---- decomposition ----
    ctx = _deterministic_total_share_context(
        question,
        meta_for_ctx,
        df,
        qa_pair.get("_canonical_schema", {}),
        qa_pair.get("_operator_hint", ""),
    )
    if ctx is None:
        ctx = _deterministic_matrix_share_context(
            question,
            meta_for_ctx,
            df,
            qa_pair.get("_canonical_schema", {}),
            qa_pair.get("_operator_hint", ""),
        )
    if (
        ctx is None
        and qa_pair.get("_operator_hint") == "direct"
        and qa_pair.get("_canonical_schema", {}).get("row_headers")
        and qa_pair.get("_canonical_schema", {}).get("column_headers")
    ):
        # A canonical single-cell/slice candidate needs no LLM decomposition;
        # decomposition can invent a second operand and corrupt a direct read.
        ctx = init_context(question, meta_for_ctx, df)
    if ctx is not None:
        cost_time_decomp = 0
    elif mode == "remove_all":
        ctx, _, _ = question_decomposition(
            opeartion_set, question, meta_for_ctx, df, model_name, mode=mode, max_steps=6, skip=True
        )
        cost_time_decomp = 0
    else:
        cost_time_decomp = 0
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Attempt {attempt + 1}/{max_retries} for question decomposition.")
                start = time.time()

                for q_attempt in range(3):
                    ctx, cur_in, cur_out = question_decomposition(
                        opeartion_set,
                        question,
                        meta_for_ctx,
                        df,
                        model_name,
                        mode=mode,
                        max_steps=6,
                        operator_hint=qa_pair.get("_operator_hint", ""),
                    )
                    if ctx is not None:
                        break
                    cur_in, cur_out = 0, 0
                    start = time.time()
                    if q_attempt == 2:
                        return None, None

                step1_in += cur_in
                step1_out += cur_out
                cost_time_decomp = time.time() - start
                print(f"[INFO] Decomposition successful on attempt {attempt + 1}.")
                break
            except Exception:
                continue

    # ---- schema linking ----
    start_time_linking = time.time()
    step2_in, step2_out = 0, 0
   
    step2_in, step2_out = schema_linking(
        ctx,
        model_name,
        grounding_by_phrase=qa_pair.get("_grounding_by_phrase", {}),
        prefer_latest_column=qa_pair.get("_dataset_name") == "hitab_num",
        canonical_schema=qa_pair.get("_canonical_schema", {}),
    )
    cost_time_linking = time.time() - start_time_linking
   

    ctx_dict = {
        "subquestions": ctx.subquestions,
        "atomic_subquestions": ctx.atomic_subquestions,
        "operation_history": ctx.operation_history,
        "reasoning_history": ctx.reasoning_history,
        "last_operation": ctx.last_operation,
        "schema_linking": ctx.atomic_subquestion_mapping,
        "canonical_schema": qa_pair.get("_canonical_schema", {}),
        "scope_decisions": qa_pair.get("_scope_decisions", {}),
        "operator_hint": qa_pair.get("_operator_hint", ""),
    }

    # ---- build subdata for program generation (NOW via value_index + chosen subtable) ----

    atomic_subquestion_subdata = {}
    for subq, relevant_info in ctx.atomic_subquestion_mapping.items():
        if "relevant_row_headers" not in relevant_info or "relevant_column_headers" not in relevant_info:
            continue
        atomic_subquestion_subdata[subq] = build_df_for_subtable(
            value_index_dir,
            chosen_subtab_id,
            relevant_row_headers=relevant_info["relevant_row_headers"],
            relevant_column_headers=relevant_info["relevant_column_headers"],
        ).to_string(index=True)

    # ---- compose program ----
    start_time_compose = time.time()
    numerical_reasoning_context = qa_pair.get("aggregation", [])
    generated_python_code, final_ans, step3_in, step3_out = program_compose(
        ctx, atomic_subquestion_subdata, model_name, numerical_reasoning_context
    )
    cost_time_compose = time.time() - start_time_compose

    result_data = {
        "question_id": qa_id,
        "raw_table_id": qa_pair["table_id"][0] if type(qa_pair["table_id"])==list else qa_pair["table_id"],
        "chosen_subtable_id": chosen_subtab_id,
        "question": question,
        "label": qa_pair.get("label"),
        "final_answer": make_json_safe(final_ans),
        "generated_python_code": generated_python_code,
        "time_cost": {
            "decomposition": cost_time_decomp,
            "composition": cost_time_compose,
            "grounding": cost_time_linking,
            "total": cost_time_decomp + cost_time_compose + cost_time_linking,
        },
        "token_cost": {
            "decomposition": [step1_in, step1_out],
            "composition": [step3_in, step3_out],
            "grounding": [step2_in, step2_out],
            "total": [step1_in + step2_in + step3_in, step1_out + step2_out + step3_out],
        },
    }

    return ctx_dict, result_data


def run_multihiertt_benchmark(
    opeartion_set, mode, model_name, rewrite_model_name, cfg, args,
    aug_mode=False, overwrite_existing=True,
):
    base_dir = cfg["base_folder"]
    out_folder = cfg["out_folder"]
    ds_cfg = cfg["datasets"][args.dataset]

    question_type = "single_tab"
    qa_path = os.path.join(base_dir, args.dataset, ds_cfg["qa_file"] if not aug_mode else ds_cfg["aug_qa_file"])
    diamb_dir = os.path.join(out_folder, args.dataset, ds_cfg["diamb_dir"])
    out_dir = os.path.join(out_folder, args.dataset, ds_cfg["result_dir"], f"runs_out_{mode}_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"🚀 [INFO] Starting Benchmark: {args.dataset}")

    value_index_root = os.path.join(out_folder, args.dataset, "our/value_index")
    table_meta_infos,_ = load_table_meta_from_layered_tree(value_index_root)
    
    
    print(f"[INFO] Loaded meta for {len(table_meta_infos)} tables")

    print(f"[INFO] Loading QA pairs from: {qa_path}")
    qa_pairs = read_jsonl(qa_path)

    if args.retry_failed:
        failed_ids = set()
        for qa in qa_pairs:
            qa_id = f"{question_type}_{qa.get('id')}"
            result_path = os.path.join(out_dir, qa_id, "result.json")
            try:
                old_result = json.load(open(result_path, "r", encoding="utf-8"))
                record = old_result[0] if isinstance(old_result, list) and old_result else {}
                if isinstance(record, dict) and record.get("error"):
                    failed_ids.add(str(qa.get("id")))
            except Exception:
                continue
        qa_pairs = [qa for qa in qa_pairs if str(qa.get("id")) in failed_ids]
        print(f"[Filtering] Selected {len(qa_pairs)} previously failed QA pairs")

    if args.qa_rids:
        requested_ids = {str(x) for x in args.qa_rids}
        qa_pairs = [qa for qa in qa_pairs if str(qa.get("id")) in requested_ids]
        found_ids = {str(qa.get("id")) for qa in qa_pairs}
        missing_ids = sorted(requested_ids - found_ids)
        print(f"[Filtering] Selected {len(qa_pairs)} QA pairs by --qa_rids")
        if missing_ids:
            print(f"[WARNING] QA ids not found: {missing_ids}")

    if aug_mode:
        qa_pairs = [x for x in qa_pairs if x.get("llm") == model_name]
        print(f"[Filtering] Resulting {len(qa_pairs)} QA pairs")

    # NOTE:
    # 这里不再 clean_qa_pairs 去改 table_id 为 subtab
    # 因为我们在 process_qa_pair 里会用 layered_tree.json 选择 best_subtable

    valid_qa_pairs: List[Dict[str, Any]] = []
    print("[INFO] Filtering QA pairs...")

    for qa in tqdm(qa_pairs, total=len(qa_pairs)):
        qa["_dataset_name"] = args.dataset
        tids = qa.get("table_id")
        if isinstance(tids, str):
            tids = [tids]
        if not isinstance(tids, list) or not tids:
            continue

        raw_tid = tids[0]
        # 必须存在 value_index/raw_tid
        if not os.path.isdir(os.path.join(value_index_root, raw_tid)):
            continue

        # label must be numeric (your original constraint)
        try:
            float(qa.get("label"))
        except Exception:
            continue

        valid_qa_pairs.append(qa)

    if not valid_qa_pairs:
        raise RuntimeError("No valid QA pairs found after filtering. Adjust filters.")

    print(f"✅ [INFO] Found {len(valid_qa_pairs)} valid QA pairs after filtering.")

    failed_cases = []
    for qa_pair in tqdm(valid_qa_pairs, total=len(valid_qa_pairs)):
        try:
            process_qa_pair(
                opeartion_set,
                qa_pair=qa_pair,
                table_meta_infos=table_meta_infos,
                value_index_root=value_index_root,
                out_dir=out_dir,
                diamb_dir=diamb_dir,
                overwrite_existing=overwrite_existing,
                question_type=question_type,
                mode=mode,
                model_name=model_name,
                rewrite_model_name=rewrite_model_name,
                use_embed_for_subtab=False,  # 你要 embedding 就改 True
            )
        except Exception as exc:
            qa_id = f"{question_type}_{qa_pair.get('id')}"
            failed_cases.append(qa_pair.get("id"))
            print(f"[ERROR] Failed {qa_id}: {type(exc).__name__}: {exc}")

            # Replace a possibly stale successful result.  Evaluation will
            # count this case as incorrect instead of silently reusing output
            # from an older run.
            case_dir = os.path.join(out_dir, qa_id)
            os.makedirs(case_dir, exist_ok=True)
            failure_record = [{
                "question_id": qa_id,
                "final_answer": None,
                "error": f"{type(exc).__name__}: {exc}",
            }]
            with open(os.path.join(case_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(failure_record, f, indent=4, ensure_ascii=False)

    if failed_cases:
        print(f"[WARNING] Failed QA pairs ({len(failed_cases)}): {failed_cases}")

    print("---", failed_cases)


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument(
        "--mode",
        type=str,
        default="raw",
        choices=["raw"],
    )
    parser.add_argument("--model", type=str, default="GPT-4o")
    parser.add_argument(
        "--rewrite_model",
        type=str,
        default="GPT-4o",
        help="Model name whose question-rewriting outputs should be loaded.",
    )
    parser.add_argument("--aug", action="store_true")
    parser.add_argument(
        "--qa_rids",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=[],
        help="Comma-separated QA ids to process; omit to run the full dataset.",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Process only QA pairs whose existing result.json contains an error.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()
    model_name = MODEL_MAP[args.model]
    rewrite_model_name = MODEL_MAP[args.rewrite_model]

    with open("../config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    if "hitab_num" in args.dataset:
        opeartion_set = [
            """
        - filter_tree, filter_level
        - sum, average, count, max, min
        - argmax, argmin, kth-argmax, kth-argmin, pair-argmax, pair-argmin
        - difference, proportion, difference_rate, opposite
        - greater_than, less_than, eq, not_eq
        """,
            """
        subquestions: {"raw": "What is the maximum sales in any region in 2020?"}
        table metadata: {"row_headers": ["Region A", "Region B", "Region C"], "column_headers": ["2020 Sales"]}
        action history: []
        Function:
        infer_calculation_formula(
            "raw",
            "max(#1, #2, #3)",
            {
            "#1": "Region A, 2020 Sales",
            "#2": "Region B, 2020 Sales",
            "#3": "Region C, 2020 Sales"
            }
        )
        Explanation: Use the max operator to select the largest value among the regions.
        """,
        ]
    elif "multihiertt_num" in args.dataset:
        opeartion_set = [
            """
        - Add (+), Subtract (−), Multiply (×), Divide (÷), Exp (^)
        """,
            """
        subquestions: {"raw": "What is the sum of revenue in 2018 and 2019?"}
        table metadata: {"column_headers": ["2017 Revenue", "2018 Revenue", "2019 Revenue"]}
        action history: []
        Function:
        infer_calculation_formula(
            "raw",
            "#1 + #2",
            {
            "#1": "2018 Revenue",
            "#2": "2019 Revenue"
            }
        )
        Explanation: The answer requires adding revenues from 2018 and 2019. Use + from the Operator Set.
        """,
        ]
    else:
        opeartion_set = [
            """
        Operator Set:
        - sum, count, sort, argmax, argmin, max, min
        - diff, average, divide, multiply, add, subtract
        - comparison (greater_than, less_than, equal)
        """,
            """
        subquestions: {"raw": "What is the difference in population between 2021 and 2019?"}
        table metadata: {"column_headers": ["2019 Population", "2021 Population"]}
        action history: []
        Function:
        infer_calculation_formula(
            "raw",
            "#2 - #1",
            {
            "#1": "2019 Population",
            "#2": "2021 Population"
            }
        )
        Explanation: Difference is calculated using the diff (-) operator from the Operator Set.
        """,
        ]

    run_multihiertt_benchmark(
        opeartion_set, args.mode, model_name, rewrite_model_name, cfg, args,
        aug_mode=args.aug, overwrite_existing=True
    )

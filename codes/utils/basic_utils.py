import re
import json
from typing import List, Dict, Any, Iterable, Optional
import os
import pickle
import time
import tqdm
from tqdm import tqdm
import traceback
import math
import statistics
import re, json
import numpy as np
from sentence_transformers import SentenceTransformer
import numpy as np
import json, ast, re

_VALID_ESC = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})')

def _escape_invalid_backslashes(s: str) -> str:
    marker = "\u0000"
    s_marked = _VALID_ESC.sub(lambda m: marker + m.group(0)[1:], s)
    s_marked = s_marked.replace("\\", "\\\\")
    s_fixed = re.sub(marker + r'(["\\/bfnrt])', r'\\\1', s_marked)
    s_fixed = re.sub(marker + r'u([0-9a-fA-F]{4})', r'\\u\1', s_fixed)
    return s_fixed

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _repair_common_quote_bugs(s: str) -> str:
    """
    Repairs common LLM mistakes:
      A) "X" in "Y"  -> "X in Y"
      B) "X" for "Y" -> "X for Y"
      C) Remove stray quote before ' in ' / ' for ' if it's not the full pattern
    """
    s2 = s

    # A/B) Merge the full pattern: "X" in "Y"  OR  "X" for "Y"
    # Works well for your example.
    s2 = re.sub(
        r'"([^"]+)"\s+(in|for)\s+"([^"]+)"',
        lambda m: '"' + m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) + '"',
        s2
    )

    # C) If there's still a dangling quote right before in/for (partial cases), remove it
    s2 = re.sub(r'"\s+(for|in)\s+', r' \1 ', s2)

    return s2

def safe_json_loads(s: str):
    raw = s
    s = _strip_code_fences(s)

    # 1) strict JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) invalid backslashes
    try:
        return json.loads(_escape_invalid_backslashes(s))
    except json.JSONDecodeError:
        pass

    # 3) quote repairs (+ retry)
    s_repaired = _repair_common_quote_bugs(s)
    try:
        return json.loads(s_repaired)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_escape_invalid_backslashes(s_repaired))
    except json.JSONDecodeError:
        pass

    # 4) python literal fallback (trusted LLM output only)
    try:
        return ast.literal_eval(s_repaired)
    except Exception as e:
        raise json.JSONDecodeError(
            f"Failed to parse even after sanitization: {e}\n"
            f"RAW:\n{raw}\n\nSTRIPPED:\n{s}\n\nREPAIRED:\n{s_repaired}",
            s_repaired,
            0,
        )

def parse_llm_dict(text: str):
    if not text:
        raise ValueError("Empty LLM output text")

    s = text.strip()

    # 1) Remove ```json ... ``` fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()

    # 2) Try strict JSON first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 3) Fallback: sometimes models output Python dict with single quotes
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 4) Last resort: extract the first {...} block and parse it
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    raise ValueError(f"Cannot parse LLM output as dict. Head={text[:200]!r}")


# Merge "string1" + "string2" into "string1string2"
def clean_json_concatenation(s):
    return re.sub(r'"([^"]*)"\s*\+\s*"([^"]*)"', lambda m: f'"{m.group(1)}{m.group(2)}"', s)


def extract_json_from_text(text: str) -> Any:
    """
    Robustly extract a JSON object/array from arbitrary LLM output.

    Handles:
      - Leading/trailing text
      - ```json ... ``` or ``` fences
      - Newlines inside string literals
      - Invalid backslash escapes inside string literals (e.g. \s, \')
    Returns:
      Python object (dict or list).
    Raises:
      ValueError / json.JSONDecodeError if it really can't be parsed.
    """
    s = text.strip()

    # 1) Strip ``` fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        # drop first line (``` or ```json)
        lines = lines[1:]
        # drop last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # 2) Extract the first JSON-looking block starting with { or [
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        raise ValueError("No JSON object or array start ('{' or '[') found in text.")

    # Try to find the matching closing brace/bracket using a simple stack
    stack = []
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opening = stack.pop()
            if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                # mismatched but keep scanning
                continue
            if not stack:
                end = i + 1
                break

    if end is None:
        candidate = s[start:]
    else:
        candidate = s[start:end]

    # 3) Sanitize JSON-like string: handle newlines + invalid backslash escapes
    def _sanitize_json_like(raw: str) -> str:
        out = []
        in_string = False
        i = 0
        n = len(raw)
        # Valid escape chars in JSON strings
        valid_escapes = set(['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'])

        while i < n:
            ch = raw[i]

            if not in_string:
                out.append(ch)
                if ch == '"':
                    in_string = True
                i += 1
            else:
                # Inside a string literal
                if ch == '\\':
                    if i + 1 < n:
                        nxt = raw[i + 1]
                        if nxt in valid_escapes:
                            # Keep valid escape as is: \n, \", \u, etc.
                            out.append('\\')
                            out.append(nxt)
                            i += 2
                        else:
                            # Invalid escape (e.g. \s, \(), turn it into \\s so JSON sees
                            # literal backslash + char.
                            out.append('\\')
                            out.append('\\')
                            out.append(nxt)
                            i += 2
                    else:
                        # Trailing backslash at end of string → make it literal "\\"
                        out.append('\\')
                        out.append('\\')
                        i += 1
                elif ch == '"':
                    # End of string
                    out.append(ch)
                    in_string = False
                    i += 1
                elif ch in ('\n', '\r'):
                    # Newlines not allowed in JSON strings → replace with space
                    out.append(' ')
                    i += 1
                else:
                    out.append(ch)
                    i += 1

        return ''.join(out)

    print('----', candidate)
    try:
        return json.loads(candidate)
    except:
        import ast
        return ast.literal_eval(candidate)



def embed_fn(texts: list[str]):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, normalize_embeddings=True)

def _cosine_sim(a, b):
    # a, b: 1D numpy arrays
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _choose_best_subtab(query: str,
                        candidate_subtab_ids,
                        raw_table_id: str | None = None,
                        embed_fn=None):
    """
    Choose best subtab for `query`.
    If `embed_fn` is provided, use embedding similarity on cleaned titles.
    Otherwise, fall back to token-overlap scoring.

    `candidate_subtab_ids` example:
        ["subtab_business-table06_Number of tables on this sheet:",
         "subtab_business-table06_Advertising",
         "subtab_business-table06_Sports Activities", ...]
    """
    if not candidate_subtab_ids:
        return None

    # ---- 1) Clean titles ----
    cleaned_titles = candidate_subtab_ids
    # [clean_subtab_title(sid, raw_table_id) for sid in candidate_subtab_ids]

    # ---- 2) Filter meta subtables first ----
    filtered_ids = []
    filtered_titles = []
    for sid, title in zip(candidate_subtab_ids, cleaned_titles):
        # if _is_meta_title(title):
        #     continue
        filtered_ids.append(sid)
        filtered_titles.append(title)

    # if everything was meta, fall back to using all
    if not filtered_ids:
        filtered_ids = candidate_subtab_ids
        filtered_titles = cleaned_titles

    # embed query + titles
    q_emb = embed_fn([query])[0]
    subt_embs = embed_fn(filtered_titles)

    best_id = None
    best_score = -1.0
    for sid, emb in zip(filtered_ids, subt_embs):
        score = _cosine_sim(q_emb, emb)
        if score > best_score:
            best_score = score
            best_id = sid

    if best_id is not None:
        return best_id



def clean_qa_pairs(qa_pairs, multi_subtab_ids, embed_fn=None):
    """
    Remap table_id for QA pairs to the most appropriate subtable.

    multi_subtab_ids: dict
        raw_table_id -> [subtab_id1, subtab_id2, ...]
    embed_fn: callable or None
        If provided, embed_fn(list[str]) -> list/array of vectors.
    """
    updated_qa_items = []

    for item in qa_pairs:
        raw_tid = item.get("table_id")
        if isinstance(raw_tid, list):
            raw_tid = raw_tid[0]

        query = item.get("query", "")

        if raw_tid in multi_subtab_ids:
            candidate_subtab_ids = multi_subtab_ids[raw_tid]
            if len(candidate_subtab_ids) > 1:
                best_subtab_id = _choose_best_subtab(
                    query,
                    candidate_subtab_ids,
                    raw_table_id=raw_tid,
                    embed_fn=embed_fn,
                )
                print(query, best_subtab_id)
            else:
                best_subtab_id = candidate_subtab_ids[0]
            item["table_id"] = [best_subtab_id] 
        else:
            item["table_id"] = [raw_tid]

        updated_qa_items.append(item)

    return updated_qa_items


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {file_path} at line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object per line in {file_path}, got {type(obj)} at line {lineno}")
            out.append(obj)
    return out


def write_jsonl(file_path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def analyze_llm_output(llm_output: str):
    # Define regex patterns for different actions
    # NOTE: The regex for multihop_question_decomposition has been adjusted.
    # It uses three capturing groups for the arguments, and the outer part now correctly
    # matches the full function signature up to the final closing parenthesis.
    patterns = {
        # "infer_calculation_formula": r'infer_calculation_formula\(\s*"([^"]+)",\s*"([^"]+)",\s*(\{.*?\})\s*\)\s*"',
        "infer_calculation_formula": r'infer_calculation_formula\(\s*"([^"]*)",\s*"([^"]*)",\s*(\{.*?\})\s*\)',
        "multihop_question_decomposition": r'multihop_question_decomposition\(\s*"([^"]+)",\s*\[(.*?)],\s*(\{.*?\}\s*)\)',
        # "generate_execute_program": r'generate_execute_program\(\s*(\{.*?\})\)\s*Explanation:\s*"([^"]+)"',
    }

    # Clean up the input string to match the simplified regex if it has extra quotes
    # For your specific example: multihop_question_decomposition(...)

    # print('-----------', llm_output, type(llm_output))
    if isinstance(llm_output, dict): llm_output = f'{llm_output["name"]}("{llm_output["parameters"]["subquestion_id"]}", "{llm_output["parameters"]["formula"].replace("(", "").replace(")", "").split("*")[0].strip()}", {json.dumps(llm_output["parameters"]["operands"], ensure_ascii=False)})'
    llm_output = llm_output.strip().strip('"') 

    # Iterate over each action type
    result = {}
    for action, pattern in patterns.items():
        # Use re.DOTALL to allow matching across newlines if the JSON/list is long
        match = re.search(pattern, llm_output, re.DOTALL)
        # print(action, match) # For debugging

        if match:
            # Extract the relevant parts based on the action
            if action == "infer_calculation_formula":
                # Assuming this pattern is correct for its expected format
                subq_id = match.group(1)
                formula = match.group(2)
                operands_str = match.group(3)

                
                # NOTE: The original code assumes a fourth group for 'explanation', 
                # but the regex only has 3. I am leaving the logic as is for now, 
                # but be aware of the group count mismatch in the original 'infer' regex.

                # print("---- operands_str ----")
                # print(repr(operands_str))

                try:
                    operands = safe_json_loads(operands_str)
                except:
                    operands = safe_json_loads(clean_json_concatenation(operands_str))
                
                result = {
                    "action": action,
                    "parameters": {
                        "subq_id": subq_id,
                        "formula": formula,
                        "subquestions": operands
                    },
                    # "explanation": explanation # Not clearly captured by original regex
                }
                break
                
            elif action == "multihop_question_decomposition":
                subq_id = match.group(1).strip()
                
                # The 'order' argument is a list string like '\"#1\",\"#2\"'
                order_str = match.group(2).strip()
                
                # Convert the order string ['#1', '#2'] to a list ['#1', '#2']
                # Step 1: Remove quotes and split by comma
                order = [item.strip().strip('"') for item in order_str.split(',') if item.strip()]

                operands_str = match.group(3).strip()

                
                # Correctly load the JSON object for subquestions
                try:
                    operands = json.loads(operands_str)
                except json.JSONDecodeError as e:
                    try:
                        operands = json.loads(clean_json_concatenation(operands_str))
                    except:
                        print(f"Error decoding subquestions JSON: {e}")
                        operands = {}

                
                
                # Assuming no explanation is captured in this simplified case
                # If an explanation is expected, the regex needs a fourth capturing group
                
                result = {
                    "action": action,
                    "parameters": {
                        "subq_id": subq_id,
                        "order": order,
                        "subquestions": operands
                    },
                    # "explanation": "" # Placeholder if no explanation is explicitly captured
                }
                break

    return result

def load_table_meta(meta_dir: str) -> Dict[str, Any]:
    table_meta_infos: Dict[str, Any] = {}
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"Meta directory not found: {meta_dir}")

    for name in os.listdir(meta_dir):
        if not name.endswith(".json"):
            continue
        table_id = os.path.splitext(name)[0]
        path = os.path.join(meta_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            col_grps = meta[0]["column_headers"]
            meaningless_mask = [str(c).startswith("Column ") for c in col_grps]
            meaningless_ratio = sum(meaningless_mask) / max(1, len(col_grps))
            if meaningless_ratio > 0.60:
                continue
        except Exception as e:
            print(f"[WARN] Failed to load meta for {table_id}: {e}")
            continue

        table_meta_infos[table_id] = meta

    return table_meta_infos

def load_table_meta_new(meta_dir: str) -> Dict[str, Any]:
    table_meta_infos: Dict[str, Any] = {}
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"Meta directory not found: {meta_dir}")

    for name in os.listdir(meta_dir):
        if not name.endswith(".json"):
            continue
        table_id = os.path.splitext(name)[0]
        path = os.path.join(meta_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                predicted_tab_meta = json.load(f)
            predicted_col_headers, predicted_row_headers = [], []

            try:
                for key, val in predicted_tab_meta['parsed_json'].items():
                    predicted_col_headers.extend([' '.join(_) for _ in val['column_header']])
                    predicted_row_headers.extend([' '.join(_) for _ in val['row_header']])
            except:
                print('error')
                pass

            updated_meta =  [{'column_headers':predicted_col_headers, "row_headers":predicted_row_headers}]
        except Exception as e:
            print(f"[WARN] Failed to load meta for {table_id}: {e}")
            continue

        table_meta_infos[table_id] = updated_meta

    return table_meta_infos


def load_table_meta_gt(meta_data_fpath) -> Dict[str, Any]:
    table_meta_infos: Dict[str, Any] = {}
    
    read_fcontent = json.load(open(meta_data_fpath, 'r'))

    for table_id, predicted_tab_meta in read_fcontent.items():
        predicted_col_headers, predicted_row_headers = [], []

        try:
            for key, val in predicted_tab_meta['table_meta'].items():
                predicted_col_headers.extend([' '.join(_) for _ in val['column_header']])
                predicted_row_headers.extend([' '.join(_) for _ in val['row_header']])
        except:
            print('error')
            pass

        updated_meta =  [{'column_headers':predicted_col_headers, "row_headers":predicted_row_headers}]

        table_meta_infos[table_id] = updated_meta

    return table_meta_infos

def load_multitab_mapping_jsonl(path: str) -> Dict[str, List[str]]:
    """
    line: {"raw_table_id":"...", "subtab_ids":[...]}
    """
    m: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw = rec.get("raw_table_id")
            subs = rec.get("subtab_ids", []) or []
            if raw:
                # stable dedup
                m[raw] = list(dict.fromkeys([s for s in subs if isinstance(s, str) and s.strip()]))
    return m

def load_table_meta_from_layered_tree(
    value_index_root: str,
    return_tree_map: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each subtable_id, return UNIQUE complete header paths from layered_tree
    nodes (for example ``large companies | 250-499``).  Parent paths are also
    retained so schema linking can select a group which the resolver later
    expands to its descendant leaves.

    table_meta_infos[subtable_id] = [{
        "subtable_titles": [...],   # typically [title]
        "column_headers": [...],    # unique labels of all colhdr nodes (leaf + non-leaf)
        "row_headers": [...],       # unique labels of all rowhdr nodes (leaf + non-leaf)
    }]

    This matches your downstream access:
      table_meta_infos[table_id][0]['column_headers']
      table_meta_infos[table_id][0]['row_headers']
    """
    table_meta_infos: Dict[str, List[Dict[str, Any]]] = {}
    layered_trees: Dict[str, Dict[str, Any]] = {}
    last_layered_tree: Dict[str, Any] = {}

    if not os.path.isdir(value_index_root):
        raise FileNotFoundError(f"value_index_root not found: {value_index_root}")

    def push_unique(lst: List[str], x: Any):
        if x is None:
            return
        s = str(x).strip()
        if not s:
            return
        if s not in lst:
            lst.append(s)

    for raw_id in os.listdir(value_index_root):
        raw_dir = os.path.join(value_index_root, raw_id)
        if not os.path.isdir(raw_dir):
            continue

        lt_path = os.path.join(raw_dir, "layered_tree.json")
        if not os.path.exists(lt_path):
            continue

        try:
            lt = json.load(open(lt_path, "r", encoding="utf-8"))
        except Exception:
            continue

        last_layered_tree = lt
        # A raw table may contain one or more subtable roots.  Keep both the
        # raw-table lookup and (below) subtable lookups so callers can select
        # the correct tree for each QA item.
        layered_trees[raw_id] = lt

        nodes = lt.get("nodes", [])
        edges = lt.get("edges", [])

        node_by_id = {n.get("id"): n for n in nodes if isinstance(n, dict) and n.get("id")}

        # parent map (prefer parent_id if present)
        parent_of: Dict[str, str] = {}
        has_parent_id = any(isinstance(n, dict) and "parent_id" in n for n in nodes)
        if has_parent_id:
            for n in nodes:
                nid = n.get("id")
                pid = n.get("parent_id")
                if nid and pid:
                    parent_of[nid] = pid
        else:
            for p, c in edges:
                if c not in parent_of:
                    parent_of[c] = p

        def find_subtable_ancestor(nid: str) -> Optional[str]:
            cur = nid
            seen = set()
            while cur and cur in node_by_id and cur not in seen:
                seen.add(cur)
                n = node_by_id[cur]
                if n.get("type") == "subtable":
                    return cur  # st::...
                cur = parent_of.get(cur)
            return None

        # collect subtables
        subtable_nodes = [n for n in nodes if isinstance(n, dict) and n.get("type") == "subtable"]
        stid2sid: Dict[str, str] = {}
        stid2title: Dict[str, str] = {}
        for st in subtable_nodes:
            st_node_id = st.get("id")
            sid = st.get("subtable_id") or st_node_id
            title = st.get("title") or sid
            if st_node_id:
                stid2sid[st_node_id] = sid
                stid2title[st_node_id] = title

        # per subtable label pools
        col_labels_by_st: Dict[str, List[str]] = {st_id: [] for st_id in stid2sid}
        row_labels_by_st: Dict[str, List[str]] = {st_id: [] for st_id in stid2sid}

        for n in nodes:
            if not isinstance(n, dict):
                continue
            t = n.get("type")
            if t not in ("colhdr", "rowhdr"):
                continue
            nid = n.get("id")
            if not nid:
                continue

            st_anc = find_subtable_ancestor(nid)
            if not st_anc:
                continue

            path = n.get("path")
            if isinstance(path, list) and path:
                lab = " | ".join(
                    str(part).strip() for part in path if str(part).strip()
                )
            else:
                lab = n.get("label", "")
            if t == "colhdr":
                push_unique(col_labels_by_st.setdefault(st_anc, []), lab)
            else:
                push_unique(row_labels_by_st.setdefault(st_anc, []), lab)

        # export keyed by subtable_id
        for st_node_id, sid in stid2sid.items():
            title = stid2title.get(st_node_id, sid)
            table_meta_infos[sid] = [{
                "subtable_titles": [title],
                "column_headers": col_labels_by_st.get(st_node_id, []),
                "row_headers": row_labels_by_st.get(st_node_id, []),
            }]

            layered_trees[sid] = lt

    if return_tree_map:
        return table_meta_infos, layered_trees
    return table_meta_infos, last_layered_tree


def parse_llm_json(text: str):
    # 1) If there is a ```json ... ``` block, extract its body
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if m:
        text = m.group(1)

    # 2) Keep only the largest {...} slice
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]

    # 3) Normalize quotes and remove trailing commas
    text = (text
            .replace("\u201c", '"').replace("\u201d", '"')  # smart double quotes → "
            .replace("\u2018", "'").replace("\u2019", "'")) # smart single quotes → '
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # trailing comma before } or ]
    return text



def extract_python_code(llm_output: str) -> str:
    """
    Extracts the Python code snippet from the LLM's output.
    Assumes the code is contained within the final ```python ... ``` block.
    """
    # Regex to find content between ```python and ```
    match = re.search(r"```python\n(.*?)\n```", llm_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        # The first capturing group (.*?) contains the code
        return match.group(1).strip()
    else:
        # Fallback or error handling if the format is not matched
        return ""

def run_extracted_code(python_code: str):
    """
    Executes the extracted Python code and returns the value of the 
    'final_answer' variable.
    """
    # Create a dictionary to execute the code within.
    # This keeps the namespace isolated and allows us to capture variables.
    exec_scope = {}
    
    try:
        # The exec() function executes the code string.
        # We use a mutable dictionary (exec_scope) to capture output variables.
        exec(python_code, exec_scope)
        
        # The prompt mandates the final answer is stored in 'final_answer'
        if 'final_answer' in exec_scope:
            return exec_scope['final_answer']
        else:
            return "Error: 'final_answer' variable not defined in the executed code."
            
    except Exception as e:
        return f"Execution Error: {e}"
    
def build_union_meta_for_raw(raw_id: str, raw2subtab: Dict[str, List[str]], table_meta_infos: Dict[str, List[Dict[str, Any]]]):
    """
    Union metadata across ALL subtables under a raw table.
    Output format matches prompt input:
      {"subtable_titles":[...], "column_headers":[...], "row_headers":[...]}
    """
    raw_ids = raw_id if isinstance(raw_id, list) else [raw_id]
    sub_ids = []

    for rid in raw_ids:
        # Single-table datasets such as HiTab use the raw table id directly
        # as the metadata key and therefore do not need a multitab mapping.
        if rid in table_meta_infos and rid not in sub_ids:
            sub_ids.append(rid)

        # Multi-subtable datasets map one raw table id to several subtable ids.
        for sid in raw2subtab.get(rid, []) or []:
            if sid not in sub_ids:
                sub_ids.append(sid)

    titles, cols, rows = [], [], []

    def push_unique(lst, x):
        if x is None:
            return
        s = str(x).strip()
        if not s:
            return
        if s not in lst:
            lst.append(s)

    for sid in sub_ids:
        if sid not in table_meta_infos:
            continue
        item = table_meta_infos[sid][0]
        for t in item.get("subtable_titles", []):
            push_unique(titles, t)
        for c in item.get("column_headers", []):
            push_unique(cols, c)
        for r in item.get("row_headers", []):
            push_unique(rows, r)

    return {"subtable_titles": titles, "column_headers": cols, "row_headers": rows}

if __name__=='__main__':
    # str_ = 'infer_calculation_formula("raw", "(#2 - #1) / #1", {"#1": "Entergy Arkansas in 2015: First Quarter", "#2": "Entergy Arkansas in 2016: First Quarter"})'
    # print(analyze_llm_output(str_))
    text = """
{
  "function": "multihop_question_decomposition(\"raw\", [\"#11\", \"#12\"], {\"#11\": \"What is the number of women reported with multiple sclerosis in 2011?\", \"#12\": \"What is the number of men reported with multiple sclerosis in 2011?\"})",
  "explanation": "The question requires finding the ratio of women to men with multiple sclerosis, which involves two separate data fetches before calculating the ratio."
}
    """

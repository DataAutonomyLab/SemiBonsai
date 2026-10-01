# -*- coding: utf-8 -*-
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set


# -----------------------------
# 0) Build indices from layered_tree.json
# -----------------------------
def index_layered_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    nodes = tree.get("nodes", [])
    id2n: Dict[str, Dict[str, Any]] = {n["id"]: n for n in nodes}
    parent: Dict[str, Optional[str]] = {n["id"]: n.get("parent_id") for n in nodes}
    children: Dict[str, List[str]] = {n["id"]: list(n.get("children_ids", [])) for n in nodes}

    label2colhdr_ids: Dict[str, List[str]] = {}
    label2rowhdr_ids: Dict[str, List[str]] = {}
    title2subtable_ids: Dict[str, List[str]] = {}

    for n in nodes:
        t = n.get("type")
        if t == "subtable":
            title = str(n.get("title", "")).strip()
            if title:
                title2subtable_ids.setdefault(title, []).append(n["id"])
        elif t == "colhdr":
            lab = str(n.get("label", "")).strip()
            if lab:
                label2colhdr_ids.setdefault(lab, []).append(n["id"])
        elif t == "rowhdr":
            lab = str(n.get("label", "")).strip()
            if lab:
                label2rowhdr_ids.setdefault(lab, []).append(n["id"])

    def subtable_ancestor(nid: str) -> Optional[str]:
        cur = nid
        while cur is not None:
            if id2n[cur]["type"] == "subtable":
                return cur
            cur = parent.get(cur)
        return None

    return {
        "id2n": id2n,
        "parent": parent,
        "children": children,
        "title2subtable_ids": title2subtable_ids,
        "label2colhdr_ids": label2colhdr_ids,
        "label2rowhdr_ids": label2rowhdr_ids,
        "subtable_ancestor": subtable_ancestor,
    }


# -----------------------------
# 1) Utilities
# -----------------------------
def descendants(nid: str, children: Dict[str, List[str]]) -> List[str]:
    stk = [nid]
    out = []
    while stk:
        x = stk.pop()
        out.append(x)
        stk.extend(children.get(x, []))
    return out


def leaf_descendants_in_same_subtable(
    nid: str,
    idx: Dict[str, Any],
    node_type: str,
    subtable_root: str,
) -> List[str]:
    id2n = idx["id2n"]
    children = idx["children"]
    sub_anc = idx["subtable_ancestor"]

    leaves = []
    for d in descendants(nid, children):
        dn = id2n[d]
        if dn.get("type") != node_type:
            continue
        if not dn.get("is_leaf", False):
            continue
        if sub_anc(d) != subtable_root:
            continue
        leaves.append(d)
    return leaves


def _push_unique_str(lst: List[str], x: Any) -> None:
    if x is None:
        return
    s = str(x).strip()
    if not s:
        return
    if s not in lst:
        lst.append(s)


def _node_text(node: Dict[str, Any], field: str) -> str:
    """Return a usable node label without stringifying JSON null as ``None``."""
    value = node.get(field)
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


# -----------------------------
# 2) Range phrase handler
# -----------------------------
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def detect_range_endpoints(phrase: str) -> Optional[Tuple[int, int]]:
    yrs = _YEAR_RE.findall(phrase)
    if len(yrs) >= 2:
        a, b = int(yrs[0]), int(yrs[1])
        if a <= b:
            return a, b
        return b, a
    return None


def expand_year_range_titles(start_year: int, end_year: int, available_titles: Set[str]) -> List[str]:
    out = []
    for y in range(start_year, end_year + 1):
        s = str(y)
        if s in available_titles:
            out.append(s)
    return out


# -----------------------------
# 3) Map LLM-selected LABELS -> candidate node ids (C(u))
# -----------------------------
def labels_to_candidates(
    phrase_groundings: List[Dict[str, Any]],
    idx: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    title2subtable_ids = idx["title2subtable_ids"]
    label2colhdr_ids = idx["label2colhdr_ids"]
    label2rowhdr_ids = idx["label2rowhdr_ids"]

    out: Dict[str, Dict[str, List[str]]] = {}

    for pg in phrase_groundings:
        phrase = str(pg.get("phrase", "")).strip()
        if not phrase:
            continue
        sel = pg.get("selected_nodes", {}) or {}
        cand = {"subtable_titles": [], "column_headers": [], "row_headers": []}

        for t in sel.get("subtable_titles", []) or []:
            tt = str(t).strip()
            cand["subtable_titles"].extend(title2subtable_ids.get(tt, []))

        for lab in sel.get("column_headers", []) or []:
            ll = str(lab).strip()
            cand["column_headers"].extend(label2colhdr_ids.get(ll, []))

        for lab in sel.get("row_headers", []) or []:
            ll = str(lab).strip()
            cand["row_headers"].extend(label2rowhdr_ids.get(ll, []))

        for k in cand:
            seen = set()
            new_list = []
            for x in cand[k]:
                if x not in seen:
                    seen.add(x)
                    new_list.append(x)
            cand[k] = new_list

        out[phrase] = cand

    return out


# -----------------------------
# 4) Component selection utilities
# -----------------------------
def build_phrase_candidates_by_component(
    phrase2cands: Dict[str, Dict[str, List[str]]],
    idx: Dict[str, Any],
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Dict[str, List[str]]]]]:
    id2n = idx["id2n"]
    sub_anc = idx["subtable_ancestor"]

    comp_cover: Dict[str, Set[str]] = {}
    phrase_nodes_by_comp: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    for phrase, cands in phrase2cands.items():
        phrase_nodes_by_comp[phrase] = {}
        for bucket in ("subtable_titles", "column_headers", "row_headers"):
            for nid in cands.get(bucket, []):
                comp = nid if id2n[nid]["type"] == "subtable" else sub_anc(nid)
                if comp is None:
                    continue
                phrase_nodes_by_comp[phrase].setdefault(
                    comp,
                    {"subtable_titles": [], "column_headers": [], "row_headers": []},
                )
                phrase_nodes_by_comp[phrase][comp][bucket].append(nid)
                comp_cover.setdefault(comp, set()).add(phrase)

    return comp_cover, phrase_nodes_by_comp


def greedy_set_cover_components(
    phrases: Set[str],
    comp_cover: Dict[str, Set[str]],
    required_components: Set[str],
) -> Tuple[Set[str], Set[str]]:
    uncovered = set(phrases)
    chosen = set(required_components)

    for c in list(required_components):
        uncovered -= comp_cover.get(c, set())

    while uncovered:
        best_c, best_gain = None, 0
        for c, covers in comp_cover.items():
            if c in chosen:
                continue
            gain = len(covers & uncovered)
            if gain > best_gain:
                best_c, best_gain = c, gain
        if best_c is None:
            break
        chosen.add(best_c)
        uncovered -= comp_cover.get(best_c, set())

    return chosen, uncovered


# -----------------------------
# 5) Steiner + lifting in component
# -----------------------------
def steiner_nodes_in_component(
    terminals: List[str],
    component_root: str,
    parent: Dict[str, Optional[str]],
) -> Set[str]:
    S: Set[str] = set()
    for t in terminals:
        cur = t
        while cur is not None:
            S.add(cur)
            if cur == component_root:
                break
            cur = parent.get(cur)
    return S


def steiner_leaves_in_component(steiner_nodes: Set[str], children: Dict[str, List[str]]) -> List[str]:
    leaves = []
    for u in steiner_nodes:
        has_child_in_steiner = any(v in steiner_nodes for v in children.get(u, []))
        if not has_child_in_steiner:
            leaves.append(u)
    return leaves


def lift_steiner_root_to_leaf_paths(
    component_root: str,
    steiner_nodes: Set[str],
    idx: Dict[str, Any],
) -> Set[str]:
    id2n = idx["id2n"]
    parent = idx["parent"]
    children = idx["children"]

    lifted: Set[str] = set()
    steiner_leaves = steiner_leaves_in_component(steiner_nodes, children)

    def add_chain_to_root(x: str, stop: str) -> None:
        cur = x
        while cur is not None:
            lifted.add(cur)
            if cur == stop:
                break
            cur = parent.get(cur)

    for x in steiner_leaves:
        xn = id2n[x]
        t = xn.get("type")

        if t in ("colhdr", "rowhdr") and xn.get("is_leaf", False):
            add_chain_to_root(x, component_root)
            continue

        if t in ("colhdr", "rowhdr") and not xn.get("is_leaf", True):
            leaf_ds = leaf_descendants_in_same_subtable(x, idx, t, component_root)
            if not leaf_ds:
                add_chain_to_root(x, component_root)
            else:
                for leaf in leaf_ds:
                    add_chain_to_root(leaf, component_root)
            continue

        add_chain_to_root(x, component_root)

    lifted |= set(steiner_nodes)
    return lifted


# -----------------------------
# 6) Enumerate explicit paths and stringify them per phrase
# -----------------------------
def node_path_to_subtable(leaf_id: str, idx: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    id2n = idx["id2n"]
    parent = idx["parent"]

    chain = []
    cur = leaf_id
    while cur is not None:
        chain.append(cur)
        if id2n[cur]["type"] == "subtable":
            break
        cur = parent.get(cur)

    if not chain or id2n[chain[-1]]["type"] != "subtable":
        return None, list(reversed(chain))

    chain = list(reversed(chain))
    return chain[0], chain


def enumerate_lifted_leaf_paths(lifted_nodes: Set[str], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Enumerate subtable->...->leaf paths where leaf is a colhdr/rowhdr leaf and leaf node is in lifted_nodes.
    Return a list of records:
      {type, leaf_path_id, node_ids, label_path, path_str}
    """
    id2n = idx["id2n"]

    out: List[Dict[str, Any]] = []
    for nid in lifted_nodes:
        n = id2n[nid]
        if n.get("type") not in ("colhdr", "rowhdr"):
            continue
        if not n.get("is_leaf", False):
            continue
        pid = str(n.get("path_id", "")).strip()
        if not pid:
            continue

        st, chain = node_path_to_subtable(nid, idx)
        if st is None:
            continue

        # keep the chain consistent with lifted subtree (still guarantees subtable+leaf)
        chain2 = []
        for x in chain:
            if x == chain[0] or x == chain[-1] or x in lifted_nodes:
                chain2.append(x)
        chain = chain2

        label_path: List[str] = []
        for x in chain:
            xn = id2n[x]
            if xn["type"] == "subtable":
                label_path.append(_node_text(xn, "title"))
            else:
                label_path.append(_node_text(xn, "label"))

        # sanitize empty labels
        label_path = [s for s in label_path if str(s).strip()]

        out.append(
            {
                "type": n["type"],  # colhdr / rowhdr
                "leaf_node_id": nid,
                "leaf_path_id": pid,
                "node_ids": chain,
                "label_path": label_path,
                # Preserve hierarchy boundaries. Downstream candidate
                # generation and schema linking can now distinguish a full
                # path from an accidental bag of labels.
                "path_str": " | ".join(label_path),
            }
        )

    # stable ordering
    out.sort(key=lambda r: (r["type"], r["path_str"], r["leaf_path_id"]))
    return out


def collect_phrase_paths_as_strings(paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    From enumerated path records, output:
      - path_strings: list[str]
      - path_ids: list[str]
    """
    path_strings: List[str] = []
    path_ids: List[str] = []
    for rec in paths:
        _push_unique_str(path_strings, rec.get("path_str", ""))
        _push_unique_str(path_ids, rec.get("leaf_path_id", ""))
    return {"path_strings": path_strings, "path_ids": path_ids}


# -----------------------------
# 7) Main API: phrase -> list of string paths
# -----------------------------
def prune_phrase_groundings_with_layered_tree(
    layered_tree: Dict[str, Any],
    phrase_groundings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    idx = index_layered_tree(layered_tree)
    id2n = idx["id2n"]
    parent = idx["parent"]
    sub_anc = idx["subtable_ancestor"]

    phrase2cands = labels_to_candidates(phrase_groundings, idx)

    avail_titles = {
        str(n.get("title", "")).strip()
        for n in layered_tree.get("nodes", [])
        if n.get("type") == "subtable" and str(n.get("title", "")).strip()
    }

    phrase2selected_titles: Dict[str, List[str]] = {}
    for pg in phrase_groundings:
        ph = str(pg.get("phrase", "")).strip()
        sel = (pg.get("selected_nodes", {}) or {}).get("subtable_titles", []) or []
        phrase2selected_titles[ph] = [str(x).strip() for x in sel if str(x).strip()]

    comp_cover, phrase_nodes_by_comp = build_phrase_candidates_by_component(phrase2cands, idx)

    # require components from range phrases (KEEP ALL years)
    required_components: Set[str] = set()
    for phrase in phrase2cands.keys():
        yr_rng = detect_range_endpoints(phrase)
        if not yr_rng:
            continue
        start_y, end_y = yr_rng
        llm_titles = phrase2selected_titles.get(phrase, [])
        kept_titles = [t for t in llm_titles if t in avail_titles]
        if len(kept_titles) <= 2:
            kept_titles = expand_year_range_titles(start_y, end_y, avail_titles)
        for t in kept_titles:
            for nid in idx["title2subtable_ids"].get(t, []):
                required_components.add(nid)

    chosen_components, uncovered = greedy_set_cover_components(set(phrase2cands.keys()), comp_cover, required_components)

    # prune candidates to chosen components
    pruned_phrase_nodes: Dict[str, Dict[str, List[str]]] = {}
    for phrase in phrase2cands.keys():
        comp_dict = phrase_nodes_by_comp.get(phrase, {})
        out = {"subtable_titles": [], "column_headers": [], "row_headers": []}
        for comp in chosen_components:
            if comp not in comp_dict:
                continue
            for bucket in out:
                out[bucket].extend(comp_dict[comp][bucket])

        for bucket in out:
            seen = set()
            new_list = []
            for x in out[bucket]:
                if x not in seen:
                    seen.add(x)
                    new_list.append(x)
            out[bucket] = new_list
        pruned_phrase_nodes[phrase] = out

    # per-phrase Steiner + lifting + enumerate paths -> stringify
    phrase2paths: Dict[str, Any] = {}
    for phrase, buckets in pruned_phrase_nodes.items():
        comp2terminals: Dict[str, List[str]] = {}
        for bucket in ("subtable_titles", "column_headers", "row_headers"):
            for nid in buckets.get(bucket, []):
                comp = nid if id2n[nid]["type"] == "subtable" else sub_anc(nid)
                if comp is None or comp not in chosen_components:
                    continue
                comp2terminals.setdefault(comp, []).append(nid)

        lifted_phrase_nodes: Set[str] = set()
        for comp_root, terminals in comp2terminals.items():
            seen = set()
            terminals = [t for t in terminals if not (t in seen or seen.add(t))]
            if not terminals:
                continue
            steiner = steiner_nodes_in_component(terminals, comp_root, parent)
            lifted = lift_steiner_root_to_leaf_paths(comp_root, steiner, idx)
            lifted_phrase_nodes |= lifted

        # enumerate leaf paths and stringify
        leaf_paths = enumerate_lifted_leaf_paths(lifted_phrase_nodes, idx)
        packed = collect_phrase_paths_as_strings(leaf_paths)

        phrase2paths[phrase] = {
            "selected_node_ids": buckets,
            "path_strings": packed["path_strings"],  # <-- you want: multiple strings per phrase
            "path_ids": packed["path_ids"],          # <-- corresponding leaf path_ids for value-index lookup
            "path_records": leaf_paths,              # optional: richer info (type/node_ids/label_path)
        }

    underspecified_phrases, linking_paths = [], []
    grounding_by_phrase: Dict[str, List[str]] = {}

    for key, val in phrase2paths.items():
        underspecified_phrases.append(key)
        phrase_paths = list(dict.fromkeys(val["path_strings"]))
        grounding_by_phrase[key] = phrase_paths
        linking_paths.extend(phrase_paths)
    return underspecified_phrases, list(dict.fromkeys(linking_paths)), grounding_by_phrase

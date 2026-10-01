#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import os
import numpy as np
import yaml


# ============================================================
# Data container
# ============================================================
@dataclass(frozen=True)
class Outcome:
    acc: float
    cost: float


# ============================================================
# IO
# ============================================================
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v).strip()


def get_key(rec: Dict[str, Any], key_field: str) -> str:
    return _stringify(rec.get(key_field, ""))


def _to01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 1.0 if v >= 0.5 else 0.0


# ============================================================
# Arms + pools
# ============================================================
def derive_arms_from_items(
    items: List[Dict[str, Any]],
    llm_id_field: str = "llm_id",
    llm_name_field: str = "llm",
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    """
    Returns:
      old2new, new2old, arm_names_new
    """
    old_ids: List[int] = []
    arm_names_old: Dict[int, str] = {}

    for rec in items:
        try:
            old = int(rec.get(llm_id_field))
        except Exception:
            continue
        old_ids.append(old)
        nm = _stringify(rec.get(llm_name_field, ""))
        if nm:
            arm_names_old.setdefault(old, nm)

    kept_old = sorted(set(old_ids))
    if not kept_old:
        raise ValueError(f"No valid '{llm_id_field}' found.")

    old2new = {old: new for new, old in enumerate(kept_old)}
    new2old = {new: old for old, new in old2new.items()}
    arm_names_new = {old2new[old]: arm_names_old.get(old, f"arm{old}") for old in kept_old}
    return old2new, new2old, arm_names_new


def build_pool_with_remap(
    items: List[Dict[str, Any]],
    key_field: str,
    old2new: Dict[int, int],
    llm_id_field: str = "llm_id",
) -> Dict[str, Dict[int, Outcome]]:
    pool: Dict[str, Dict[int, Outcome]] = {}
    for rec in items:
        k = get_key(rec, key_field)
        if not k:
            continue

        try:
            old_arm = int(rec.get(llm_id_field))
        except Exception:
            continue
        if old_arm not in old2new:
            continue
        a = old2new[old_arm]

        acc = _to01(rec.get("accuracy", 0))
        try:
            cost = float(rec.get("cost", 0.0))
        except Exception:
            cost = 0.0

        pool.setdefault(k, {})[a] = Outcome(acc=float(acc), cost=float(cost))
    return pool


def filter_complete(pool: Dict[str, Dict[int, Outcome]], n_arms: int) -> Dict[str, Dict[int, Outcome]]:
    out: Dict[str, Dict[int, Outcome]] = {}
    for k, arms in pool.items():
        if all(a in arms for a in range(n_arms)):
            out[k] = arms
    return out


def build_stream_keys(items: List[Dict[str, Any]], key_field: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for rec in items:
        k = get_key(rec, key_field)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


# ============================================================
# Embeddings (no cache, minimal)
# ============================================================
def encode_hf_meanpool_l2(
    texts: List[str],
    hf_encoder: str,
    batch_size: int,
    max_len: int,
    device: str,
) -> np.ndarray:
    """
    Returns (n, d) float32, L2-normalized.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(hf_encoder, use_fast=True)
    enc = AutoModel.from_pretrained(hf_encoder).eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)

    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            bt = texts[i : i + batch_size]
            t = tok(bt, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            y = enc(**t).last_hidden_state
            mask = t["attention_mask"].unsqueeze(-1).to(y.dtype)
            pooled = (y * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = F.normalize(pooled.float(), p=2, dim=-1)
            outs.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.vstack(outs) if outs else np.zeros((0, 0), dtype=np.float32)


def build_X(
    keys: List[str],
    key2text: Dict[str, str],
    hf_encoder: str,
    batch_size: int,
    max_len: int,
    device: str,
    add_bias: bool,
) -> np.ndarray:
    texts = [key2text.get(k, k) for k in keys]
    vecs = encode_hf_meanpool_l2(texts, hf_encoder, batch_size, max_len, device)
    if add_bias:
        bias = np.ones((vecs.shape[0], 1), dtype=np.float32)
        vecs = np.concatenate([bias, vecs], axis=1)
    return vecs.astype(np.float32)


# ============================================================
# LinUCB (solve instead of inverse)
# ============================================================
class SimpleLinUCB:
    """
    score_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
    A^{-1}x via solve(A, x).
    """

    def __init__(self, n_arms: int, d: int, alpha: float, ridge: float):
        self.n_arms = int(n_arms)
        self.d = int(d)
        self.alpha = float(alpha)
        self.ridge = float(ridge)

        self.A = [np.eye(self.d, dtype=np.float64) * self.ridge for _ in range(self.n_arms)]
        self.b = [np.zeros((self.d, 1), dtype=np.float64) for _ in range(self.n_arms)]
        self.theta: Optional[List[np.ndarray]] = None

    def warm_start_full_feedback(self, X: np.ndarray, keys: List[str], pool: Dict[str, Dict[int, Outcome]], reward_fn):
        for i, k in enumerate(keys):
            if k not in pool:
                continue
            x = X[i].reshape(self.d, 1).astype(np.float64)
            for a in range(self.n_arms):
                r = float(reward_fn(pool[k][a]))
                self.A[a] += x @ x.T
                self.b[a] += r * x

    def finalize(self):
        self.theta = []
        for a in range(self.n_arms):
            th = np.linalg.solve(self.A[a], self.b[a])
            self.theta.append(th)

    def score_all(self, x_row: np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise RuntimeError("Call finalize() before score_all().")
        x = x_row.reshape(self.d, 1).astype(np.float64)

        scores = np.zeros(self.n_arms, dtype=np.float64)
        for a in range(self.n_arms):
            mu = float(self.theta[a].T @ x)
            Ax = np.linalg.solve(self.A[a], x)
            sigma = float(np.sqrt(max(0.0, float(x.T @ Ax))))
            scores[a] = mu + self.alpha * sigma
        return scores


def train_two_heads_linucb(
    X_train: np.ndarray,
    train_keys: List[str],
    train_pool: Dict[str, Dict[int, Outcome]],
    n_arms: int,
    alpha: float,
    ridge: float,
) -> Tuple[SimpleLinUCB, SimpleLinUCB]:
    d = int(X_train.shape[1])
    m_acc = SimpleLinUCB(n_arms, d, alpha=alpha, ridge=ridge)
    m_cost = SimpleLinUCB(n_arms, d, alpha=alpha, ridge=ridge)

    m_acc.warm_start_full_feedback(X_train, train_keys, train_pool, reward_fn=lambda out: float(out.acc))
    m_cost.warm_start_full_feedback(X_train, train_keys, train_pool, reward_fn=lambda out: float(out.cost))

    m_acc.finalize()
    m_cost.finalize()
    return m_acc, m_cost


def compute_arm_cost_prior_from_train(train_keys, train_pool, n_arms) -> List[float]:
    sums = np.zeros(n_arms, dtype=np.float64)
    cnts = np.zeros(n_arms, dtype=np.float64)
    for k in train_keys:
        for a in range(n_arms):
            out = train_pool[k][a]
            sums[a] += float(out.cost)
            cnts[a] += 1.0
    return [float(x) for x in (sums / np.maximum(1.0, cnts))]


def compute_arm_acc_prior_from_train(train_keys, train_pool, n_arms) -> List[float]:
    sums = np.zeros(n_arms, dtype=np.float64)
    cnts = np.zeros(n_arms, dtype=np.float64)
    for k in train_keys:
        for a in range(n_arms):
            out = train_pool[k][a]
            sums[a] += float(out.acc)
            cnts[a] += 1.0
    return [float(x) for x in (sums / np.maximum(1.0, cnts))]


# ============================================================
# Online routing (budget-stop): output per-instance chosen arm
# ============================================================
def route_eval_instances_budget_stop(
    *,
    model_acc: SimpleLinUCB,
    model_cost: SimpleLinUCB,
    X_eval: np.ndarray,
    eval_keys: List[str],
    eval_pool: Dict[str, Dict[int, Outcome]],
    B_total: float,
    eta: float,
    lambda_init: float,
    hard_budget_filter: bool,
    filter_cost_prior: bool,
    arm_cost_prior: Optional[List[float]],
    beta_acc_prior: float,
    arm_acc_prior: Optional[List[float]],
    arm_names_new: Dict[int, str],
) -> Dict[str, Any]:
    """
    Returns:
      {
        "budget": ...,
        "n_answered": ...,
        "N_total": ...,
        "total_cost": ...,
        "avg_accuracy": ...,
        "decisions": [
           {"idx":i, "key":..., "chosen_arm":a, "llm":..., "cost":..., "acc":..., "spent":..., "lambda":...},
           ...
        ]
      }
    """
    n_arms = model_acc.n_arms
    N_total = len(eval_keys)

    beta = float(beta_acc_prior)
    beta = 0.0 if beta < 0 else (1.0 if beta > 1.0 else beta)

    if beta > 0.0 and (arm_acc_prior is None or len(arm_acc_prior) != n_arms):
        raise ValueError("beta_acc_prior>0 but arm_acc_prior missing/mismatched.")

    if filter_cost_prior and (arm_cost_prior is None or len(arm_cost_prior) != n_arms):
        raise ValueError("filter_cost_prior=True but arm_cost_prior missing/mismatched.")

    lam = float(lambda_init)
    spent = 0.0
    acc_sum = 0.0
    answered = 0

    decisions: List[Dict[str, Any]] = []

    for i, k in enumerate(eval_keys):
        x = X_eval[i]

        a_hat = model_acc.score_all(x)
        a_hat = np.clip(a_hat, 0.0, 1.0)

        if beta > 0.0:
            a_hat = (1.0 - beta) * a_hat + beta * np.asarray(arm_acc_prior, dtype=np.float64)

        c_hat = model_cost.score_all(x)
        util = a_hat - lam * c_hat

        if hard_budget_filter:
            remaining = float(B_total - spent)
            c_for_filter = np.asarray(arm_cost_prior, dtype=np.float64) if filter_cost_prior else c_hat
            feasible = np.where(c_for_filter <= remaining)[0]
            if feasible.size > 0:
                a_sel = int(feasible[np.argmax(util[feasible])])
            else:
                a_sel = int(np.argmin(c_for_filter))
        else:
            a_sel = int(np.argmax(util))

        out = eval_pool[k][a_sel]
        c_real = float(out.cost)

        # budget stop: do NOT count this instance
        if spent + c_real > float(B_total) + 1e-12:
            break

        spent += c_real
        acc_sum += float(out.acc)
        answered += 1

        prefix_budget = (answered / max(1, N_total)) * float(B_total)
        lam = max(0.0, lam + float(eta) * (spent - prefix_budget))

        decisions.append(
            {
                "idx": int(i),
                "key": k,
                "chosen_arm": int(a_sel),
                "llm": arm_names_new.get(a_sel, f"arm{a_sel}"),
                "cost": float(out.cost),
                "acc": float(out.acc),
                "spent": float(spent),
                "lambda": float(lam),
            }
        )

    avg_acc = float(acc_sum / max(1, answered))
    return {
        "budget": float(B_total),
        "n_answered": int(answered),
        "N_total": int(N_total),
        "total_cost": float(spent),
        "avg_accuracy": float(avg_acc),
        "decisions": decisions,
    }


# ============================================================
# Config loader
# ============================================================
def load_cfg(cfg_path: Path, dataset: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if "llm_routing" not in cfg:
        raise KeyError("config missing top-level key: llm_routing")
    if "datasets" not in cfg["llm_routing"] or dataset not in cfg["llm_routing"]["datasets"]:
        raise KeyError(f"config missing: llm_routing.datasets.{dataset}")
    return cfg["llm_routing"]["datasets"][dataset], cfg['datasets'][dataset], cfg




# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--config", type=str, default="../config.yaml")

    # model / features
    ap.add_argument("--hf_encoder", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--add_bias", action="store_true")
    ap.add_argument("--no_bias", action="store_true")

    # LinUCB
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--ridge", type=float, default=None)

    # dual update + budgets (NEW): list of budgets
    ap.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=None,
        help="List of total budget constraints. Example: --budgets 10 20 50",
    )
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--lambda_init", type=float, default=None)
    ap.add_argument("--hard_budget_filter", action="store_true")
    ap.add_argument("--no_hard_budget_filter", action="store_true")
    ap.add_argument("--filter_cost_prior", action="store_true")
    ap.add_argument("--no_filter_cost_prior", action="store_true")
    ap.add_argument("--beta_acc_prior", type=float, default=None)

    # eval order
    ap.add_argument("--shuffle_eval", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # output
    ap.add_argument("--out_json", type=str, default=None)

    args = ap.parse_args()

    ds_cfg, initial_cfg, basic_cfg = load_cfg(Path(args.config), args.dataset)
    out_json = Path(os.path.join(basic_cfg["out_folder"], args.dataset, "our", ds_cfg["out_json"]))
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # ----- resolve params from config, allow CLI override -----
    train_path = Path(os.path.join(basic_cfg['base_folder'], args.dataset, ds_cfg["train_path"]))
    eval_path = Path(os.path.join(basic_cfg['base_folder'], args.dataset, ds_cfg["eval_path"]))

    key_field = ds_cfg.get("key_field", "question")
    llm_id_field = ds_cfg.get("llm_id_field", "llm_id")
    llm_name_field = ds_cfg.get("llm_name_field", "llm")

    hf_encoder = args.hf_encoder or ds_cfg.get("hf_encoder", "distilbert-base-uncased")
    batch_size = int(args.batch_size or ds_cfg.get("batch_size", 32))
    max_len = int(args.max_len or ds_cfg.get("max_len", 128))
    device = args.device or ds_cfg.get("device", "cuda")

    alpha = float(args.alpha if args.alpha is not None else ds_cfg.get("alpha", 0.8))
    ridge = float(args.ridge if args.ridge is not None else ds_cfg.get("ridge", 1.0))

    # budgets priority: CLI > config budgets > config budget_total (backward compatible)
    if args.budgets is not None:
        budgets = [float(x) for x in args.budgets]
    else:
        if "budgets" in ds_cfg:
            budgets = [float(x) for x in ds_cfg["budgets"]]
        elif "budget_total" in ds_cfg:
            budgets = [float(ds_cfg["budget_total"])]
        else:
            raise KeyError("Need `budgets` (list) or `budget_total` (float) in config.")

    eta = float(args.eta if args.eta is not None else ds_cfg.get("eta", 1.0))
    lambda_init = float(args.lambda_init if args.lambda_init is not None else ds_cfg.get("lambda_init", 0.0))
    beta_acc_prior = float(args.beta_acc_prior if args.beta_acc_prior is not None else ds_cfg.get("beta_acc_prior", 0.0))

    # flags
    add_bias = bool(ds_cfg.get("add_bias", True))
    if args.no_bias:
        add_bias = False
    if args.add_bias:
        add_bias = True

    hard_budget_filter = bool(ds_cfg.get("hard_budget_filter", True))
    if args.no_hard_budget_filter:
        hard_budget_filter = False
    if args.hard_budget_filter:
        hard_budget_filter = True

    filter_cost_prior = bool(ds_cfg.get("filter_cost_prior", True))
    if args.no_filter_cost_prior:
        filter_cost_prior = False
    if args.filter_cost_prior:
        filter_cost_prior = True

    
    

    # ----- load data -----
    train_items = read_json(train_path)
    eval_items = read_json(eval_path)
    if not isinstance(train_items, list) or not isinstance(eval_items, list):
        raise ValueError("train/eval must be JSON lists")

    all_items = train_items + eval_items

    old2new, new2old, arm_names_new = derive_arms_from_items(
        all_items, llm_id_field=llm_id_field, llm_name_field=llm_name_field
    )
    n_arms = len(old2new)

    train_pool = filter_complete(build_pool_with_remap(train_items, key_field, old2new, llm_id_field), n_arms)
    eval_pool = filter_complete(build_pool_with_remap(eval_items, key_field, old2new, llm_id_field), n_arms)

    train_keys = [k for k in build_stream_keys(train_items, key_field) if k in train_pool]
    eval_keys = [k for k in build_stream_keys(eval_items, key_field) if k in eval_pool]

    if args.shuffle_eval:
        rr = random.Random(int(args.seed))
        rr.shuffle(eval_keys)

    if not train_keys or not eval_keys:
        raise ValueError("No complete keys in train/eval after filtering.")

    # key2text: raw question text (key itself)
    key2text: Dict[str, str] = {}
    for rec in all_items:
        k = get_key(rec, key_field)
        if k and k not in key2text:
            key2text[k] = k

    # ----- embeddings -----
    X_train = build_X(train_keys, key2text, hf_encoder, batch_size, max_len, device, add_bias=add_bias)
    X_eval = build_X(eval_keys, key2text, hf_encoder, batch_size, max_len, device, add_bias=add_bias)

    # ----- train 2-head LinUCB -----
    model_acc, model_cost = train_two_heads_linucb(
        X_train=X_train,
        train_keys=train_keys,
        train_pool=train_pool,
        n_arms=n_arms,
        alpha=alpha,
        ridge=ridge,
    )

    arm_cost_prior = compute_arm_cost_prior_from_train(train_keys, train_pool, n_arms)
    arm_acc_prior = compute_arm_acc_prior_from_train(train_keys, train_pool, n_arms)

    # ----- route on eval for EACH budget -----
    routing_by_budget: List[Dict[str, Any]] = []
    for B_total in budgets:
        routed = route_eval_instances_budget_stop(
            model_acc=model_acc,
            model_cost=model_cost,
            X_eval=X_eval,
            eval_keys=eval_keys,
            eval_pool=eval_pool,
            B_total=float(B_total),
            eta=eta,
            lambda_init=lambda_init,
            hard_budget_filter=hard_budget_filter,
            filter_cost_prior=filter_cost_prior,
            arm_cost_prior=arm_cost_prior,
            beta_acc_prior=beta_acc_prior,
            arm_acc_prior=arm_acc_prior,
            arm_names_new=arm_names_new,
        )
        routing_by_budget.append(routed)

        print(
            f"[B={float(B_total):.6f}] answered={routed['n_answered']}/{routed['N_total']} "
            f"acc={routed['avg_accuracy']:.4f} cost={routed['total_cost']:.6f}"
        )

    result = {
        "dataset": args.dataset,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "n_train": int(len(train_keys)),
        "n_eval": int(len(eval_keys)),
        "n_arms": int(n_arms),
        "arm_names_new": {str(k): v for k, v in sorted(arm_names_new.items())},
        "arm_cost_prior_train_mean": arm_cost_prior,
        "arm_acc_prior_train_mean": arm_acc_prior,
        "params": {
            "hf_encoder": hf_encoder,
            "batch_size": batch_size,
            "max_len": max_len,
            "device": device,
            "add_bias": add_bias,
            "linucb_alpha": alpha,
            "linucb_ridge": ridge,
            "budgets": [float(b) for b in budgets],
            "eta": eta,
            "lambda_init": lambda_init,
            "hard_budget_filter": hard_budget_filter,
            "filter_cost_prior": filter_cost_prior,
            "beta_acc_prior": beta_acc_prior,
            "shuffle_eval": bool(args.shuffle_eval),
            "seed": int(args.seed),
        },
        "routing_by_budget": routing_by_budget,
    }

    write_json(out_json, result)
    print(f"[OK] wrote routing decisions: {out_json}")


if __name__ == "__main__":
    main()
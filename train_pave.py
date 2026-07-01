#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.get_embedding import (  # noqa: E402
    SequenceEmbeddingOptions,
    extract_embeddings,
    list_registered_models,
    load_embedder,
    register_model,
)


DEFAULT_CLASSIFICATION_JSON = ROOT / "data" / "virus_antibody_classification_dataset.json"
DEFAULT_REGRESSION_JSON = ROOT / "data" / "virus_antibody_regression_dataset.json"
DEFAULT_CLASSIFICATION_JSON_CLEAN = ROOT / "data" / "virus_antibody_classification_dataset_clean.json"
DEFAULT_REGRESSION_JSON_CLEAN = ROOT / "data" / "virus_antibody_regression_dataset_clean.json"
DATASET_VARIANT_CHOICES: Tuple[str, ...] = ("default", "clean")

DEFAULT_INDEPENDENT_JSON = ROOT / "data" / "ch505_validation.json"
DEFAULT_VRC601_INDEPENDENT_JSON = ROOT / "data" / "vrc601_validation.json"
DEFAULT_INDEPENDENT_JSONS = (DEFAULT_INDEPENDENT_JSON, DEFAULT_VRC601_INDEPENDENT_JSON)
DEFAULT_CACHE_DIR = ROOT / "results" / "pave" / "embedding_cache"
DEFAULT_EPITOPE_FEATURE_CSV = ROOT / "data" / "Env_HXB2_feature.csv"
DEFAULT_ALIGNMENT_FASTA = ROOT / "data" / "ch505_aligned_to_training.fasta"
DEFAULT_ANTIBODY_ID_MAPPING_JSON = ROOT / "scripts" / "antibody_id_name_mapping.json"
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
N_AA = len(AA_ALPHABET)
EPITOPE_POS_FEAT_PER_POS = N_AA + 1
ANTIBODY_NAME_COLUMNS: Tuple[str, ...] = ("Antibody", "antibody", "antibody_full")
EPITOPE_CLASS_TO_REGION = {
    "CD4bs": "CD4bs",
    "V3": "V3",
    "V1V2": "V1V2",
    "MPER": "MPER",
    "fusion_peptide": "fusion_peptide",
    "CD4i": "CD4bs",
    "interface": "CD4bs",
    "glycan_other": "V3",
    "multi": "CD4bs",
    "silent_face": "V3",
    "other": "CD4bs",
    "unknown": "CD4bs",
}
_DEFAULT_MODEL_WEIGHT_ROOT = (
    ROOT.parents[1] / "DNAFM" / "model_weight"
    if len(ROOT.parents) > 1
    else ROOT / "model_weight"
)
MODEL_WEIGHT_ROOT = (
    _DEFAULT_MODEL_WEIGHT_ROOT
    if _DEFAULT_MODEL_WEIGHT_ROOT.exists()
    else ROOT / "model_weight"
)
PROTEIN_MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "ESM1b-650M": {
        "model_name": "ESM-1b",
        "model_dir": str(MODEL_WEIGHT_ROOT / "esm1b_t33_650M_UR50S"),
    },
    "ESM2-8M": {
        "model_name": "ESM-2",
        "model_dir": str(MODEL_WEIGHT_ROOT / "esm2_t6_8M_UR50D"),
    },
    "ESM2-35M": {
        "model_name": "ESM-2",
        "model_dir": str(MODEL_WEIGHT_ROOT / "esm2_t12_35M_UR50D"),
    },
    "ESM2-150M": {
        "model_name": "ESM-2",
        "model_dir": str(MODEL_WEIGHT_ROOT / "esm2_t30_150M_UR50D"),
    },
    "ESM2-650M": {
        "model_name": "ESM-2",
        "model_dir": str(MODEL_WEIGHT_ROOT / "esm2_t33_650M_UR50D"),
    },
    "ESM2-3B": {
        "model_name": "ESM-2",
        "model_dir": str(MODEL_WEIGHT_ROOT / "esm2_t36_3B_UR50D"),
    },
    "ESMC-300M": {
        "model_name": "ESMC",
        "model_dir": str(MODEL_WEIGHT_ROOT / "esmc-300m-2024-12"),
    },
    "ESMC-600M": {
        "model_name": "ESMC",
        "model_dir": str(MODEL_WEIGHT_ROOT / "esmc-600m-2024-12"),
    },
    "ESMC-6B": {
        "model_name": "ESMC",
        "model_dir": str(MODEL_WEIGHT_ROOT / "ESMC-6B"),
    },
    "Antiberta2": {
        "model_name": "Antiberta2",
        "model_dir": str(MODEL_WEIGHT_ROOT / "antiberta2"),
    },
    "Antiberta2-H": {
        "model_name": "Antiberta2-H",
        "model_dir": str(MODEL_WEIGHT_ROOT / "antiberta2"),
    },
    "Antiberta2-L": {
        "model_name": "Antiberta2-L",
        "model_dir": str(MODEL_WEIGHT_ROOT / "antiberta2"),
    },
    "LucaOne-default-step36M": {
        "model_name": "LucaOne-default-step36M",
        "model_dir": str(MODEL_WEIGHT_ROOT / "LucaOne-default-step36M"),
    },
}
PROTEIN_LANGUAGE_MODELS: Tuple[str, ...] = tuple(PROTEIN_MODEL_PRESETS)
CONSOLE = Console(highlight=False)


def _fmt_float(value: float, digits: int = 4) -> str:
    if value != value or np.isinf(value):
        return "nan"
    return f"{value:.{digits}f}"


def _print_antibody_id_mapping_warning(prefix: str, missing_rows: int, total_rows: int, examples: Sequence[str]) -> None:
    example_text = ", ".join(examples) if examples else "none"
    body = (
        f"[bold yellow]{prefix} antibody ID mapping incomplete[/]\n"
        f"Missing rows: [bold]{missing_rows}[/] / {total_rows}\n"
        f"Examples: [italic]{example_text}[/]"
    )
    CONSOLE.print(Panel.fit(body, title="Antibody ID", border_style="yellow"))


_TRAIN_BALANCE_SEED_OFFSET = 971_042


def _print_train_balance_single_class_warning(fold_index: int, antibodies: Sequence[str]) -> None:
    ab_text = ", ".join(antibodies) if antibodies else "none"
    body = (
        f"[bold yellow]Fold {fold_index:02d}: train balance skipped for single-class antibodies[/]\n"
        f"Antibodies: [italic]{ab_text}[/]\n"
        "Kept original training samples for those antibodies (no upsampling)."
    )
    CONSOLE.print(Panel.fit(body, title="Train balance", border_style="yellow"))


def _count_unique_groups(data: pd.DataFrame, positions: np.ndarray, group_col: str) -> Optional[int]:
    # group_col may be a sentinel like "random" (no grouping) rather than a real
    # column; in that case there is no group count to report.
    if group_col not in data.columns:
        return None
    group_values = data.iloc[np.asarray(positions, dtype=np.int64)][group_col]
    return int(group_values.fillna("__missing_group__").astype(str).nunique())


def _print_fold_header(
    fold_index: int,
    train_size: int,
    val_size: int,
    test_size: int,
    *,
    train_groups: Optional[int] = None,
    val_groups: Optional[int] = None,
    test_groups: Optional[int] = None,
    train_raw_size: Optional[int] = None,
) -> None:
    def _format_split(size: int, groups: Optional[int]) -> str:
        return f"{size}" if groups is None else f"{size} ({groups} groups)"

    train_text = _format_split(train_size, train_groups)
    if train_raw_size is not None and int(train_raw_size) != int(train_size):
        if train_groups is None:
            train_text = f"{train_size} (raw {train_raw_size})"
        else:
            train_text = f"{train_size} (raw {train_raw_size}, {train_groups} groups)"

    grid = Table.grid(padding=(0, 2))
    grid.add_row("[bold cyan]Fold[/]", f"{fold_index:02d}")
    grid.add_row("[green]Train[/]", train_text)
    grid.add_row("[yellow]Val[/]", _format_split(val_size, val_groups))
    grid.add_row("[magenta]Test[/]", _format_split(test_size, test_groups))
    CONSOLE.print(Panel.fit(grid, border_style="cyan"))


def _print_training_feature_summary(
    args: argparse.Namespace,
    *,
    env_base: np.ndarray,
    vh_all: np.ndarray,
    vl_all: np.ndarray,
    env_aux: Optional[np.ndarray],
    epitope_pos: np.ndarray,
    epitope_token: np.ndarray,
    env_aux_info: Dict[str, Any],
    env_tok_all: Optional[Any],
    vh_tok_all: Optional[Any],
    vl_tok_all: Optional[Any],
) -> None:
    """Print which tensors feed the model and their last-dimension sizes before training."""
    model_arch = _selected_model_arch(args)
    use_tok = model_arch in {"cross_attn", "hybrid_cross_attn"} and env_tok_all is not None
    is_hybrid = model_arch == "hybrid_cross_attn"
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Branch", style="dim", no_wrap=True)
    table.add_column("Component", no_wrap=True)
    table.add_column("Source", overflow="fold")
    table.add_column("Dim", justify="right", no_wrap=True)
    table.add_column("Switch / arg", style="dim", overflow="fold")

    backend = str(getattr(args, "embedding_backend", "pave"))
    backend_note = f"hash_dim={int(args.hash_embedding_dim)}" if backend == "hash" else "PLM pooled"
    emb_switch = "--embedding_backend"
    if backend == "hash":
        emb_switch += ", --hash_embedding_dim"
    table.add_row("—", "embedding_backend", backend, backend_note, emb_switch)

    env_emb = int(env_base.shape[1])
    vh_d = int(vh_all.shape[1])
    vl_d = int(vl_all.shape[1])
    aux_total = int(env_aux.shape[1]) if env_aux is not None else 0
    d_grid = int(env_aux_info.get("dim", 0)) if args.use_env_aux_features else 0
    d_epi_pos = int(epitope_pos.shape[1])
    d_epi_tok = int(epitope_token.shape[1])

    pool_branch = "pooled (active)" if (not use_tok or is_hybrid) else "pooled (built)"
    table.add_row(
        pool_branch,
        "Env sequence embedding",
        f"{args.env_seq_col} / {args.env_model}",
        str(env_emb),
        "--env_model, --env_seq_col",
    )
    if d_grid > 0:
        table.add_row(
            pool_branch,
            "Env aux (region / glyco stats)",
            f"use_env_aux_features / bins={int(args.env_aux_region_bins)}",
            str(d_grid),
            "--use_env_aux_features / --no_env_aux_features, --env_aux_region_bins",
        )
    if d_epi_pos > 0:
        table.add_row(
            pool_branch,
            "Epitope position one-hot",
            Path(str(args.epitope_feature_csv)).name,
            str(d_epi_pos),
            "--use_epitope_pos_features / --no_epitope_pos_features",
        )
    if d_epi_tok > 0:
        table.add_row(
            pool_branch,
            "Epitope token pooled embedding",
            f"{args.epitope_token_pool} / {args.env_model}",
            str(d_epi_tok),
            "--use_epitope_token_features / --no_epitope_token_features, --epitope_token_pool",
        )
    table.add_row(
        pool_branch,
        "Env vector total",
        "PLM + aux blocks above" if aux_total > 0 else "PLM only",
        str(env_emb + aux_total),
        "— (concat of rows above)",
    )
    table.add_row(
        pool_branch,
        "VH sequence embedding",
        f"{args.vh_seq_col} / {args.vh_model}",
        str(vh_d),
        "--vh_model, --vh_seq_col",
    )
    table.add_row(
        pool_branch,
        "VL sequence embedding",
        f"{args.vl_seq_col} / {args.vl_model}",
        str(vl_d),
        "--vl_model, --vl_seq_col",
    )

    if use_tok:
        env_tok_col = getattr(args, "token_env_seq_col", None) or args.env_seq_col
        ted = int(env_tok_all[0].shape[1])
        thd = int(vh_tok_all[0].shape[1])
        tld = int(vl_tok_all[0].shape[1])
        tok_branch = "tokens (active)"
        tok_switch = (
            "--use_token_cross_attention, --token_shard_proj_dim"
            + (", --token_env_seq_col" if getattr(args, "token_env_seq_col", None) else "")
        )
        table.add_row(tok_branch, "Env per-token embedding", f"{env_tok_col} / {args.env_model}", str(ted), tok_switch)
        table.add_row(
            tok_branch,
            "VH per-token embedding",
            f"{args.vh_seq_col} / {args.vh_model}",
            str(thd),
            "--use_token_cross_attention, --token_shard_proj_dim",
        )
        table.add_row(
            tok_branch,
            "VL per-token embedding",
            f"{args.vl_seq_col} / {args.vl_model}",
            str(tld),
            "--use_token_cross_attention, --token_shard_proj_dim",
        )

    if args.use_antibody_id_feature:
        table.add_row(
            "other",
            "Antibody ID embedding",
            "use_antibody_id_feature",
            str(int(args.num_ab_id_embeddings)),
            "--use_antibody_id_feature / --no_antibody_id_feature",
        )

    if args.task in {"classification", "multitask"}:
        balance_on = bool(getattr(args, "balance_train_by_antibody", False))
        table.add_row(
            "other",
            "Train balance by antibody",
            "upsample per fold (train only)" if balance_on else "disabled",
            "on" if balance_on else "off",
            "--balance_train_by_antibody / --no_balance_train_by_antibody",
        )

    if model_arch == "hybrid_cross_attn":
        mode = "pooled trunk + token gated residual (PAVEHybridCrossAttn)"
    elif use_tok:
        mode = "token cross-attention (PAVECrossAttn)"
    else:
        mode = "pooled (PAVE)"
    CONSOLE.print(Panel.fit(table, title=f"[bold]Features & dimensions[/] — {mode}", border_style="green"))
    if use_tok and (not is_hybrid) and aux_total > 0:
        CONSOLE.print(
            "[dim]Note: pooled Env+aux rows are still built for bookkeeping, but the forward pass uses token tensors only.[/]"
        )


def _print_epoch_table(
    epoch: int,
    epochs: int,
    train_loss: float,
    columns: Sequence[str],
    rows: Sequence[Tuple[str, Sequence[str]]],
) -> None:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_lines=False,
    )
    table.add_column("Epoch", style="bold cyan", no_wrap=True)
    table.add_column("train_loss", justify="right", no_wrap=True)
    table.add_column("Split", style="bold", no_wrap=True)
    for column in columns:
        table.add_column(column, justify="right", no_wrap=True)
    for row_idx, (split_name, values) in enumerate(rows):
        style = "green" if split_name.lower() == "val" else "magenta" if split_name.lower() == "test" else "white"
        epoch_text = f"{epoch:03d}/{epochs:03d}" if row_idx == 0 else ""
        train_loss_text = _fmt_float(train_loss, 5) if row_idx == 0 else ""
        table.add_row(epoch_text, train_loss_text, f"[{style}]{split_name}[/]", *values)
    CONSOLE.print(table)


def _fmt_metric_cell(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f or np.isinf(f):
        return "nan"
    return f"{f:.{digits}f}"


def _print_hyperparams_table(output: Dict[str, Any]) -> None:
    """Print a compact two-column hyperparameter summary (not the full args blob)."""
    args_dict: Dict[str, Any] = dict(output.get("args") or {})
    hp: Dict[str, Any] = dict(output.get("hyperparams") or {})
    models = output.get("models") or {}
    rows: List[Tuple[str, Any]] = []
    seen: set[str] = set()

    def add(name: str, value: Any) -> None:
        if name in seen:
            return
        seen.add(name)
        rows.append((name, value))

    for key in (
        "task",
        "model_arch",
        "cv_scheme",
        "num_folds",
        "seed",
        "epochs",
        "batch_size",
        "num_workers",
        "lr",
        "weight_decay",
        "patience",
        "grad_clip",
        "early_stop_metric_cls",
        "early_stop_metric_reg",
        "embedding_backend",
        "hash_embedding_dim",
        "embedding_batch_size",
        "use_token_cross_attention",
        "cross_attn_heads",
        "cross_attn_layers",
        "hybrid_token_gate_hidden",
        "token_batch_size",
        "token_shard_proj_dim",
    ):
        if key in args_dict:
            add(key, args_dict[key])
    for role in ("env", "vh", "vl"):
        if role in models:
            add(f"model_{role}", models[role])
    for key in sorted(hp.keys()):
        add(key, hp[key])

    table = Table(title="Hyperparameters", box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for name, value in rows:
        table.add_row(name, str(value))
    CONSOLE.print(table)


def _print_per_fold_test_metrics_table(per_fold: Dict[str, Any], *, title: str) -> None:
    if not per_fold:
        return
    table = Table(title=title, box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Folds", justify="right")
    for metric in sorted(per_fold.keys()):
        cell = per_fold[metric]
        if not isinstance(cell, dict):
            continue
        table.add_row(
            metric,
            _fmt_metric_cell(cell.get("mean")),
            _fmt_metric_cell(cell.get("std")),
            str(int(cell.get("num_folds", 0))),
        )
    CONSOLE.print(table)


def _print_independent_ensemble_table(ensemble: Dict[str, Any], *, title: str = "Independent test (ensemble)") -> None:
    """Print average vs vote for classification-style metrics; regression-only if no vote."""
    if not ensemble:
        return
    if "classification" in ensemble and isinstance(ensemble["classification"], dict):
        cls_block = ensemble["classification"]
        avg = cls_block.get("average") or {}
        vote = cls_block.get("vote") or {}
        metrics = sorted(set(avg.keys()) | set(vote.keys()))
        table = Table(title=f"{title} — classification", box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("average", justify="right")
        table.add_column("vote", justify="right")
        for m in metrics:
            table.add_row(m, _fmt_metric_cell(avg.get(m)), _fmt_metric_cell(vote.get(m)))
        CONSOLE.print(table)
        reg = ensemble.get("regression") or {}
        reg_avg = reg.get("average") if isinstance(reg, dict) else None
        if isinstance(reg_avg, dict) and reg_avg:
            t2 = Table(title=f"{title} — regression (average)", box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
            t2.add_column("Metric", style="cyan", no_wrap=True)
            t2.add_column("average", justify="right")
            for m in sorted(reg_avg.keys()):
                t2.add_row(m, _fmt_metric_cell(reg_avg.get(m)))
            CONSOLE.print(t2)
        return

    avg = ensemble.get("average")
    vote = ensemble.get("vote")
    if isinstance(avg, dict) and isinstance(vote, dict) and vote:
        metrics = sorted(set(avg.keys()) | set(vote.keys()))
        table = Table(title=title, box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("average", justify="right")
        table.add_column("vote", justify="right")
        for m in metrics:
            table.add_row(m, _fmt_metric_cell(avg.get(m)), _fmt_metric_cell(vote.get(m)))
        CONSOLE.print(table)
        return
    if isinstance(avg, dict) and avg:
        table = Table(title=title, box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("average", justify="right")
        for m in sorted(avg.keys()):
            table.add_row(m, _fmt_metric_cell(avg.get(m)))
        CONSOLE.print(table)


def _print_final_output(output: Dict[str, Any]) -> None:
    run_dir = str(output.get("run_dir", ""))
    CONSOLE.print(Panel.fit(f"[bold]Run dir[/]\n{run_dir}", title="Done", border_style="green"))
    _print_hyperparams_table(output)

    cv = output.get("cv_test_summary") or {}
    per_fold = cv.get("per_fold_test_metrics")
    if isinstance(per_fold, dict) and "classification" in per_fold and "regression" in per_fold:
        cls_pf = per_fold.get("classification") or {}
        reg_pf = per_fold.get("regression") or {}
        if cls_pf:
            _print_per_fold_test_metrics_table(cls_pf, title="CV per-fold test — classification (mean ± std)")
        if reg_pf:
            _print_per_fold_test_metrics_table(reg_pf, title="CV per-fold test — regression (mean ± std)")
    elif isinstance(per_fold, dict) and per_fold:
        task = str(output.get("task", "task"))
        _print_per_fold_test_metrics_table(per_fold, title=f"CV per-fold test — {task} (mean ± std)")

    ens_list = output.get("independent_evaluations")
    if isinstance(ens_list, list) and ens_list:
        for block in ens_list:
            title = f"Independent test (ensemble) — {block.get('name', 'cohort')}"
            _print_independent_ensemble_table(block.get("ensemble_metrics") or {}, title=title)
    else:
        ens = output.get("ensemble_independent_metrics")
        if ens:
            _print_independent_ensemble_table(ens, title="Independent test (ensemble)")


def _selected_model_arch(args: argparse.Namespace) -> str:
    arch = str(getattr(args, "model_arch", "auto") or "auto").lower()
    if arch == "auto":
        return "cross_attn" if bool(getattr(args, "use_token_cross_attention", False)) else "pave"
    return arch


def _single_task_early_stop_value(
    task: str,
    val_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> Tuple[float, bool]:
    """Return (val_key, larger_is_better) for early stopping."""
    if task == "classification":
        cls_choice = str(getattr(args, "early_stop_metric_cls", "f1")).lower()
        metric_key = {"acc": "accuracy", "auc": "roc_auc", "f1": "f1_binary"}[cls_choice]
        v = float(val_metrics.get(metric_key, float("-inf")))
        if not np.isfinite(v):
            v = float("-inf")
        return v, True
    reg_choice = str(getattr(args, "early_stop_metric_reg", "rmse")).lower()
    if reg_choice == "rmse":
        v = float(val_metrics.get("rmse", float("inf")))
        if not np.isfinite(v):
            v = float("inf")
        return v, False
    if reg_choice == "r2":
        v = float(val_metrics.get("r2", float("-inf")))
        if not np.isfinite(v):
            v = float("-inf")
        return v, True
    # "R" maps to pearson_r
    v = float(val_metrics.get("pearson_r", float("-inf")))
    if not np.isfinite(v):
        v = float("-inf")
    return v, True


def _print_registered_models(models: Sequence[str]) -> None:
    table = Table(title="Registered Models", box=box.SIMPLE_HEAD)
    table.add_column("Name", style="cyan")
    for name in models:
        table.add_row(str(name))
    CONSOLE.print(table)


def _print_protein_models_table(rows: Sequence[Tuple[str, str, str]]) -> None:
    table = Table(title="Protein Models", box=box.SIMPLE_HEAD)
    table.add_column("Name", style="cyan")
    table.add_column("Backend", style="green")
    table.add_column("Model Dir", style="white")
    for name, backend, model_dir in rows:
        table.add_row(name, backend, model_dir)
    CONSOLE.print(table)


def _load_esmc_embedder(model_name: str, model_dir: Optional[str] = None, **_: Any) -> Tuple[Any, SequenceEmbeddingOptions]:
    from models.esmc_model import ESMCModel

    mdir = model_dir or str(MODEL_WEIGHT_ROOT / "esmc-300m-2024-12")
    model = ESMCModel(
        model_name=model_name,
        model_path=mdir,
        device=None,
        input_type="protein",
        max_length=2048,
    )
    return model, SequenceEmbeddingOptions(
        batch_size=16,
        pool="mean",
        layer_name=None,
        max_length=2048,
        hidden_size=model.embedding_dim or None,
        is_cnn=False,
    )


register_model("ESMC", _load_esmc_embedder)


def _load_antiberta2_embedder(
    model_name: str,
    model_dir: Optional[str] = None,
    **_: Any,
) -> Tuple[Any, SequenceEmbeddingOptions]:
    from models.antiberta2_model import Antiberta2Model

    mdir = model_dir or str(MODEL_WEIGHT_ROOT / "antiberta2")
    chain_token = None
    if model_name.endswith("-H"):
        chain_token = "Ḣ"
    elif model_name.endswith("-L"):
        chain_token = "Ḷ"
    model = Antiberta2Model(
        model_name=model_name,
        model_path=mdir,
        device=None,
        max_length=256,
        chain_token=chain_token,
    )
    return model, SequenceEmbeddingOptions(
        batch_size=16,
        pool="mean",
        layer_name=None,
        max_length=256,
        hidden_size=model.embedding_dim or None,
        is_cnn=False,
    )


register_model("Antiberta2", _load_antiberta2_embedder)
register_model("Antiberta2-H", _load_antiberta2_embedder)
register_model("Antiberta2-L", _load_antiberta2_embedder)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    s_true = float(np.std(y_true))
    s_pred = float(np.std(y_pred))
    if s_true < 1e-12 or s_pred < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    if value is None or value != value:
        return 0.0
    return float(value)


def _logits_to_prob(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)))


def find_best_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -1.0
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    for t in thresholds:
        pred = (prob >= t).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, pred, average="binary", zero_division=0))
        elif metric == "mcc":
            score = float(matthews_corrcoef(y_true, pred))
        else:
            raise ValueError(metric)
        if score > best_score:
            best_t, best_score = t, score
    return float(best_t), float(best_score)


def _classification_metrics_from_prob(
    y_true: np.ndarray,
    prob: np.ndarray,
    pred_label: Optional[np.ndarray] = None,
    *,
    prob_threshold: float = 0.5,
) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64)
    if pred_label is None:
        pred = (prob >= prob_threshold).astype(np.int64)
    else:
        pred = np.asarray(pred_label, dtype=np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, pred, average="binary", zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
        metrics["avg_precision"] = float(average_precision_score(y_true, prob))
        metrics["mcc"] = float(matthews_corrcoef(y_true, pred))
        metrics["recall"] = float(recall_score(y_true, pred, average="binary", zero_division=0))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["avg_precision"] = float("nan")
        metrics["mcc"] = float("nan")
        metrics["recall"] = float("nan")
    return metrics


def _classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    prob_threshold: float = 0.5,
) -> Dict[str, float]:
    prob = _logits_to_prob(logits)
    return _classification_metrics_from_prob(y_true.astype(np.int64), prob, prob_threshold=prob_threshold)


def _grouped_accuracy_summary(
    frame: pd.DataFrame,
    group_col: str,
    *,
    y_col: str = "y_true",
    pred_col: str = "vote_y_pred_label",
) -> Dict[str, Dict[str, Any]]:
    if group_col not in frame.columns:
        return {}
    grouped: Dict[str, Dict[str, Any]] = {}
    for key, group in frame.groupby(group_col, sort=True):
        y = group[y_col].astype(int).to_numpy()
        pred = group[pred_col].astype(int).to_numpy()
        grouped[str(key)] = {
            "accuracy": float(accuracy_score(y, pred)),
            "n": int(len(group)),
        }
    return grouped


def _ch505_ensemble_grouped_accuracy(ensemble_prediction_path: str | Path) -> Dict[str, Any]:
    """Per-lineage and per-timepoint accuracy on CH505 ensemble predictions."""
    path = Path(ensemble_prediction_path)
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if "average_y_pred_label" in frame.columns:
        pred_col = "average_y_pred_label"
    elif "vote_y_pred_label" in frame.columns:
        pred_col = "vote_y_pred_label"
    else:
        return {"prediction_file": str(path), "error": "no ensemble prediction label column"}

    summary: Dict[str, Any] = {
        "prediction_file": str(path),
        "pred_column": pred_col,
    }
    by_lineage = _grouped_accuracy_summary(frame, "lineage", pred_col=pred_col)
    if by_lineage:
        summary["by_lineage"] = by_lineage
        summary["mean_accuracy_by_lineage"] = float(
            np.mean([row["accuracy"] for row in by_lineage.values()])
        )
    by_timepoint = _grouped_accuracy_summary(frame, "timepoint", pred_col=pred_col)
    if by_timepoint:
        summary["by_timepoint"] = by_timepoint
        summary["mean_accuracy_by_timepoint"] = float(
            np.mean([row["accuracy"] for row in by_timepoint.values()])
        )
    return summary


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": r2,
        "pearson_r": _pearson(y_true, y_pred),
        "spearman_r": _spearman(y_true, y_pred),
    }


def _empty_regression_metrics() -> Dict[str, float]:
    return {
        "mae": float("nan"),
        "rmse": float("nan"),
        "mse": float("nan"),
        "r2": float("nan"),
        "pearson_r": float("nan"),
        "spearman_r": float("nan"),
    }


def _select_task_prediction(task: str, pred_cls: np.ndarray | torch.Tensor, pred_reg: np.ndarray | torch.Tensor):
    return pred_cls if task == "classification" else pred_reg


@dataclass
class SplitData:
    train_pos: np.ndarray
    val_pos: np.ndarray
    test_pos: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_mean: float
    y_std: float
    class_label_map: Optional[Dict[Any, int]]
    y_cls_train: Optional[np.ndarray] = None
    y_cls_val: Optional[np.ndarray] = None
    y_cls_test: Optional[np.ndarray] = None
    y_reg_train: Optional[np.ndarray] = None
    y_reg_val: Optional[np.ndarray] = None
    y_reg_test: Optional[np.ndarray] = None
    reg_mask_train: Optional[np.ndarray] = None
    reg_mask_val: Optional[np.ndarray] = None
    reg_mask_test: Optional[np.ndarray] = None


class MultiTaskFeatureDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        ab_ids: np.ndarray,
        env: np.ndarray,
        vh: np.ndarray,
        vl: np.ndarray,
        y_cls: np.ndarray,
        y_reg: np.ndarray,
        reg_mask: np.ndarray,
    ) -> None:
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.ab_ids = torch.as_tensor(ab_ids, dtype=torch.long)
        self.env = torch.as_tensor(env, dtype=torch.float32)
        self.vh = torch.as_tensor(vh, dtype=torch.float32)
        self.vl = torch.as_tensor(vl, dtype=torch.float32)
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.float32)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.reg_mask = torch.as_tensor(reg_mask, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y_cls.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.ids[index],
            self.ab_ids[index],
            self.env[index],
            self.vh[index],
            self.vl[index],
            self.y_cls[index],
            self.y_reg[index],
            self.reg_mask[index],
        )


class FeatureDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        ab_ids: np.ndarray,
        env: np.ndarray,
        vh: np.ndarray,
        vl: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.ab_ids = torch.as_tensor(ab_ids, dtype=torch.long)
        self.env = torch.as_tensor(env, dtype=torch.float32)
        self.vh = torch.as_tensor(vh, dtype=torch.float32)
        self.vl = torch.as_tensor(vl, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ids[index], self.ab_ids[index], self.env[index], self.vh[index], self.vl[index], self.y[index]


class ShardedTokenStore:
    """Lazy-loading store for variable-length token embeddings saved as shards.

    Shard files are loaded on first access and kept in a dict cache so
    each shard is read from disk at most once per process.  The total
    in-memory footprint equals only the shards that have been touched,
    which DataLoader workers cycle through batch-by-batch.

    Parameters
    ----------
    shard_paths : list[str]
        Ordered list of ``.pt`` shard file paths as produced by
        ``_extract_token_feature_matrix``.
    shard_size : int
        Number of sequences per shard (must match how shards were written).
    n : int
        Total number of sequences (used for bounds checking).
    """

    def __init__(self, shard_paths: List[str], shard_size: int, n: int, max_cache_shards: int = 8) -> None:
        self.shard_paths = shard_paths
        self.shard_size = shard_size
        self.n = n
        self.max_cache_shards = max(1, max_cache_shards)
        self._cache: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        if shard_idx not in self._cache:
            shard = torch.load(self.shard_paths[shard_idx], map_location="cpu", weights_only=False)
            self._cache[shard_idx] = shard["token_seqs"]
            # LRU eviction: drop the least-recently-used shard when over budget
            while len(self._cache) > self.max_cache_shards:
                self._cache.popitem(last=False)
        else:
            # Mark as most recently used
            self._cache.move_to_end(shard_idx)
        return self._cache[shard_idx][local_idx]

    def get_numpy(self, idx: int) -> np.ndarray:
        t = self[idx]
        return t.numpy().astype(np.float32)

    def get_all_lengths(self) -> List[int]:
        """Return per-sample sequence lengths (first dim of each token tensor).

        Scans shards sequentially via the LRU cache; result is memoized on this
        object so subsequent calls are O(1).
        """
        if hasattr(self, "_lengths_cache"):
            return self._lengths_cache  # type: ignore[attr-defined]
        lengths: List[int] = []
        for shard_idx in range(len(self.shard_paths)):
            first_global = shard_idx * self.shard_size
            if first_global >= self.n:
                break
            # Warm the LRU cache for this shard
            _ = self[first_global]
            seqs = self._cache[shard_idx]  # guaranteed MRU after above access
            n_in_shard = min(self.shard_size, self.n - first_global)
            lengths.extend(int(t.shape[0]) for t in seqs[:n_in_shard])
        self._lengths_cache: List[int] = lengths  # type: ignore[attr-defined]
        return lengths

    def subset(self, indices) -> "ShardedTokenStoreView":
        """Return a lazy view over a subset of indices (no data is loaded)."""
        return ShardedTokenStoreView(self, list(indices))

    def to_list(self) -> List[np.ndarray]:
        """Load everything into a flat list (only safe when data fits in RAM)."""
        out: List[np.ndarray] = []
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu", weights_only=False)
            out.extend(t.numpy().astype(np.float32) for t in shard["token_seqs"])
        return out


class ShardedTokenStoreView:
    """A read-only index-remapped view of a ShardedTokenStore.

    Accessing ``view[i]`` retrieves ``store[indices[i]]`` lazily.
    No data is copied at construction time.
    """

    def __init__(self, store: ShardedTokenStore, indices: List[int]) -> None:
        self._store = store
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._store[self._indices[idx]]

    def get_all_lengths(self) -> List[int]:
        """Return per-sample lengths for this view (remapped from parent store)."""
        parent_lengths = self._store.get_all_lengths()
        return [parent_lengths[i] for i in self._indices]

    def subset(self, indices) -> "ShardedTokenStoreView":
        """Return a lazy view over a subset of this view (composes index maps)."""
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        else:
            indices = list(indices)
        remapped = [self._indices[int(i)] for i in indices]
        return ShardedTokenStoreView(self._store, remapped)


def _get_store_lengths(store) -> List[int]:
    """Return per-sample sequence lengths from any supported store type."""
    if isinstance(store, (ShardedTokenStore, ShardedTokenStoreView)):
        return store.get_all_lengths()
    # Plain list / array of tensors or ndarrays
    return [int(np.asarray(t).shape[0]) for t in store]


class BucketBatchSampler:
    """Groups samples by similar total token length to minimise padding waste.

    Sorts all indices by ``lengths[i]``, groups them into batches of
    ``batch_size``, then shuffles the *batch order* each epoch (not the
    within-batch order) so the model sees every length-group in a random
    sequence while keeping per-batch padding near-minimal.
    """

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool, seed: int = 0) -> None:
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self._batches: List[List[int]] = [
            sorted_idx[i: i + batch_size] for i in range(0, len(sorted_idx), batch_size)
        ]
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __iter__(self):
        order = list(range(len(self._batches)))
        if self._shuffle:
            rng = np.random.default_rng(self._seed + self._epoch)
            rng.shuffle(order)
        self._epoch += 1
        for i in order:
            yield self._batches[i]

    def __len__(self) -> int:
        return len(self._batches)


class TokenFeatureDataset(Dataset):
    """Dataset storing variable-length per-token embeddings for cross-attention.

    ``env_tokens``, ``vh_tokens``, ``vl_tokens`` can be a ``ShardedTokenStore``,
    ``ShardedTokenStoreView``, or a plain ``List[np.ndarray / torch.Tensor]``.
    Items are fetched lazily so the full token matrix never needs to reside
    entirely in RAM.
    """

    def __init__(
        self,
        ids: np.ndarray,
        ab_ids: np.ndarray,
        env_tokens,
        vh_tokens,
        vl_tokens,
        y: np.ndarray,
        env_pool: Optional[np.ndarray] = None,
        vh_pool: Optional[np.ndarray] = None,
        vl_pool: Optional[np.ndarray] = None,
    ) -> None:
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.ab_ids = torch.as_tensor(ab_ids, dtype=torch.long)
        self.env_tokens = env_tokens
        self.vh_tokens = vh_tokens
        self.vl_tokens = vl_tokens
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.env_pool = None if env_pool is None else torch.as_tensor(env_pool, dtype=torch.float32)
        self.vh_pool = None if vh_pool is None else torch.as_tensor(vh_pool, dtype=torch.float32)
        self.vl_pool = None if vl_pool is None else torch.as_tensor(vl_pool, dtype=torch.float32)

    @staticmethod
    def _to_tensor(store, idx: int) -> torch.Tensor:
        if isinstance(store, (ShardedTokenStore, ShardedTokenStoreView)):
            return store[idx].float()
        return torch.as_tensor(store[idx], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        sample = (
            self.ids[index],
            self.ab_ids[index],
            self._to_tensor(self.env_tokens, index),
            self._to_tensor(self.vh_tokens, index),
            self._to_tensor(self.vl_tokens, index),
            self.y[index],
        )
        if self.env_pool is not None and self.vh_pool is not None and self.vl_pool is not None:
            return sample + (self.env_pool[index], self.vh_pool[index], self.vl_pool[index])
        return sample


class MultiTaskTokenFeatureDataset(Dataset):
    """Multitask variant of TokenFeatureDataset."""

    def __init__(
        self,
        ids: np.ndarray,
        ab_ids: np.ndarray,
        env_tokens,
        vh_tokens,
        vl_tokens,
        y_cls: np.ndarray,
        y_reg: np.ndarray,
        reg_mask: np.ndarray,
        env_pool: Optional[np.ndarray] = None,
        vh_pool: Optional[np.ndarray] = None,
        vl_pool: Optional[np.ndarray] = None,
    ) -> None:
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.ab_ids = torch.as_tensor(ab_ids, dtype=torch.long)
        self.env_tokens = env_tokens
        self.vh_tokens = vh_tokens
        self.vl_tokens = vl_tokens
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.float32)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.reg_mask = torch.as_tensor(reg_mask, dtype=torch.float32)
        self.env_pool = None if env_pool is None else torch.as_tensor(env_pool, dtype=torch.float32)
        self.vh_pool = None if vh_pool is None else torch.as_tensor(vh_pool, dtype=torch.float32)
        self.vl_pool = None if vl_pool is None else torch.as_tensor(vl_pool, dtype=torch.float32)

    @staticmethod
    def _to_tensor(store, idx: int) -> torch.Tensor:
        if isinstance(store, (ShardedTokenStore, ShardedTokenStoreView)):
            return store[idx].float()
        return torch.as_tensor(store[idx], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y_cls.shape[0])

    def __getitem__(self, index: int):
        sample = (
            self.ids[index],
            self.ab_ids[index],
            self._to_tensor(self.env_tokens, index),
            self._to_tensor(self.vh_tokens, index),
            self._to_tensor(self.vl_tokens, index),
            self.y_cls[index],
            self.y_reg[index],
            self.reg_mask[index],
        )
        if self.env_pool is not None and self.vh_pool is not None and self.vl_pool is not None:
            return sample + (self.env_pool[index], self.vh_pool[index], self.vl_pool[index])
        return sample



def _pad_token_batch(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (L_i, D) tensors to (B, max_L, D); mask (B, max_L) True=valid."""
    max_len = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[1]
    padded = torch.zeros(len(tensors), max_len, D, dtype=torch.float32)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
    for i, t in enumerate(tensors):
        L = t.shape[0]
        if L > 0:
            padded[i, :L] = t
            mask[i, :L] = True
    return padded, mask


def _token_pad_collate_fn(batch: List) -> Tuple:
    """Collate for TokenFeatureDataset."""
    has_pool = len(batch[0]) == 9
    if has_pool:
        ids, ab_ids, env_toks, vh_toks, vl_toks, ys, env_pool, vh_pool, vl_pool = zip(*batch)
    else:
        ids, ab_ids, env_toks, vh_toks, vl_toks, ys = zip(*batch)
    env_padded, env_mask = _pad_token_batch(list(env_toks))
    vh_padded, vh_mask = _pad_token_batch(list(vh_toks))
    vl_padded, vl_mask = _pad_token_batch(list(vl_toks))
    out = (
        torch.stack(list(ids)),
        torch.stack(list(ab_ids)),
        env_padded, env_mask,
        vh_padded, vh_mask,
        vl_padded, vl_mask,
        torch.stack(list(ys)),
    )
    if has_pool:
        out = out + (torch.stack(list(env_pool)), torch.stack(list(vh_pool)), torch.stack(list(vl_pool)))
    return out


def _multitask_token_pad_collate_fn(batch: List) -> Tuple:
    """Collate for MultiTaskTokenFeatureDataset."""
    has_pool = len(batch[0]) == 11
    if has_pool:
        ids, ab_ids, env_toks, vh_toks, vl_toks, y_cls, y_reg, reg_mask, env_pool, vh_pool, vl_pool = zip(*batch)
    else:
        ids, ab_ids, env_toks, vh_toks, vl_toks, y_cls, y_reg, reg_mask = zip(*batch)
    env_padded, env_mask = _pad_token_batch(list(env_toks))
    vh_padded, vh_mask = _pad_token_batch(list(vh_toks))
    vl_padded, vl_mask = _pad_token_batch(list(vl_toks))
    out = (
        torch.stack(list(ids)),
        torch.stack(list(ab_ids)),
        env_padded, env_mask,
        vh_padded, vh_mask,
        vl_padded, vl_mask,
        torch.stack(list(y_cls)),
        torch.stack(list(y_reg)),
        torch.stack(list(reg_mask)),
    )
    if has_pool:
        out = out + (torch.stack(list(env_pool)), torch.stack(list(vh_pool)), torch.stack(list(vl_pool)))
    return out


def _effective_token_batch_size(args: argparse.Namespace) -> int:
    token_batch_size = int(getattr(args, "token_batch_size", 0) or 0)
    base_batch_size = max(1, int(args.batch_size))
    if token_batch_size <= 0:
        return base_batch_size
    return max(1, min(base_batch_size, token_batch_size))


def _effective_token_num_workers(args: argparse.Namespace) -> int:
    token_num_workers = getattr(args, "token_num_workers", None)
    if token_num_workers is None:
        return int(args.num_workers)
    return max(0, int(token_num_workers))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ProjectionEncoder(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, hidden_dim: int, dropout: float, num_blocks: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(proj_dim, hidden_dim, dropout) for _ in range(max(0, int(num_blocks)))]
        )
        self.output_norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.blocks(self.input(x)))


class GatedAntibodyFusion(nn.Module):
    def __init__(self, proj_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        feature_dim = 4 * proj_dim
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, z_vh: torch.Tensor, z_vl: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_vh, z_vl, torch.abs(z_vh - z_vl), z_vh * z_vl], dim=-1)
        gate = self.gate(features)
        base = gate * z_vh + (1.0 - gate) * z_vl
        return self.norm(base + self.residual(features))


class EnvConditioner(nn.Module):
    def __init__(self, proj_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, z_env: torch.Tensor, z_ab: torch.Tensor) -> torch.Tensor:
        scale, shift = self.net(z_env).chunk(2, dim=-1)
        scale = 0.5 * torch.tanh(scale)
        shift = 0.5 * torch.tanh(shift)
        return self.norm(z_ab * (1.0 + scale) + shift)


class LowRankBilinearInteraction(nn.Module):
    def __init__(self, proj_dim: int, rank: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        rank = max(1, int(rank))
        self.rank = rank
        self.env_factor = nn.Linear(proj_dim, rank, bias=False)
        self.ab_factor = nn.Linear(proj_dim, rank, bias=False)
        self.out = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, z_env: torch.Tensor, z_ab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        factors = self.env_factor(z_env) * self.ab_factor(z_ab)
        score = factors.sum(dim=-1, keepdim=True) / float(self.rank) ** 0.5
        return self.out(factors), score


class TokenProjector(nn.Module):
    """Project per-token PLM embeddings to a shared proj_dim space."""

    def __init__(self, in_dim: int, proj_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossAttentionEncoder(nn.Module):
    """Multi-layer cross-attention: query tokens attend to key/value tokens.

    Uses pre-LN on the query before attention and post-LN FFN residual.
    """

    def __init__(self, proj_dim: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.q_norms = nn.ModuleList([nn.LayerNorm(proj_dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, proj_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim * 4, proj_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(proj_dim) for _ in range(num_layers)])

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_pad_mask: Optional[torch.Tensor] = None,
        *,
        return_attn_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        query : (B, L_q, D)
        key_value : (B, L_kv, D)
        kv_pad_mask : (B, L_kv) bool, True = padding positions to ignore
        return_attn_weights : If True, also return per-layer attention weights.
        average_attn_weights : Passed to ``nn.MultiheadAttention``; if False, weights are
            ``(B, num_heads, L_q, L_kv)`` per layer instead of averaged over heads.
        """
        x = query
        layer_weights: List[torch.Tensor] = []
        for attn, q_norm, ffn, ffn_norm in zip(self.attn_layers, self.q_norms, self.ffns, self.ffn_norms):
            if return_attn_weights:
                attn_out, w = attn(
                    q_norm(x),
                    key_value,
                    key_value,
                    key_padding_mask=kv_pad_mask,
                    need_weights=True,
                    average_attn_weights=average_attn_weights,
                )
                layer_weights.append(w.detach())
            else:
                attn_out, _ = attn(q_norm(x), key_value, key_value, key_padding_mask=kv_pad_mask)
            x = x + attn_out
            x = ffn_norm(x + ffn(x))
        if return_attn_weights:
            return x, layer_weights
        return x


class PAVECrossAttn(nn.Module):
    """PAVE with token-level cross-attention between Env and antibody (VH+VL) sequences.

    The antibody token sequence (VH concatenated with VL) attends to the Env
    token sequence via multi-layer cross-attention.  The attended representation
    is mean-pooled and fused with a gated VH+VL pooled representation before
    being passed through the same bilinear-interaction and prediction heads as
    the base PAVE model.
    """

    def __init__(
        self,
        in_env: int,
        in_vh: int,
        in_vl: int,
        num_ab_embeddings: int,
        proj_dim: int,
        fusion_hidden: int,
        dropout: float,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        interaction_rank: int = 128,
        num_res_blocks: int = 2,
        head_hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        # Ensure heads divides proj_dim
        cross_attn_heads = max(1, int(cross_attn_heads))
        while proj_dim % cross_attn_heads != 0 and cross_attn_heads > 1:
            cross_attn_heads -= 1
        interaction_rank = interaction_rank if interaction_rank > 0 else max(1, proj_dim // 2)
        head_hidden = max(fusion_hidden, proj_dim * max(1, int(head_hidden_mult)))

        # Per-token projectors
        self.env_tok_proj = TokenProjector(in_env, proj_dim, dropout)
        self.vh_tok_proj = TokenProjector(in_vh, proj_dim, dropout)
        self.vl_tok_proj = TokenProjector(in_vl, proj_dim, dropout)

        # Cross-attention: antibody token queries attend to Env token keys/values
        self.cross_attn = CrossAttentionEncoder(proj_dim, cross_attn_heads, cross_attn_layers, dropout)

        # Residual refinement for pooled representations (used in contrastive losses)
        self.env_pool_refine = nn.Sequential(
            *[ResidualMLPBlock(proj_dim, fusion_hidden, dropout) for _ in range(max(0, int(num_res_blocks)))]
        )
        self.vh_pool_refine = nn.Sequential(
            *[ResidualMLPBlock(proj_dim, fusion_hidden, dropout) for _ in range(max(0, int(num_res_blocks)))]
        )
        self.vl_pool_refine = nn.Sequential(
            *[ResidualMLPBlock(proj_dim, fusion_hidden, dropout) for _ in range(max(0, int(num_res_blocks)))]
        )
        self.pool_norm = nn.LayerNorm(proj_dim)

        self.ab_fusion = GatedAntibodyFusion(proj_dim, fusion_hidden, dropout)

        self.use_ab_id_feature = num_ab_embeddings > 1
        if self.use_ab_id_feature:
            self.ab_id_embedding = nn.Embedding(num_ab_embeddings, proj_dim, padding_idx=0)
            self.ab_id_norm = nn.LayerNorm(proj_dim)
            self.ab_id_gate = nn.Sequential(
                nn.Linear(2 * proj_dim, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, proj_dim),
                nn.Sigmoid(),
            )

        self.env_conditioner = EnvConditioner(proj_dim, fusion_hidden, dropout)
        self.bilinear = LowRankBilinearInteraction(proj_dim, interaction_rank, fusion_hidden, dropout)

        pair_feature_dim = 6 * proj_dim + 2
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_feature_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(pair_feature_dim + proj_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            ResidualMLPBlock(head_hidden, head_hidden * 2, dropout),
            nn.Linear(head_hidden, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(pair_feature_dim + proj_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            ResidualMLPBlock(head_hidden, head_hidden * 2, dropout),
            nn.Linear(head_hidden, 1),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D), mask: (B, L) True=valid -> (B, D)."""
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (x * mask_f).sum(dim=1) / denom

    def forward(
        self,
        env_tok: torch.Tensor,
        vh_tok: torch.Tensor,
        vl_tok: torch.Tensor,
        env_mask: torch.Tensor,
        vh_mask: torch.Tensor,
        vl_mask: torch.Tensor,
        ab_id: Optional[torch.Tensor] = None,
        *,
        return_cross_attn_weights: bool = False,
        cross_attn_average_heads: bool = True,
    ):
        # Project tokens to proj_dim
        env_proj = self.env_tok_proj(env_tok)   # (B, L_env, proj_dim)
        vh_proj = self.vh_tok_proj(vh_tok)       # (B, L_vh,  proj_dim)
        vl_proj = self.vl_tok_proj(vl_tok)       # (B, L_vl,  proj_dim)

        # Masked mean-pool + residual refinement for contrastive loss representations
        z_env = F.normalize(self.pool_norm(self.env_pool_refine(self._masked_mean(env_proj, env_mask))), dim=-1)
        z_vh = F.normalize(self.pool_norm(self.vh_pool_refine(self._masked_mean(vh_proj, vh_mask))), dim=-1)
        z_vl = F.normalize(self.pool_norm(self.vl_pool_refine(self._masked_mean(vl_proj, vl_mask))), dim=-1)

        # Gated fusion of pooled VH and VL (used in heavy-light contrastive loss)
        z_ab_pool = F.normalize(self.ab_fusion(z_vh, z_vl), dim=-1)

        # Cross-attention: concatenated VH+VL tokens attend to Env tokens
        ab_tok = torch.cat([vh_proj, vl_proj], dim=1)       # (B, L_ab, proj_dim)
        ab_mask_cat = torch.cat([vh_mask, vl_mask], dim=1)  # (B, L_ab)
        # MHA key_padding_mask: True = positions to IGNORE (padding)
        env_pad = ~env_mask
        cross_kw = dict(
            return_attn_weights=return_cross_attn_weights,
            average_attn_weights=cross_attn_average_heads,
        )
        if return_cross_attn_weights:
            ab_cross, cross_attn_weights = self.cross_attn(ab_tok, env_proj, env_pad, **cross_kw)
        else:
            ab_cross = self.cross_attn(ab_tok, env_proj, env_pad, **cross_kw)
        z_ab_cross = F.normalize(self._masked_mean(ab_cross, ab_mask_cat), dim=-1)

        # Combine cross-attended and pooled antibody representations
        z_ab = F.normalize(z_ab_cross + z_ab_pool, dim=-1)

        # Optional antibody ID feature
        if self.use_ab_id_feature and ab_id is not None:
            ab_id = ab_id.clamp(min=0, max=self.ab_id_embedding.num_embeddings - 1)
            z_ab_id = self.ab_id_norm(self.ab_id_embedding(ab_id))
            known_mask = (ab_id > 0).unsqueeze(-1).float()
            mix = self.ab_id_gate(torch.cat([z_ab, z_ab_id], dim=-1)) * known_mask
            z_ab = F.normalize((1.0 - mix) * z_ab + mix * z_ab_id, dim=-1)

        z_ab_cond = F.normalize(self.env_conditioner(z_env, z_ab), dim=-1)
        bilinear, bilinear_score = self.bilinear(z_env, z_ab_cond)
        bilinear = F.normalize(bilinear, dim=-1)
        cosine = (z_env * z_ab).sum(dim=-1, keepdim=True)
        conditioned_cosine = (z_env * z_ab_cond).sum(dim=-1, keepdim=True)
        pair_features = torch.cat(
            [
                z_env,
                z_ab,
                torch.abs(z_env - z_ab_cond),
                z_env * z_ab_cond,
                z_ab_cond,
                bilinear,
                cosine,
                conditioned_cosine + bilinear_score,
            ],
            dim=-1,
        )
        pair = F.normalize(self.pair_encoder(pair_features), dim=-1)
        head_input = torch.cat([pair_features, pair], dim=-1)
        pred_cls = self.cls_head(head_input).squeeze(-1)
        pred_reg = self.reg_head(head_input).squeeze(-1)
        if return_cross_attn_weights:
            return pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair, cross_attn_weights
        return pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair


class PAVEHybridCrossAttn(nn.Module):
    """Pooled-main PAVE with token cross-attention gated residual on antibody."""

    def __init__(
        self,
        in_env: int,
        in_vh: int,
        in_vl: int,
        tok_env_dim: int,
        tok_vh_dim: int,
        tok_vl_dim: int,
        num_ab_embeddings: int,
        proj_dim: int,
        fusion_hidden: int,
        dropout: float,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        interaction_rank: int = 128,
        num_res_blocks: int = 2,
        head_hidden_mult: int = 2,
        token_gate_hidden: int = 0,
    ) -> None:
        super().__init__()
        interaction_rank = interaction_rank if interaction_rank > 0 else max(1, proj_dim // 2)
        head_hidden = max(fusion_hidden, proj_dim * max(1, int(head_hidden_mult)))
        token_gate_hidden = int(token_gate_hidden) if int(token_gate_hidden) > 0 else int(fusion_hidden)
        cross_attn_heads = max(1, int(cross_attn_heads))
        while proj_dim % cross_attn_heads != 0 and cross_attn_heads > 1:
            cross_attn_heads -= 1

        # Pooled main trunk (same as PAVE)
        self.env_proj = ProjectionEncoder(in_env, proj_dim, fusion_hidden, dropout, num_res_blocks)
        self.vh_proj = ProjectionEncoder(in_vh, proj_dim, fusion_hidden, dropout, num_res_blocks)
        self.vl_proj = ProjectionEncoder(in_vl, proj_dim, fusion_hidden, dropout, num_res_blocks)
        self.ab_fusion = GatedAntibodyFusion(proj_dim, fusion_hidden, dropout)

        # Token branch
        self.env_tok_proj = TokenProjector(tok_env_dim, proj_dim, dropout)
        self.vh_tok_proj = TokenProjector(tok_vh_dim, proj_dim, dropout)
        self.vl_tok_proj = TokenProjector(tok_vl_dim, proj_dim, dropout)
        self.cross_attn = CrossAttentionEncoder(proj_dim, cross_attn_heads, cross_attn_layers, dropout)

        self.token_gate = nn.Sequential(
            nn.Linear(3 * proj_dim, token_gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_gate_hidden, proj_dim),
            nn.Sigmoid(),
        )

        self.use_ab_id_feature = num_ab_embeddings > 1
        if self.use_ab_id_feature:
            self.ab_id_embedding = nn.Embedding(num_ab_embeddings, proj_dim, padding_idx=0)
            self.ab_id_norm = nn.LayerNorm(proj_dim)
            self.ab_id_gate = nn.Sequential(
                nn.Linear(2 * proj_dim, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, proj_dim),
                nn.Sigmoid(),
            )

        self.env_conditioner = EnvConditioner(proj_dim, fusion_hidden, dropout)
        self.bilinear = LowRankBilinearInteraction(proj_dim, interaction_rank, fusion_hidden, dropout)

        pair_feature_dim = 6 * proj_dim + 2
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_feature_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(pair_feature_dim + proj_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            ResidualMLPBlock(head_hidden, head_hidden * 2, dropout),
            nn.Linear(head_hidden, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(pair_feature_dim + proj_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            ResidualMLPBlock(head_hidden, head_hidden * 2, dropout),
            nn.Linear(head_hidden, 1),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (x * mask_f).sum(dim=1) / denom

    def forward(
        self,
        env_x: torch.Tensor,
        vh_x: torch.Tensor,
        vl_x: torch.Tensor,
        env_tok: torch.Tensor,
        vh_tok: torch.Tensor,
        vl_tok: torch.Tensor,
        env_mask: torch.Tensor,
        vh_mask: torch.Tensor,
        vl_mask: torch.Tensor,
        ab_id: Optional[torch.Tensor] = None,
        *,
        return_cross_attn_weights: bool = False,
        cross_attn_average_heads: bool = True,
    ):
        z_env = F.normalize(self.env_proj(env_x), dim=-1)
        z_vh = F.normalize(self.vh_proj(vh_x), dim=-1)
        z_vl = F.normalize(self.vl_proj(vl_x), dim=-1)
        z_ab_pool = F.normalize(self.ab_fusion(z_vh, z_vl), dim=-1)

        env_tok_proj = self.env_tok_proj(env_tok)
        vh_tok_proj = self.vh_tok_proj(vh_tok)
        vl_tok_proj = self.vl_tok_proj(vl_tok)
        ab_tok = torch.cat([vh_tok_proj, vl_tok_proj], dim=1)
        ab_mask_cat = torch.cat([vh_mask, vl_mask], dim=1)
        cross_kw = dict(
            return_attn_weights=return_cross_attn_weights,
            average_attn_weights=cross_attn_average_heads,
        )
        if return_cross_attn_weights:
            ab_cross, cross_attn_weights = self.cross_attn(ab_tok, env_tok_proj, ~env_mask, **cross_kw)
        else:
            ab_cross = self.cross_attn(ab_tok, env_tok_proj, ~env_mask, **cross_kw)
        z_ab_token = F.normalize(self._masked_mean(ab_cross, ab_mask_cat), dim=-1)
        gate = self.token_gate(torch.cat([z_ab_pool, z_ab_token, z_env], dim=-1))
        z_ab = F.normalize(z_ab_pool + gate * z_ab_token, dim=-1)

        if self.use_ab_id_feature and ab_id is not None:
            ab_id = ab_id.clamp(min=0, max=self.ab_id_embedding.num_embeddings - 1)
            z_ab_id = self.ab_id_norm(self.ab_id_embedding(ab_id))
            known_mask = (ab_id > 0).unsqueeze(-1).float()
            mix = self.ab_id_gate(torch.cat([z_ab, z_ab_id], dim=-1)) * known_mask
            z_ab = F.normalize((1.0 - mix) * z_ab + mix * z_ab_id, dim=-1)

        z_ab_cond = F.normalize(self.env_conditioner(z_env, z_ab), dim=-1)
        bilinear, bilinear_score = self.bilinear(z_env, z_ab_cond)
        bilinear = F.normalize(bilinear, dim=-1)
        cosine = (z_env * z_ab).sum(dim=-1, keepdim=True)
        conditioned_cosine = (z_env * z_ab_cond).sum(dim=-1, keepdim=True)
        pair_features = torch.cat(
            [
                z_env,
                z_ab,
                torch.abs(z_env - z_ab_cond),
                z_env * z_ab_cond,
                z_ab_cond,
                bilinear,
                cosine,
                conditioned_cosine + bilinear_score,
            ],
            dim=-1,
        )
        pair = F.normalize(self.pair_encoder(pair_features), dim=-1)
        head_input = torch.cat([pair_features, pair], dim=-1)
        pred_cls = self.cls_head(head_input).squeeze(-1)
        pred_reg = self.reg_head(head_input).squeeze(-1)
        if return_cross_attn_weights:
            return pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair, gate, cross_attn_weights
        return pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair, gate


class PAVE(nn.Module):
    def __init__(
        self,
        in_env: int,
        in_vh: int,
        in_vl: int,
        num_ab_embeddings: int,
        proj_dim: int,
        fusion_hidden: int,
        dropout: float,
        interaction_rank: int = 128,
        num_res_blocks: int = 2,
        head_hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        interaction_rank = interaction_rank if interaction_rank > 0 else max(1, proj_dim // 2)
        head_hidden = max(fusion_hidden, proj_dim * max(1, int(head_hidden_mult)))

        self.env_proj = ProjectionEncoder(in_env, proj_dim, fusion_hidden, dropout, num_res_blocks)
        self.vh_proj = ProjectionEncoder(in_vh, proj_dim, fusion_hidden, dropout, num_res_blocks)
        self.vl_proj = ProjectionEncoder(in_vl, proj_dim, fusion_hidden, dropout, num_res_blocks)
        self.ab_fusion = GatedAntibodyFusion(proj_dim, fusion_hidden, dropout)
        self.use_ab_id_feature = num_ab_embeddings > 1
        if self.use_ab_id_feature:
            self.ab_id_embedding = nn.Embedding(num_ab_embeddings, proj_dim, padding_idx=0)
            self.ab_id_norm = nn.LayerNorm(proj_dim)
            self.ab_id_gate = nn.Sequential(
                nn.Linear(2 * proj_dim, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, proj_dim),
                nn.Sigmoid(),
            )
        self.env_conditioner = EnvConditioner(proj_dim, fusion_hidden, dropout)
        self.bilinear = LowRankBilinearInteraction(proj_dim, interaction_rank, fusion_hidden, dropout)

        pair_feature_dim = 6 * proj_dim + 2
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_feature_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(pair_feature_dim + proj_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            ResidualMLPBlock(head_hidden, head_hidden * 2, dropout),
            nn.Linear(head_hidden, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(pair_feature_dim + proj_dim, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            ResidualMLPBlock(head_hidden, head_hidden * 2, dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        env_x: torch.Tensor,
        vh_x: torch.Tensor,
        vl_x: torch.Tensor,
        ab_id: Optional[torch.Tensor] = None,
    ):
        z_env = F.normalize(self.env_proj(env_x), dim=-1)
        z_vh = F.normalize(self.vh_proj(vh_x), dim=-1)
        z_vl = F.normalize(self.vl_proj(vl_x), dim=-1)
        z_ab = F.normalize(self.ab_fusion(z_vh, z_vl), dim=-1)
        if self.use_ab_id_feature and ab_id is not None:
            ab_id = ab_id.clamp(min=0, max=self.ab_id_embedding.num_embeddings - 1)
            z_ab_id = self.ab_id_norm(self.ab_id_embedding(ab_id))
            known_mask = (ab_id > 0).unsqueeze(-1).float()
            mix = self.ab_id_gate(torch.cat([z_ab, z_ab_id], dim=-1)) * known_mask
            z_ab = F.normalize((1.0 - mix) * z_ab + mix * z_ab_id, dim=-1)
        z_ab_cond = F.normalize(self.env_conditioner(z_env, z_ab), dim=-1)
        bilinear, bilinear_score = self.bilinear(z_env, z_ab_cond)
        bilinear = F.normalize(bilinear, dim=-1)
        cosine = (z_env * z_ab).sum(dim=-1, keepdim=True)
        conditioned_cosine = (z_env * z_ab_cond).sum(dim=-1, keepdim=True)
        pair_features = torch.cat(
            [
                z_env,
                z_ab,
                torch.abs(z_env - z_ab_cond),
                z_env * z_ab_cond,
                z_ab_cond,
                bilinear,
                cosine,
                conditioned_cosine + bilinear_score,
            ],
            dim=-1,
        )
        pair = F.normalize(self.pair_encoder(pair_features), dim=-1)
        head_input = torch.cat([pair_features, pair], dim=-1)
        pred_cls = self.cls_head(head_input).squeeze(-1)
        pred_reg = self.reg_head(head_input).squeeze(-1)
        return pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair


def _info_nce_diag(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    if z_a.shape[0] < 2:
        return z_a.new_zeros(())
    temp = max(float(temperature), 1e-6)
    logits = z_a @ z_b.t() / temp
    target = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - pt).pow(gamma) * bce).mean()


def _virus_antibody_loss_classification(
    z_env: torch.Tensor,
    z_ab: torch.Tensor,
    y_bin: torch.Tensor,
    temperature: float,
    use_hard_negative: bool,
    hard_negative_topk: int,
) -> torch.Tensor:
    if z_env.shape[0] < 2:
        return z_env.new_zeros(())
    temp = max(float(temperature), 1e-6)
    sim = z_env @ z_ab.t() / temp
    y_bin = y_bin.float()
    pos_idx = torch.where(y_bin > 0.5)[0]
    if pos_idx.numel() == 0:
        return z_env.new_zeros(())
    log_prob = F.log_softmax(sim, dim=1)
    base = -log_prob[pos_idx, pos_idx].mean()
    if not use_hard_negative:
        return base

    neg_idx = torch.where(y_bin <= 0.5)[0]
    if neg_idx.numel() == 0:
        return base
    hard_losses = []
    for pos in pos_idx.tolist():
        neg_sims = sim[pos, neg_idx]
        k = min(int(hard_negative_topk), int(neg_sims.numel()))
        if k > 0:
            hard = torch.topk(neg_sims, k=k, largest=True).values.mean()
            hard_losses.append(F.relu(0.2 - sim[pos, pos] + hard))
    if not hard_losses:
        return base
    return base + torch.stack(hard_losses).mean()


def _virus_antibody_loss_regression(
    z_env: torch.Tensor,
    z_ab: torch.Tensor,
    y_reg: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if z_env.shape[0] < 2:
        return z_env.new_zeros(())
    temp = max(float(temperature), 1e-6)
    sim = z_env @ z_ab.t() / temp
    y = y_reg.float()
    shifted = y - y.min()
    scale = shifted.std().clamp(min=1e-6)
    weights = torch.exp(-shifted / scale).detach()
    log_prob = F.log_softmax(sim, dim=1)
    return -((weights * torch.diag(log_prob)).sum() / weights.sum().clamp(min=1e-6))


def _supcon_loss_classification(z: torch.Tensor, y_bin: torch.Tensor, temperature: float) -> torch.Tensor:
    n = z.shape[0]
    if n < 3:
        return z.new_zeros(())
    temp = max(float(temperature), 1e-6)
    sim = z @ z.t() / temp
    eye = torch.eye(n, device=z.device)
    sim = sim - eye * 1e9
    labels = y_bin.view(-1, 1)
    positives = (labels == labels.t()).float() - eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = positives.sum(dim=1).clamp(min=1.0)
    return (-(positives * log_prob).sum(dim=1) / denom).mean()


def _supcon_loss_regression(
    z: torch.Tensor,
    y_reg: torch.Tensor,
    sigma: float,
    temperature: float,
) -> torch.Tensor:
    n = z.shape[0]
    if n < 3:
        return z.new_zeros(())
    temp = max(float(temperature), 1e-6)
    sigma = max(float(sigma), 1e-6)
    sim = z @ z.t() / temp
    eye = torch.eye(n, device=z.device)
    sim = sim - eye * 1e9
    y = y_reg.view(-1, 1)
    weights = torch.exp(-torch.abs(y - y.t()) / sigma) * (1.0 - eye)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((weights * log_prob).sum() / weights.sum().clamp(min=1e-6))


def _default_data_path(task: str, variant: str = "default") -> Path:
    variant = str(variant or "default").strip().lower()
    if variant not in DATASET_VARIANT_CHOICES:
        raise ValueError(
            f"dataset_variant must be one of {list(DATASET_VARIANT_CHOICES)}, got {variant!r}"
        )
    if task in {"classification", "multitask"}:
        return DEFAULT_CLASSIFICATION_JSON_CLEAN if variant == "clean" else DEFAULT_CLASSIFICATION_JSON
    return DEFAULT_REGRESSION_JSON_CLEAN if variant == "clean" else DEFAULT_REGRESSION_JSON


def _resolve_training_data_path(args: argparse.Namespace) -> Path:
    if args.data_json:
        return Path(args.data_json).expanduser().resolve()
    variant = str(getattr(args, "dataset_variant", "default") or "default")
    return _default_data_path(args.task, variant).resolve()


def _load_json_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    data = pd.DataFrame(payload)
    if data.empty:
        raise ValueError(f"No records found in {path}")
    return data


def _clean_sequence(value: Any, *, strip_stop: bool, strip_gaps: bool) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    seq = str(value).strip().replace(" ", "").replace("\n", "").replace("\r", "").upper()
    if strip_stop:
        seq = seq.replace("*", "")
    if strip_gaps:
        seq = seq.replace("-", "")
    return seq


def _prepare_dataframe_from_frame(
    data: pd.DataFrame,
    args: argparse.Namespace,
    *,
    independent: bool = False,
    source: str = "dataframe",
) -> pd.DataFrame:
    data = data.reset_index(drop=True).copy()
    data["__row_id"] = np.arange(len(data), dtype=np.int64)
    if "idx" in data.columns:
        data["__display_id"] = pd.to_numeric(data["idx"], errors="coerce").fillna(data["__row_id"]).astype(np.int64)
    else:
        data["__display_id"] = data["__row_id"].astype(np.int64)

    required = [args.env_seq_col, args.vh_seq_col, args.vl_seq_col]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required sequence columns in {source}: {missing}")
    for col in required:
        data[col] = [
            _clean_sequence(v, strip_stop=args.strip_stop, strip_gaps=args.strip_gaps)
            for v in data[col].tolist()
        ]
    data = data[(data[args.env_seq_col] != "") & (data[args.vh_seq_col] != "") & (data[args.vl_seq_col] != "")].copy()

    if args.task == "multitask":
        cls_target_col = args.ind_label_col if independent and args.ind_label_col else args.label_col
        if cls_target_col not in data.columns:
            raise KeyError(f"Missing multitask classification target column {cls_target_col!r} in {source}")
        reg_target_col = args.ind_reg_label_col if independent and args.ind_reg_label_col else args.reg_label_col
        if reg_target_col not in data.columns and independent and "logic50" in data.columns:
            reg_target_col = "logic50"
        if reg_target_col not in data.columns:
            raise KeyError(f"Missing multitask regression target column {reg_target_col!r} in {source}")
        data["__target_cls"] = pd.to_numeric(data[cls_target_col], errors="coerce")
        data["__target_reg"] = pd.to_numeric(data[reg_target_col], errors="coerce")
        data["__target"] = data["__target_cls"]
        data = data.dropna(subset=["__target_cls"]).copy()
    else:
        target_col = args.ind_label_col if independent and args.ind_label_col else args.label_col
        use_ic50_binary_fallback = False
        if target_col not in data.columns:
            if independent and args.task == "regression" and "logic50" in data.columns:
                target_col = "logic50"
            elif (
                independent
                and args.task == "classification"
                and target_col == args.label_col
                and "ic50" in data.columns
            ):
                # VRC601-style cohorts: no binary_label_50 column; derive from ic50.
                use_ic50_binary_fallback = True
            else:
                raise KeyError(f"Missing target column {target_col!r} in {source}")

        if use_ic50_binary_fallback:
            data["__target"] = [_ic50_to_binary_label50(v) for v in data["ic50"].tolist()]
        else:
            data["__target"] = pd.to_numeric(data[target_col], errors="coerce")

        if args.task == "regression":
            if args.exact_only and not independent and "censor_type" in data.columns:
                data = data[data["censor_type"].astype(str).str.lower() == "exact"].copy()
            data = data.dropna(subset=["__target"]).copy()
        else:
            data = data.dropna(subset=["__target"]).copy()

    if not independent and args.limit_rows > 0:
        data = data.head(args.limit_rows).copy()
    if independent and args.ind_limit_rows > 0:
        data = data.head(args.ind_limit_rows).copy()

    data = data.reset_index(drop=True)
    data["__row_id"] = np.arange(len(data), dtype=np.int64)
    if data.empty:
        raise ValueError(f"No usable records remain after filtering {source}")
    return data


def _prepare_dataframe(
    args: argparse.Namespace,
    path: str | Path,
    *,
    independent: bool = False,
) -> pd.DataFrame:
    path = Path(path)
    return _prepare_dataframe_from_frame(
        _load_json_records(path),
        args,
        independent=independent,
        source=str(path),
    )


def _normalize_antibody_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _first_existing_column(data: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in data.columns:
            return column
    return None


def _ic50_to_binary_label50(value: Any) -> float:
    """CH505-aligned binary_label_50 from ic50: 1 = sensitive (IC50 <= 50 µg/ml), 0 = resistant/censored."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    if text.startswith(">"):
        return 0.0
    if text.startswith("<"):
        return 1.0
    try:
        ic50 = float(text)
    except ValueError:
        return float("nan")
    return 1.0 if ic50 <= 50.0 else 0.0


def _load_antibody_name_to_raw_id(path: str | Path) -> Dict[str, int]:
    mapping_path = Path(path)
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in antibody id mapping file {mapping_path}")
    raw_mapping = payload.get("name_to_id", payload)
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Expected a name_to_id object in antibody id mapping file {mapping_path}")
    out: Dict[str, int] = {}
    for name, raw_id in raw_mapping.items():
        normalized = _normalize_antibody_name(name)
        if not normalized:
            continue
        out[normalized] = int(raw_id)
    return out


def _extract_antibody_name_to_raw_id(data: pd.DataFrame) -> Dict[str, int]:
    name_col = _first_existing_column(data, ANTIBODY_NAME_COLUMNS)
    if name_col is None or "ab_id" not in data.columns:
        return {}
    raw_ids = pd.to_numeric(data["ab_id"], errors="coerce")
    mapping: Dict[str, int] = {}
    for name, raw_id in zip(data[name_col].tolist(), raw_ids.tolist()):
        normalized = _normalize_antibody_name(name)
        if not normalized or pd.isna(raw_id):
            continue
        raw_int = int(raw_id)
        prev = mapping.get(normalized)
        if prev is not None and prev != raw_int:
            raise ValueError(
                f"Antibody {normalized!r} maps to multiple ab_id values ({prev} vs {raw_int}) in the dataset."
            )
        mapping[normalized] = raw_int
    return mapping


def _attach_antibody_id_feature(
    data: pd.DataFrame,
    args: argparse.Namespace,
    *,
    independent: bool,
    known_name_to_raw_id: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, int]]:
    name_col = _first_existing_column(data, ANTIBODY_NAME_COLUMNS)
    name_series = (
        data[name_col].map(_normalize_antibody_name)
        if name_col is not None
        else pd.Series([""] * len(data), index=data.index, dtype="object")
    )
    file_mapping = _load_antibody_name_to_raw_id(args.antibody_id_mapping_json) if args.antibody_id_mapping_json else {}
    merged_name_to_raw_id: Dict[str, int] = dict(file_mapping)
    if known_name_to_raw_id:
        merged_name_to_raw_id.update(known_name_to_raw_id)
    merged_name_to_raw_id.update(_extract_antibody_name_to_raw_id(data))

    if merged_name_to_raw_id:
        raw_from_name = name_series.map(merged_name_to_raw_id)
    else:
        raw_from_name = pd.Series([np.nan] * len(data), index=data.index, dtype=np.float64)

    if "ab_id" in data.columns:
        raw_from_column = pd.to_numeric(data["ab_id"], errors="coerce")
        both_mask = raw_from_column.notna() & raw_from_name.notna()
        raw_from_column_int = raw_from_column.fillna(-1).astype(np.int64)
        raw_from_name_int = raw_from_name.fillna(-1).astype(np.int64)
        mismatch_mask = both_mask & (raw_from_column_int != raw_from_name_int)
        if mismatch_mask.any():
            bad = sorted(name_series[mismatch_mask].unique().tolist())[:5]
            raise ValueError(
                f"Found ab_id mismatches between dataset and antibody mapping for antibodies: {bad}"
            )
    else:
        raw_from_column = pd.Series([np.nan] * len(data), index=data.index, dtype=np.float64)

    raw_final = raw_from_column.where(raw_from_column.notna(), raw_from_name).astype("Int64")
    raw_final_values = raw_final.fillna(-1).astype(np.int64).to_numpy()
    ab_index = np.where(raw_final_values >= 0, raw_final_values + 1, 0).astype(np.int64)

    updated = data.copy()
    updated["ab_id"] = raw_final_values
    updated["__ab_id_index"] = ab_index

    missing_mask = raw_final.isna()
    missing_names = sorted({name for name in name_series[missing_mask].tolist() if name})
    observed_max_raw_id = int(raw_final_values.max()) if raw_final_values.size > 0 else -1
    max_raw_id = max(max(merged_name_to_raw_id.values(), default=-1), observed_max_raw_id)
    info = {
        "enabled": True,
        "independent": bool(independent),
        "name_column": name_col,
        "mapping_json": str(Path(args.antibody_id_mapping_json).resolve()) if args.antibody_id_mapping_json else None,
        "resolved_rows": int((~missing_mask).sum()),
        "missing_rows": int(missing_mask.sum()),
        "resolved_ratio": float((~missing_mask).mean()) if len(updated) > 0 else 0.0,
        "num_antibodies_in_mapping": int(len(merged_name_to_raw_id)),
        "num_embedding_tokens": int(max_raw_id + 2),
        "missing_antibodies_preview": missing_names[:20],
    }
    if info["missing_rows"] > 0:
        prefix = "Independent" if independent else "Training"
        _print_antibody_id_mapping_warning(
            prefix=prefix,
            missing_rows=int(info["missing_rows"]),
            total_rows=len(updated),
            examples=missing_names[:5],
        )
    return updated, info, merged_name_to_raw_id


def _make_split_data(
    data: pd.DataFrame,
    args: argparse.Namespace,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
) -> SplitData:
    y_raw = data["__target"].to_numpy()
    train_pos = np.asarray(train_pos, dtype=np.int64)
    val_pos = np.asarray(val_pos, dtype=np.int64)
    test_pos = np.asarray(test_pos, dtype=np.int64)

    y_mean = 0.0
    y_std = 1.0
    class_label_map: Optional[Dict[Any, int]] = None
    if args.task in {"classification", "multitask"}:
        labels = np.unique(y_raw)
        if len(labels) != 2:
            raise ValueError(f"Classification expects exactly two labels, got {labels.tolist()}")
        class_label_map = {label: i for i, label in enumerate(labels)}
        y_all = np.asarray([class_label_map[v] for v in y_raw], dtype=np.float32)
    else:
        y_all = y_raw.astype(np.float32)
        train_y = y_all[train_pos]
        y_mean = float(np.mean(train_y))
        y_std = float(np.std(train_y) + 1e-8)
        y_all = (y_all - y_mean) / y_std

    split = SplitData(
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
        y_train=y_all[train_pos].astype(np.float32),
        y_val=y_all[val_pos].astype(np.float32),
        y_test=y_all[test_pos].astype(np.float32),
        y_mean=y_mean,
        y_std=y_std,
        class_label_map=class_label_map,
    )
    if args.task == "multitask":
        if "__target_cls" not in data.columns or "__target_reg" not in data.columns:
            raise KeyError("Multitask training expects __target_cls and __target_reg columns.")
        y_cls_raw = data["__target_cls"].to_numpy()
        y_cls_all = np.asarray([class_label_map[v] for v in y_cls_raw], dtype=np.float32)
        y_reg_raw = pd.to_numeric(data["__target_reg"], errors="coerce").to_numpy(dtype=np.float64)
        reg_mask_all = np.isfinite(y_reg_raw)
        train_reg = y_reg_raw[train_pos][reg_mask_all[train_pos]]
        if train_reg.size == 0:
            raise ValueError("Multitask training requires at least one regression label in the train split.")
        split.y_mean = float(np.mean(train_reg))
        split.y_std = float(np.std(train_reg) + 1e-8)
        y_reg_all = np.zeros(len(data), dtype=np.float32)
        if reg_mask_all.any():
            y_reg_all[reg_mask_all] = ((y_reg_raw[reg_mask_all] - split.y_mean) / split.y_std).astype(np.float32)
        split.y_train = y_cls_all[train_pos].astype(np.float32)
        split.y_val = y_cls_all[val_pos].astype(np.float32)
        split.y_test = y_cls_all[test_pos].astype(np.float32)
        split.y_cls_train = y_cls_all[train_pos].astype(np.float32)
        split.y_cls_val = y_cls_all[val_pos].astype(np.float32)
        split.y_cls_test = y_cls_all[test_pos].astype(np.float32)
        split.y_reg_train = y_reg_all[train_pos].astype(np.float32)
        split.y_reg_val = y_reg_all[val_pos].astype(np.float32)
        split.y_reg_test = y_reg_all[test_pos].astype(np.float32)
        split.reg_mask_train = reg_mask_all[train_pos].astype(np.float32)
        split.reg_mask_val = reg_mask_all[val_pos].astype(np.float32)
        split.reg_mask_test = reg_mask_all[test_pos].astype(np.float32)
    return split


def _align_train_labels(
    train_pos: np.ndarray,
    y_train: np.ndarray,
    train_pos_fit: np.ndarray,
) -> np.ndarray:
    """Map labels aligned to train_pos onto a (possibly duplicated) train_pos_fit index array."""
    train_pos = np.asarray(train_pos, dtype=np.int64)
    y_train = np.asarray(y_train, dtype=np.float32)
    train_pos_fit = np.asarray(train_pos_fit, dtype=np.int64)
    if train_pos.size != y_train.size:
        raise ValueError(
            f"train_pos length ({train_pos.size}) must match y_train length ({y_train.size})"
        )
    pos_to_idx = {int(p): i for i, p in enumerate(train_pos)}
    missing = [int(p) for p in train_pos_fit if int(p) not in pos_to_idx]
    if missing:
        raise ValueError(
            f"Balanced train positions not found in fold train split (first missing: {missing[:5]})"
        )
    return y_train[[pos_to_idx[int(p)] for p in train_pos_fit]]


def _balance_train_indices_by_antibody(
    data: pd.DataFrame,
    train_pos: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    fold_index: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Upsample minority class per antibody within a fold's training indices."""
    train_pos = np.asarray(train_pos, dtype=np.int64)
    y_train = np.asarray(y_train, dtype=np.float32)
    if train_pos.size != y_train.size:
        raise ValueError(
            f"train_pos length ({train_pos.size}) must match y_train length ({y_train.size})"
        )
    if train_pos.size == 0:
        return train_pos.copy(), {
            "enabled": True,
            "strategy": "upsample",
            "fold": int(fold_index),
            "train_raw": 0,
            "train_balanced": 0,
            "per_antibody": {},
            "skipped_single_class_antibodies": [],
        }

    ab_col = _first_existing_column(data, ANTIBODY_NAME_COLUMNS)
    if ab_col is None:
        raise KeyError(
            "balance_train_by_antibody requires an antibody column "
            f"({', '.join(ANTIBODY_NAME_COLUMNS)}) in the dataset"
        )

    ab_names = data.iloc[train_pos][ab_col].map(_normalize_antibody_name).to_numpy()
    rng = np.random.default_rng(int(args.seed) + int(fold_index) + _TRAIN_BALANCE_SEED_OFFSET)

    balanced_parts: List[np.ndarray] = []
    per_antibody: Dict[str, Any] = {}
    skipped_abs: List[str] = []

    unique_abs = sorted({n if n else "__missing_antibody__" for n in ab_names.tolist()})
    for ab_key in unique_abs:
        if ab_key == "__missing_antibody__":
            mask = ab_names == ""
        else:
            mask = ab_names == ab_key
        idx = train_pos[mask]
        y_sub = y_train[mask]
        sensitive = idx[y_sub >= 0.5]
        resistant = idx[y_sub < 0.5]
        n_s_before = int(sensitive.size)
        n_r_before = int(resistant.size)

        if n_s_before == 0 or n_r_before == 0:
            balanced_parts.append(idx)
            skipped_abs.append(ab_key)
            per_antibody[ab_key] = {
                "n_sensitive_before": n_s_before,
                "n_resistant_before": n_r_before,
                "n_sensitive_after": n_s_before,
                "n_resistant_after": n_r_before,
                "target_per_class": None,
                "skipped_single_class": True,
            }
            continue

        target = max(n_s_before, n_r_before)
        if n_s_before < target:
            extra = rng.choice(sensitive, size=target - n_s_before, replace=True)
            sensitive = np.concatenate([sensitive, extra])
        if n_r_before < target:
            extra = rng.choice(resistant, size=target - n_r_before, replace=True)
            resistant = np.concatenate([resistant, extra])

        balanced_parts.append(np.concatenate([sensitive, resistant]))
        per_antibody[ab_key] = {
            "n_sensitive_before": n_s_before,
            "n_resistant_before": n_r_before,
            "n_sensitive_after": int(sensitive.size),
            "n_resistant_after": int(resistant.size),
            "target_per_class": int(target),
            "skipped_single_class": False,
        }

    balanced = np.concatenate(balanced_parts) if balanced_parts else train_pos.copy()
    rng.shuffle(balanced)
    if skipped_abs:
        _print_train_balance_single_class_warning(fold_index, skipped_abs)

    return balanced.astype(np.int64), {
        "enabled": True,
        "strategy": "upsample",
        "fold": int(fold_index),
        "train_raw": int(train_pos.size),
        "train_balanced": int(balanced.size),
        "antibody_column": ab_col,
        "per_antibody": per_antibody,
        "skipped_single_class_antibodies": skipped_abs,
    }


def _resolve_train_positions_for_training(
    split: SplitData,
    data: pd.DataFrame,
    args: argparse.Namespace,
    fold_index: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    train_pos = np.asarray(split.train_pos, dtype=np.int64)
    if not getattr(args, "balance_train_by_antibody", False) or args.task == "regression":
        return train_pos, {"enabled": False, "strategy": None, "train_raw": int(train_pos.size), "train_balanced": int(train_pos.size)}

    if args.task == "multitask":
        if split.y_cls_train is None:
            raise ValueError("Multitask split is missing y_cls_train for train balancing.")
        y_train = split.y_cls_train
    else:
        y_train = split.y_train

    return _balance_train_indices_by_antibody(data, train_pos, y_train, args, fold_index)


def _split_data(data: pd.DataFrame, args: argparse.Namespace) -> SplitData:
    y_raw = data["__target"].to_numpy()
    positions = np.arange(len(data), dtype=np.int64)
    group_col = args.group_col
    if group_col == "random":
        groups = positions.astype(str)
    else:
        if group_col not in data.columns:
            raise KeyError(f"group_col={group_col!r} not found in dataset columns")
        groups = data[group_col].fillna("__missing_group__").astype(str).to_numpy()

    if group_col == "random" or len(np.unique(groups)) < 3:
        rng = np.random.default_rng(args.seed)
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * args.test_size)))
        n_val = max(1, int(round(len(shuffled) * args.val_size)))
        test_pos = shuffled[:n_test]
        val_pos = shuffled[n_test:n_test + n_val]
        train_pos = shuffled[n_test + n_val:]
    else:
        outer = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_val_rel, test_rel = next(outer.split(positions, y_raw, groups))
        rel_val_size = args.val_size / (1.0 - args.test_size)
        inner = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=args.seed + 1)
        train_rel, val_rel = next(inner.split(positions[train_val_rel], y_raw[train_val_rel], groups[train_val_rel]))
        train_pos = positions[train_val_rel][train_rel]
        val_pos = positions[train_val_rel][val_rel]
        test_pos = positions[test_rel]

    return _make_split_data(data, args, train_pos, val_pos, test_pos)


def _group_kfold_splits(data: pd.DataFrame, args: argparse.Namespace) -> List[SplitData]:
    y_raw = data["__target"].to_numpy()
    positions = np.arange(len(data), dtype=np.int64)
    group_col = args.group_col
    num_folds = int(args.num_folds)
    if num_folds < 2:
        raise ValueError(f"--num_folds must be >= 2, got {num_folds}")

    random_split = group_col == "random"
    if random_split:
        groups = None
        outer = KFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
    else:
        if group_col not in data.columns:
            raise KeyError(f"group_col={group_col!r} not found in dataset columns")
        groups = data[group_col].fillna("__missing_group__").astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < num_folds:
            raise ValueError(
                f"GroupKFold requires at least {num_folds} unique groups, got {len(unique_groups)} "
                f"from group_col={group_col!r}"
            )
        outer = GroupKFold(n_splits=num_folds)

    splits: List[SplitData] = []
    rel_val_size = args.val_size / max(1e-12, 1.0 - 1.0 / float(num_folds))
    rel_val_size = min(max(float(rel_val_size), 1e-6), 0.5)

    for fold_index, (train_val_rel, test_rel) in enumerate(outer.split(positions, y_raw, groups), start=1):
        train_val_pos = positions[train_val_rel]
        if random_split:
            inner = ShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=args.seed + fold_index)
            train_rel, val_rel = next(inner.split(train_val_pos, y_raw[train_val_rel]))
        else:
            train_val_groups = groups[train_val_rel]
            if len(np.unique(train_val_groups)) < 2:
                raise ValueError(f"Fold {fold_index} has fewer than two train/val groups; cannot create validation split")
            inner = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=args.seed + fold_index)
            train_rel, val_rel = next(inner.split(train_val_pos, y_raw[train_val_rel], train_val_groups))
        split = _make_split_data(
            data,
            args,
            train_val_pos[train_rel],
            train_val_pos[val_rel],
            positions[test_rel],
        )
        splits.append(split)
    return splits


def _fingerprint_records(ids: Sequence[int], sequences: Sequence[str], model_name: str, args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    digest.update(str(args.embedding_backend).encode("utf-8"))
    digest.update(model_name.encode("utf-8"))
    digest.update(str(args.embedding_batch_size).encode("utf-8"))
    digest.update(str(args.hash_embedding_dim).encode("utf-8"))
    digest.update(str(args.embedding_pool_override or "").encode("utf-8"))
    digest.update(str(args.embedding_max_length or "").encode("utf-8"))
    digest.update(str(args.strip_stop).encode("utf-8"))
    digest.update(str(args.strip_gaps).encode("utf-8"))
    for row_id, seq in zip(ids, sequences):
        digest.update(str(int(row_id)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(seq.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def _hash_sequence_features(sequences: Sequence[str], dim: int) -> np.ndarray:
    """Deterministic lightweight features for smoke tests without loading backbones."""
    alphabet = "ACDEFGHIKLMNPQRSTVWYXBZUOJ"
    features = np.zeros((len(sequences), dim), dtype=np.float32)
    for row, seq in enumerate(sequences):
        if not seq:
            continue
        for pos, char in enumerate(seq):
            bucket = (ord(char) * 131 + pos * 17) % dim
            features[row, bucket] += 1.0
            if char in alphabet:
                features[row, alphabet.index(char) % dim] += 1.0
        norm = np.linalg.norm(features[row])
        if norm > 0:
            features[row] /= norm
    return features


def _hash_token_features(seq: str, dim: int) -> np.ndarray:
    seq = _sequence_value(seq, keep_gaps=False)
    if not seq:
        return np.zeros((0, dim), dtype=np.float32)
    features = np.zeros((len(seq), dim), dtype=np.float32)
    alphabet = "ACDEFGHIKLMNPQRSTVWYXBZUOJ"
    for pos, char in enumerate(seq):
        bucket = (ord(char) * 131 + pos * 17) % dim
        features[pos, bucket] = 1.0
        if char in alphabet:
            features[pos, alphabet.index(char) % dim] += 1.0
        norm = np.linalg.norm(features[pos])
        if norm > 0:
            features[pos] /= norm
    return features


def _to_numpy_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pool_embedding_array(array: np.ndarray, strategy: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 0:
        raise ValueError("Embedding output is scalar; cannot pool to a feature vector")
    if array.ndim == 1:
        return array.astype(np.float32)
    if array.ndim == 2:
        if array.shape[0] == 1:
            return array[0].astype(np.float32)
        if strategy == "first":
            return array[0].astype(np.float32)
        if strategy == "last":
            return array[-1].astype(np.float32)
        return array.mean(axis=0).astype(np.float32)
    if array.ndim >= 3:
        squeezed = np.squeeze(array)
        if squeezed.ndim < array.ndim:
            return _pool_embedding_array(squeezed, strategy)
        axes = tuple(range(array.ndim - 2))
        reduced = array.mean(axis=axes) if axes else array
        return _pool_embedding_array(reduced, strategy)
    raise ValueError(f"Unsupported embedding shape: {array.shape}")


def _coerce_embedding_batch(output: Any, expected_count: int, pool_strategy: str) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        rows = [_pool_embedding_array(_to_numpy_array(item), pool_strategy) for item in output]
        if len(rows) != expected_count:
            raise ValueError(f"Expected {expected_count} embedding rows, got {len(rows)}")
        return np.stack(rows).astype(np.float32)

    array = _to_numpy_array(output)
    if array.ndim == 2 and array.shape[0] == expected_count:
        return array.astype(np.float32)
    if array.ndim >= 3 and array.shape[0] == expected_count:
        rows = [_pool_embedding_array(array[i], pool_strategy) for i in range(expected_count)]
        return np.stack(rows).astype(np.float32)
    if expected_count == 1:
        return _pool_embedding_array(array, pool_strategy).reshape(1, -1).astype(np.float32)
    raise ValueError(f"Cannot coerce embedding shape {array.shape} into ({expected_count}, D)")


def _coerce_token_embedding_batch(output: Any, expected_count: int) -> List[np.ndarray]:
    if isinstance(output, (list, tuple)):
        if len(output) != expected_count:
            raise ValueError(f"Expected {expected_count} token embedding rows, got {len(output)}")
        rows = []
        for item in output:
            arr = np.asarray(_to_numpy_array(item), dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"Token embedding rows must be 2D (L, D), got {arr.shape}")
            rows.append(arr)
        return rows

    array = np.asarray(_to_numpy_array(output), dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == expected_count:
        return [array[i] for i in range(expected_count)]
    if expected_count == 1 and array.ndim == 2:
        return [array]
    raise ValueError(
        f"Cannot coerce token embedding shape {array.shape}; "
        "the selected Env model must support return_tokens=True."
    )


def _embedding_overrides(args: argparse.Namespace, model_name: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"batch_size": args.embedding_batch_size, "return_numpy": True}
    if args.embedding_pool_override:
        overrides["pool"] = args.embedding_pool_override
    if args.embedding_max_length:
        overrides["max_length"] = args.embedding_max_length
    if model_name.startswith(("LucaOne-", "LucaVirus-")):
        overrides["seq_type"] = "prot"
    return overrides


def _run_parallel_embedding_jobs(
    worker_fn: Any,
    gpu_ids: List[str],
    per_worker_kwargs: List[Dict[str, Any]],
    desc: str,
) -> None:
    """Spawn one process per GPU, each with ``CUDA_VISIBLE_DEVICES`` set in its
    *OS environment before any CUDA code runs*.

    Using ``multiprocessing.Process`` with a custom environment dict avoids the
    file-descriptor inheritance problem that occurs when the parent process has
    already initialised a CUDA context: the CUDA runtime opens ``/dev/nvidiaX``
    fds, which survive ``spawn``; setting ``CUDA_VISIBLE_DEVICES`` inside the
    child function body is then ignored because CUDA has already enumerated the
    devices via the inherited fds.

    By contrast, passing the env-var through the OS environment dict (before the
    child interpreter starts) ensures CUDA sees only the assigned device.
    """
    # We still use multiprocessing.Process; the key trick is to set the env
    # variable through os.environ *in the parent* only for the duration of
    # Process.start(), then restore it afterwards.  Because the child is
    # created with ``spawn``, it inherits a *copy* of the environment at the
    # moment of os.fork/os.exec — so each child sees only its own GPU.

    ctx = multiprocessing.get_context("spawn")
    procs: List[multiprocessing.Process] = []
    saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    for gpu_id, kwargs in zip(gpu_ids, per_worker_kwargs):
        # Temporarily set CVD so the spawned child inherits the right value.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        p = ctx.Process(target=worker_fn, kwargs=kwargs)
        p.start()
        procs.append(p)

    # Restore parent's CUDA_VISIBLE_DEVICES
    if saved_cvd is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd

    for p in tqdm(procs, desc=desc, leave=False):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Embedding worker process exited with code {p.exitcode}. "
                "Check stderr for details."
            )


def _get_embedding_gpu_ids(args: argparse.Namespace) -> List[str]:
    """Return the list of physical GPU ID strings to use for parallel embedding.

    Respects ``CUDA_VISIBLE_DEVICES`` and ``--embedding_num_gpus``.
    Returns a list of length 1 when multi-GPU is disabled or unavailable.
    """
    num_avail = torch.cuda.device_count()
    if num_avail <= 1:
        return ["0"] if num_avail == 1 else []

    # Parse the physical IDs visible to this process
    env_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_vis and env_vis.lower() not in {"", "nodevfiles", "-1"}:
        phys_ids = [s.strip() for s in env_vis.split(",") if s.strip()]
    else:
        phys_ids = [str(i) for i in range(num_avail)]

    requested = int(getattr(args, "embedding_num_gpus", 0) or 0)
    if requested > 0:
        phys_ids = phys_ids[:requested]
    return phys_ids


def _resolve_model_spec(model_name: str, args: argparse.Namespace) -> Tuple[str, Optional[str], str]:
    preset = PROTEIN_MODEL_PRESETS.get(model_name)
    if preset is None:
        return model_name, args.model_dir, model_name
    model_dir = args.model_dir or preset["model_dir"]
    return preset["model_name"], model_dir, model_name


def _extract_feature_matrix(
    data: pd.DataFrame,
    seq_col: str,
    model_name: str,
    args: argparse.Namespace,
    *,
    cache_tag: str,
) -> Tuple[np.ndarray, Path]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    backend_model_name, model_dir, display_model_name = _resolve_model_spec(model_name, args)
    row_ids = data["__row_id"].to_numpy(dtype=np.int64)
    sequences = data[seq_col].astype(str).tolist()
    digest = _fingerprint_records(row_ids, sequences, display_model_name, args)
    safe_model = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in display_model_name)
    cache_path = cache_dir / f"{cache_tag}_{seq_col}_{safe_model}_{digest}.pt"

    if cache_path.exists() and not args.rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu")
        features = torch.as_tensor(payload["features"], dtype=torch.float32).numpy()
        cached_ids = torch.as_tensor(payload["row_ids"], dtype=torch.long).numpy()
        if np.array_equal(cached_ids, row_ids):
            return features, cache_path

    if args.embedding_backend == "hash":
        features = _hash_sequence_features(sequences, args.hash_embedding_dim)
        torch.save(
            {
                "row_ids": torch.as_tensor(row_ids, dtype=torch.long),
                "features": torch.as_tensor(features, dtype=torch.float32),
                "model_name": display_model_name,
                "backend_model_name": backend_model_name,
                "model_dir": model_dir,
                "seq_col": seq_col,
                "cache_tag": cache_tag,
                "embedding_backend": args.embedding_backend,
            },
            cache_path,
        )
        return features, cache_path

    overrides = _embedding_overrides(args, backend_model_name)
    gpu_ids = _get_embedding_gpu_ids(args)
    num_gpus = len(gpu_ids)

    if num_gpus > 1 and len(sequences) >= num_gpus:
        # ── Multi-GPU path ────────────────────────────────────────────────
        from scripts.embedding_worker import extract_pooled_chunk

        chunk_size = (len(sequences) + num_gpus - 1) // num_gpus
        chunks = [sequences[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus)]
        chunks = [c for c in chunks if c]

        tmpdir = tempfile.mkdtemp()
        tmp_paths = [os.path.join(tmpdir, f"pooled_chunk_{i}.npy") for i in range(len(chunks))]
        try:
            _run_parallel_embedding_jobs(
                extract_pooled_chunk,
                gpu_ids[: len(chunks)],
                [
                    dict(
                        gpu_id=gpu_ids[i],
                        sequences=chunks[i],
                        backend_model_name=backend_model_name,
                        model_dir=model_dir,
                        overrides=overrides,
                        token_pool=args.token_pool,
                        check_conda_env=args.check_conda_env,
                        strict_conda_env=args.strict_conda_env,
                        out_path=tmp_paths[i],
                    )
                    for i in range(len(chunks))
                ],
                f"Embedding {seq_col}:{model_name} (multi-GPU)",
            )
            features = np.concatenate([np.load(p) for p in tmp_paths], axis=0).astype(np.float32)
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.unlink(p)
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass
    else:
        # ── Single-GPU path ───────────────────────────────────────────────
        model, options = load_embedder(
            backend_model_name,
            model_dir=model_dir,
            check_conda_env=args.check_conda_env,
            strict_conda_env=args.strict_conda_env,
        )
        features_parts: List[np.ndarray] = []
        for start in tqdm(range(0, len(sequences), args.embedding_batch_size), desc=f"Embedding {seq_col}:{model_name}", leave=False):
            batch = sequences[start : start + args.embedding_batch_size]
            output = extract_embeddings(model, batch, options=options, **overrides)
            features_parts.append(_coerce_embedding_batch(output, len(batch), args.token_pool))
        features = np.concatenate(features_parts, axis=0).astype(np.float32)
    torch.save(
        {
            "row_ids": torch.as_tensor(row_ids, dtype=torch.long),
            "features": torch.as_tensor(features, dtype=torch.float32),
            "model_name": display_model_name,
            "backend_model_name": backend_model_name,
            "model_dir": model_dir,
            "seq_col": seq_col,
            "cache_tag": cache_tag,
        },
        cache_path,
    )
    return features, cache_path


def _rand_proj_matrix(in_dim: int, out_dim: int) -> np.ndarray:
    """Return a deterministic Gaussian random-projection matrix of shape (in_dim, out_dim).

    The seed is derived from the two dimensions so the same matrix is always
    reproduced without storing it on disk.  Columns are scaled by
    1/sqrt(out_dim) to approximately preserve L2 norms (Johnson-Lindenstrauss).
    """
    seed = int((in_dim * 2654435761 + out_dim * 2246822519) & 0xFFFF_FFFF)
    rng = np.random.RandomState(seed=seed)
    W = rng.randn(in_dim, out_dim).astype(np.float32)
    W /= np.sqrt(out_dim)
    return W


def _extract_token_feature_matrix(
    data: pd.DataFrame,
    seq_col: str,
    model_name: str,
    args: argparse.Namespace,
    *,
    cache_tag: str,
    shard_size: int = 256,
) -> Tuple[List[np.ndarray], Path]:
    """Extract per-token (variable-length) embeddings.

    To avoid OOM when accumulating thousands of (L_i, D) tensors, embeddings
    are written to sharded ``.pt`` files on the fly.  The index file
    ``<cache_path>`` records the shard paths; a ``ShardedTokenStore`` is
    returned so callers load only the shards they touch at runtime.

    Returns a ``ShardedTokenStore`` (lazy, shard-based) and the cache index path.
    """
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    backend_model_name, model_dir, display_model_name = _resolve_model_spec(model_name, args)
    row_ids = data["__row_id"].to_numpy(dtype=np.int64)
    sequences = data[seq_col].astype(str).tolist()
    digest = _fingerprint_records(row_ids, sequences, display_model_name, args)
    _tok_proj_dim = int(getattr(args, "token_shard_proj_dim", 256))
    tok_digest = hashlib.sha256(
        (digest + f"_tokens_p{_tok_proj_dim}_fp16").encode()
    ).hexdigest()[:16]
    safe_model = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in display_model_name)
    # Index file — its existence signals a completed extraction
    cache_path = cache_dir / f"{cache_tag}_{seq_col}_{safe_model}_tokens_{tok_digest}.pt"
    shard_dir = cache_dir / f"{cache_tag}_{seq_col}_{safe_model}_tokens_{tok_digest}_shards"

    # ── Load from cache ────────────────────────────────────────────────────
    max_cache_shards = int(getattr(args, "token_max_cache_shards", 8) or 8)
    if cache_path.exists() and not args.rebuild_cache:
        index = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_ids = torch.as_tensor(index["row_ids"], dtype=torch.long).numpy()
        if np.array_equal(cached_ids, row_ids):
            return ShardedTokenStore(index["shard_paths"], shard_size, len(row_ids), max_cache_shards), cache_path

    # ── Extract and save shards on the fly ────────────────────────────────
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale shards from a previous partial run
    for old in shard_dir.glob("*.pt"):
        old.unlink()

    shard_paths: List[str] = []
    shard_buf: List[torch.Tensor] = []
    shard_idx = 0
    # Random-projection matrix: lazily built on first sample (in_dim unknown until then)
    _proj_W: Optional[np.ndarray] = None

    def _apply_proj(arr: np.ndarray) -> torch.Tensor:
        """Project arr (L, D) → (L, proj_dim) with a fixed random matrix, then cast to fp16."""
        nonlocal _proj_W
        if _tok_proj_dim > 0 and arr.shape[1] != _tok_proj_dim:
            if _proj_W is None or _proj_W.shape[0] != arr.shape[1]:
                _proj_W = _rand_proj_matrix(arr.shape[1], _tok_proj_dim)
            arr = arr @ _proj_W  # (L, proj_dim), float32
        return torch.as_tensor(arr, dtype=torch.float16)

    def _flush_shard() -> None:
        nonlocal shard_idx
        sp = shard_dir / f"shard_{shard_idx:05d}.pt"
        torch.save({"token_seqs": shard_buf.copy()}, sp)
        shard_paths.append(str(sp))
        shard_buf.clear()
        shard_idx += 1

    if args.embedding_backend == "hash":
        for seq in sequences:
            arr = _hash_token_features(seq, args.hash_embedding_dim)
            shard_buf.append(_apply_proj(arr))
            if len(shard_buf) >= shard_size:
                _flush_shard()
    else:
        overrides = _embedding_overrides(args, backend_model_name)
        overrides["return_tokens"] = True
        gpu_ids = _get_embedding_gpu_ids(args)
        num_gpus = len(gpu_ids)

        if num_gpus > 1 and len(sequences) >= num_gpus:
            # ── Multi-GPU path ────────────────────────────────────────────
            from scripts.embedding_worker import extract_token_chunk

            chunk_size = (len(sequences) + num_gpus - 1) // num_gpus
            chunks = [sequences[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus)]
            chunks = [c for c in chunks if c]

            tmpdir = tempfile.mkdtemp()
            tmp_paths = [os.path.join(tmpdir, f"tok_chunk_{i}.pt") for i in range(len(chunks))]
            try:
                _run_parallel_embedding_jobs(
                    extract_token_chunk,
                    gpu_ids[: len(chunks)],
                    [
                        dict(
                            gpu_id=gpu_ids[i],
                            sequences=chunks[i],
                            backend_model_name=backend_model_name,
                            model_dir=model_dir,
                            overrides=overrides,
                            proj_dim=_tok_proj_dim,
                            check_conda_env=args.check_conda_env,
                            strict_conda_env=args.strict_conda_env,
                            out_path=tmp_paths[i],
                        )
                        for i in range(len(chunks))
                    ],
                    f"Token embed {seq_col}:{model_name} (multi-GPU)",
                )
                # Re-shard in sequence order (each worker file is one contiguous block)
                for tmp_path in tmp_paths:
                    chunk_data = torch.load(tmp_path, map_location="cpu", weights_only=False)
                    for tensor in chunk_data["token_seqs"]:
                        shard_buf.append(tensor)
                        if len(shard_buf) >= shard_size:
                            _flush_shard()
            finally:
                for p in tmp_paths:
                    if os.path.exists(p):
                        os.unlink(p)
                try:
                    os.rmdir(tmpdir)
                except OSError:
                    pass
        else:
            # ── Single-GPU path ───────────────────────────────────────────
            model, options = load_embedder(
                backend_model_name,
                model_dir=model_dir,
                check_conda_env=args.check_conda_env,
                strict_conda_env=args.strict_conda_env,
            )
            for start in tqdm(
                range(0, len(sequences), args.embedding_batch_size),
                desc=f"Token embed {seq_col}:{model_name}",
                leave=False,
            ):
                batch = sequences[start : start + args.embedding_batch_size]
                output = extract_embeddings(model, batch, options=options, **overrides)
                for arr in _coerce_token_embedding_batch(output, len(batch)):
                    shard_buf.append(_apply_proj(arr))
                    if len(shard_buf) >= shard_size:
                        _flush_shard()

    if shard_buf:
        _flush_shard()

    # Write index (tiny file, safe to keep in RAM)
    torch.save(
        {
            "row_ids": torch.as_tensor(row_ids, dtype=torch.long),
            "shard_paths": shard_paths,
            "model_name": display_model_name,
            "backend_model_name": backend_model_name,
            "model_dir": model_dir,
            "seq_col": seq_col,
            "cache_tag": cache_tag,
        },
        cache_path,
    )

    return ShardedTokenStore(shard_paths, shard_size, len(sequences), max_cache_shards), cache_path


def _build_token_feature_matrices(
    data: pd.DataFrame,
    args: argparse.Namespace,
    *,
    cache_tag: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, str]]:
    """Extract token-level embeddings for Env, VH, VL."""
    env_tok_col = getattr(args, "token_env_seq_col", None) or args.env_seq_col
    env_toks, env_cache = _extract_token_feature_matrix(data, env_tok_col, args.env_model, args, cache_tag=cache_tag)
    vh_toks, vh_cache = _extract_token_feature_matrix(data, args.vh_seq_col, args.vh_model, args, cache_tag=cache_tag)
    vl_toks, vl_cache = _extract_token_feature_matrix(data, args.vl_seq_col, args.vl_model, args, cache_tag=cache_tag)
    return env_toks, vh_toks, vl_toks, {
        "env": str(env_cache),
        "vh": str(vh_cache),
        "vl": str(vl_cache),
    }


def _env_aux_feature_names(region_bins: int) -> List[str]:
    names = [
        "env_raw_len",
        "env_aligned_len",
        "env_aligned_gap_count",
        "env_aligned_gap_ratio",
        "env_aligned_non_gap_len",
        "env_glyco_len",
        "env_glyco_marker_count",
        "env_glyco_marker_ratio",
        "env_nxs_motif_count",
        "env_nxs_motif_density",
    ]
    names.extend(f"env_aligned_gap_ratio_bin_{i}" for i in range(region_bins))
    names.extend(f"env_glyco_marker_ratio_bin_{i}" for i in range(region_bins))
    names.extend(f"env_nxs_motif_count_bin_{i}" for i in range(region_bins))
    return names


def _sequence_value(value: Any, *, keep_gaps: bool) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    seq = str(value).strip().replace(" ", "").replace("\n", "").replace("\r", "").upper()
    seq = seq.replace("*", "")
    if not keep_gaps:
        seq = seq.replace("-", "")
    return seq


def _bin_slices(length: int, bins: int) -> List[Tuple[int, int]]:
    if length <= 0:
        return [(0, 0) for _ in range(bins)]
    return [
        (int(round(i * length / bins)), int(round((i + 1) * length / bins)))
        for i in range(bins)
    ]


def _nxs_motif_positions(seq: str) -> List[int]:
    positions: List[int] = []
    for i in range(max(0, len(seq) - 2)):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in {"S", "T"}:
            positions.append(i)
    return positions


def _positions_from_rows(rows: pd.DataFrame) -> List[int]:
    positions: set[int] = set()
    for _, row in rows.iterrows():
        start = pd.to_numeric(row.get("Protein start"), errors="coerce")
        stop = pd.to_numeric(row.get("Protein stop"), errors="coerce")
        if pd.isna(start) or pd.isna(stop):
            continue
        ps = int(start)
        pe = int(stop)
        if pe < ps:
            ps, pe = pe, ps
        positions.update(range(ps, pe + 1))
    return sorted(positions)


def _load_epitope_regions_from_csv(csv_path: str | Path) -> Dict[str, List[int]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Epitope feature CSV not found: {path}")
    feature_df = pd.read_csv(path)
    required_cols = {"Feature", "Type", "Protein start", "Protein stop"}
    missing = sorted(required_cols - set(feature_df.columns))
    if missing:
        raise KeyError(f"Missing required epitope CSV columns in {path}: {missing}")

    feature = feature_df["Feature"].fillna("").astype(str)
    type_col = feature_df["Type"].fillna("").astype(str)
    regions = {
        "CD4bs": _positions_from_rows(
            feature_df[(type_col == "CD4 contact site") | feature.isin(["D loop", "V5 loop"])]
        ),
        "V3": _positions_from_rows(feature_df[feature == "V3 loop"]),
        "V1V2": _positions_from_rows(feature_df[feature.isin(["V1 loop", "V2 loop"])]),
        "MPER": list(range(659, 684)),
        "fusion_peptide": _positions_from_rows(feature_df[feature == "fusion peptide"]),
    }
    return regions


def _read_fasta_records(fasta_path: str | Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    parts: List[str] = []
    with Path(fasta_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header = line[1:]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def _build_hxb2_alignment_column_map(fasta_path: str | Path) -> Dict[int, int]:
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"Alignment FASTA not found: {path}")
    hxb2_aligned = ""
    for header, seq in _read_fasta_records(path):
        if "HXB2" in header.upper():
            hxb2_aligned = _sequence_value(seq, keep_gaps=True)
            break
    if not hxb2_aligned:
        raise ValueError(f"Could not find an HXB2 reference sequence in {path}")

    pos_to_col: Dict[int, int] = {}
    hxb2_pos = 0
    for col, aa in enumerate(hxb2_aligned):
        if aa != "-":
            hxb2_pos += 1
            pos_to_col[hxb2_pos] = col
    return pos_to_col


def _aligned_to_ungapped_map(aligned_seq: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    ungapped_idx = 0
    for col, aa in enumerate(aligned_seq):
        if aa != "-":
            mapping[col] = ungapped_idx
            ungapped_idx += 1
    return mapping


def _resolve_epitope_region(epitope_value: Any, args: argparse.Namespace, regions: Dict[str, List[int]]) -> Tuple[Optional[str], bool]:
    if epitope_value is None or (isinstance(epitope_value, float) and np.isnan(epitope_value)):
        default_region = args.default_epitope_region
        return (default_region if default_region in regions else None), True

    value = str(epitope_value).strip()
    if not value:
        default_region = args.default_epitope_region
        return (default_region if default_region in regions else None), True
    if value in EPITOPE_CLASS_TO_REGION:
        return EPITOPE_CLASS_TO_REGION[value], False
    if value in regions:
        return value, False

    lower = value.lower()
    if "cd4bs" in lower or "cd4 binding" in lower:
        return "CD4bs", False
    if "v1" in lower or "v2" in lower:
        return "V1V2", False
    if "v3" in lower:
        return "V3", False
    if "mper" in lower:
        return "MPER", False
    if "fusion peptide" in lower or "fusion_peptide" in lower:
        return "fusion_peptide", False
    return None, False


def _build_epitope_pos_feature_names(all_positions: Sequence[int]) -> List[str]:
    names: List[str] = []
    for pos in all_positions:
        names.extend(f"epitope_pos_hxb2_{pos}_aa_{aa}" for aa in AA_ALPHABET)
        names.append(f"epitope_pos_hxb2_{pos}_nxs_sequon")
    return names


def _build_single_epitope_pos_feature(
    aligned_seq: str,
    hxb2_col_map: Dict[int, int],
    region_positions: Sequence[int],
    pos_to_slot: Dict[int, int],
    dim: int,
) -> np.ndarray:
    full_feat = np.zeros(dim, dtype=np.float32)
    col_to_ungapped = _aligned_to_ungapped_map(aligned_seq)
    ungapped_seq = aligned_seq.replace("-", "")

    for pos in region_positions:
        slot = pos_to_slot.get(pos)
        if slot is None:
            continue
        col = hxb2_col_map.get(pos)
        aa = aligned_seq[col] if col is not None and col < len(aligned_seq) else "-"
        base = slot * EPITOPE_POS_FEAT_PER_POS
        if aa in AA_ALPHABET:
            full_feat[base + AA_ALPHABET.index(aa)] = 1.0

        if aa == "N" and col is not None and col in col_to_ungapped:
            ungapped_idx = col_to_ungapped[col]
            if ungapped_idx + 2 < len(ungapped_seq):
                x = ungapped_seq[ungapped_idx + 1]
                st = ungapped_seq[ungapped_idx + 2]
                if x not in {"-", "P"} and st in {"S", "T"}:
                    full_feat[base + N_AA] = 1.0
    return full_feat


def _build_epitope_pos_matrix(data: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    if not args.use_epitope_pos_features:
        return np.zeros((len(data), 0), dtype=np.float32), [], {"enabled": False}

    regions = _load_epitope_regions_from_csv(args.epitope_feature_csv)
    all_positions = sorted(set().union(*regions.values()))
    pos_to_slot = {pos: i for i, pos in enumerate(all_positions)}
    dim = len(all_positions) * EPITOPE_POS_FEAT_PER_POS
    hxb2_col_map = _build_hxb2_alignment_column_map(args.alignment_fasta)
    names = _build_epitope_pos_feature_names(all_positions)
    features = np.zeros((len(data), dim), dtype=np.float32)

    aligned_col_present = args.env_aligned_col in data.columns
    epitope_col_present = args.epitope_col in data.columns
    aligned_values = data[args.env_aligned_col].tolist() if aligned_col_present else [""] * len(data)
    epitope_values = data[args.epitope_col].tolist() if epitope_col_present else [None] * len(data)
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    stats = {
        "computed": 0,
        "cached": 0,
        "no_aligned": 0,
        "no_epitope": 0,
        "default_region_used": 0,
        "region_counts": {name: 0 for name in regions},
    }

    for row_idx, (aligned_value, epitope_value) in enumerate(zip(aligned_values, epitope_values)):
        aligned = _sequence_value(aligned_value, keep_gaps=True)
        if not aligned:
            stats["no_aligned"] += 1
            continue
        region_name, used_default = _resolve_epitope_region(epitope_value, args, regions)
        if used_default:
            stats["default_region_used"] += 1
        if region_name is None or region_name not in regions:
            stats["no_epitope"] += 1
            continue
        stats["region_counts"][region_name] += 1
        cache_key = (aligned, region_name)
        if cache_key in cache:
            features[row_idx] = cache[cache_key]
            stats["cached"] += 1
            continue
        row_feat = _build_single_epitope_pos_feature(aligned, hxb2_col_map, regions[region_name], pos_to_slot, dim)
        cache[cache_key] = row_feat
        features[row_idx] = row_feat
        stats["computed"] += 1

    info = {
        "enabled": True,
        "dim": int(dim),
        "feature_csv": str(Path(args.epitope_feature_csv)),
        "alignment_fasta": str(Path(args.alignment_fasta)),
        "env_aligned_col": args.env_aligned_col,
        "env_aligned_col_present": aligned_col_present,
        "epitope_col": args.epitope_col,
        "epitope_col_present": epitope_col_present,
        "default_epitope_region": args.default_epitope_region,
        "feat_per_position": EPITOPE_POS_FEAT_PER_POS,
        "num_positions": int(len(all_positions)),
        "all_surveillance_positions": [int(p) for p in all_positions],
        "region_position_counts": {name: int(len(positions)) for name, positions in regions.items()},
        "regions": {name: [int(p) for p in positions] for name, positions in regions.items()},
        "stats": stats,
    }
    return features, names, info


def _pool_epitope_token_embedding(
    token_emb: np.ndarray,
    aligned_seq: str,
    hxb2_col_map: Dict[int, int],
    region_positions: Sequence[int],
    pool: str,
) -> Optional[np.ndarray]:
    if token_emb.ndim != 2 or token_emb.shape[0] == 0:
        return None
    col_to_ungapped = _aligned_to_ungapped_map(aligned_seq)
    valid_indices: List[int] = []
    for pos in region_positions:
        col = hxb2_col_map.get(pos)
        if col is None or col >= len(aligned_seq) or col not in col_to_ungapped:
            continue
        ungapped_idx = col_to_ungapped[col]
        if ungapped_idx < token_emb.shape[0]:
            valid_indices.append(ungapped_idx)
    if not valid_indices:
        return None
    selected = token_emb[valid_indices, :]
    if pool == "max":
        return selected.max(axis=0).astype(np.float32)
    return selected.mean(axis=0).astype(np.float32)


def _epitope_token_cache_path(
    data: pd.DataFrame,
    args: argparse.Namespace,
    *,
    cache_tag: str,
    display_model_name: str,
) -> Path:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(str(args.embedding_backend).encode("utf-8"))
    digest.update(display_model_name.encode("utf-8"))
    digest.update(str(args.embedding_batch_size).encode("utf-8"))
    digest.update(str(args.hash_embedding_dim).encode("utf-8"))
    digest.update(str(args.embedding_max_length or "").encode("utf-8"))
    digest.update(str(args.strip_stop).encode("utf-8"))
    digest.update(str(args.strip_gaps).encode("utf-8"))
    digest.update(str(args.epitope_feature_csv).encode("utf-8"))
    digest.update(str(args.alignment_fasta).encode("utf-8"))
    digest.update(str(args.env_aligned_col).encode("utf-8"))
    digest.update(str(args.epitope_col).encode("utf-8"))
    digest.update(str(args.default_epitope_region).encode("utf-8"))
    digest.update(str(args.epitope_token_pool).encode("utf-8"))
    row_ids = data["__row_id"].to_numpy(dtype=np.int64)
    env_values = data[args.env_seq_col].astype(str).tolist()
    aligned_values = data[args.env_aligned_col].astype(str).tolist() if args.env_aligned_col in data.columns else [""] * len(data)
    epitope_values = data[args.epitope_col].astype(str).tolist() if args.epitope_col in data.columns else [""] * len(data)
    for row_id, env_seq, aligned, epitope in zip(row_ids, env_values, aligned_values, epitope_values):
        digest.update(str(int(row_id)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(env_seq.encode("utf-8"))
        digest.update(b"\0")
        digest.update(aligned.encode("utf-8"))
        digest.update(b"\0")
        digest.update(epitope.encode("utf-8"))
        digest.update(b"\0")
    safe_model = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in display_model_name)
    return cache_dir / f"{cache_tag}_epitope_token_{safe_model}_{digest.hexdigest()[:20]}.pt"


def _build_epitope_token_matrix(
    data: pd.DataFrame,
    args: argparse.Namespace,
    *,
    cache_tag: str,
) -> Tuple[np.ndarray, List[str], Dict[str, Any], Optional[Path]]:
    if not args.use_epitope_token_features:
        return np.zeros((len(data), 0), dtype=np.float32), [], {"enabled": False}, None

    backend_model_name, model_dir, display_model_name = _resolve_model_spec(args.env_model, args)
    cache_path = _epitope_token_cache_path(data, args, cache_tag=cache_tag, display_model_name=display_model_name)
    row_ids = data["__row_id"].to_numpy(dtype=np.int64)
    if cache_path.exists() and not args.rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu")
        cached_ids = torch.as_tensor(payload["row_ids"], dtype=torch.long).numpy()
        if np.array_equal(cached_ids, row_ids):
            features = torch.as_tensor(payload["features"], dtype=torch.float32).numpy()
            info = dict(payload.get("info", {}))
            info["loaded_from_cache"] = True
            names = list(payload.get("feature_names", [f"epitope_token_emb_{i}" for i in range(features.shape[1])]))
            return features, names, info, cache_path

    regions = _load_epitope_regions_from_csv(args.epitope_feature_csv)
    hxb2_col_map = _build_hxb2_alignment_column_map(args.alignment_fasta)
    aligned_col_present = args.env_aligned_col in data.columns
    epitope_col_present = args.epitope_col in data.columns
    aligned_values = data[args.env_aligned_col].tolist() if aligned_col_present else [""] * len(data)
    epitope_values = data[args.epitope_col].tolist() if epitope_col_present else [None] * len(data)
    sequences = data[args.env_seq_col].astype(str).tolist()

    stats = {
        "computed": 0,
        "cached_region": 0,
        "no_aligned": 0,
        "no_epitope": 0,
        "no_valid_tokens": 0,
        "default_region_used": 0,
        "region_counts": {name: 0 for name in regions},
    }
    region_cache: Dict[Tuple[str, str, str], np.ndarray] = {}
    pooled_rows: List[np.ndarray] = []
    token_dim: Optional[int] = None

    model = None
    options = None
    overrides: Dict[str, Any] = {}
    if args.embedding_backend != "hash":
        model, options = load_embedder(
            backend_model_name,
            model_dir=model_dir,
            check_conda_env=args.check_conda_env,
            strict_conda_env=args.strict_conda_env,
        )
        overrides = _embedding_overrides(args, backend_model_name)
        overrides["return_tokens"] = True

    for start in tqdm(range(0, len(sequences), args.embedding_batch_size), desc="Epitope token embedding", leave=False):
        batch_seq = sequences[start:start + args.embedding_batch_size]
        batch_aligned = aligned_values[start:start + args.embedding_batch_size]
        batch_epitope = epitope_values[start:start + args.embedding_batch_size]
        if args.embedding_backend == "hash":
            token_batch = [_hash_token_features(seq, args.hash_embedding_dim) for seq in batch_seq]
        else:
            assert model is not None and options is not None
            output = extract_embeddings(model, batch_seq, options=options, **overrides)
            token_batch = _coerce_token_embedding_batch(output, len(batch_seq))

        for env_seq, aligned_value, epitope_value, token_emb in zip(batch_seq, batch_aligned, batch_epitope, token_batch):
            if token_dim is None and token_emb.ndim == 2 and token_emb.shape[1] > 0:
                token_dim = int(token_emb.shape[1])
            aligned = _sequence_value(aligned_value, keep_gaps=True)
            if not aligned:
                stats["no_aligned"] += 1
                pooled_rows.append(np.zeros((token_dim or args.hash_embedding_dim,), dtype=np.float32))
                continue
            region_name, used_default = _resolve_epitope_region(epitope_value, args, regions)
            if used_default:
                stats["default_region_used"] += 1
            if region_name is None or region_name not in regions:
                stats["no_epitope"] += 1
                pooled_rows.append(np.zeros((token_dim or args.hash_embedding_dim,), dtype=np.float32))
                continue
            stats["region_counts"][region_name] += 1
            cache_key = (env_seq, aligned, region_name)
            if cache_key in region_cache:
                pooled_rows.append(region_cache[cache_key])
                stats["cached_region"] += 1
                continue
            pooled = _pool_epitope_token_embedding(
                token_emb,
                aligned,
                hxb2_col_map,
                regions[region_name],
                args.epitope_token_pool,
            )
            if pooled is None:
                stats["no_valid_tokens"] += 1
                pooled = np.zeros((token_dim or args.hash_embedding_dim,), dtype=np.float32)
            region_cache[cache_key] = pooled
            pooled_rows.append(pooled)
            stats["computed"] += 1

    if token_dim is None:
        token_dim = int(pooled_rows[0].shape[0]) if pooled_rows else 0
    features = np.stack([
        row if row.shape[0] == token_dim else np.zeros((token_dim,), dtype=np.float32)
        for row in pooled_rows
    ]).astype(np.float32) if pooled_rows else np.zeros((len(data), 0), dtype=np.float32)
    names = [f"epitope_token_emb_{i}" for i in range(features.shape[1])]
    info = {
        "enabled": True,
        "dim": int(features.shape[1]),
        "model": args.env_model,
        "backend_model_name": backend_model_name,
        "model_dir": model_dir,
        "embedding_backend": args.embedding_backend,
        "pool": args.epitope_token_pool,
        "feature_csv": str(Path(args.epitope_feature_csv)),
        "alignment_fasta": str(Path(args.alignment_fasta)),
        "env_aligned_col": args.env_aligned_col,
        "env_aligned_col_present": aligned_col_present,
        "epitope_col": args.epitope_col,
        "epitope_col_present": epitope_col_present,
        "default_epitope_region": args.default_epitope_region,
        "region_position_counts": {name: int(len(positions)) for name, positions in regions.items()},
        "stats": stats,
        "loaded_from_cache": False,
    }
    torch.save(
        {
            "row_ids": torch.as_tensor(row_ids, dtype=torch.long),
            "features": torch.as_tensor(features, dtype=torch.float32),
            "feature_names": names,
            "info": info,
        },
        cache_path,
    )
    return features, names, info, cache_path


def _build_env_aux_matrix(data: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    region_bins = max(1, int(args.env_aux_region_bins))
    names = _env_aux_feature_names(region_bins)
    if not args.use_env_aux_features:
        return np.zeros((len(data), 0), dtype=np.float32), [], {"enabled": False}

    aligned_col = args.env_aligned_col
    glyco_col = args.env_glyco_col
    aligned_values = data[aligned_col].tolist() if aligned_col in data.columns else [""] * len(data)
    glyco_values = data[glyco_col].tolist() if glyco_col in data.columns else [""] * len(data)
    raw_values = data[args.env_seq_col].tolist()
    features = np.zeros((len(data), len(names)), dtype=np.float32)

    for row, (raw_value, aligned_value, glyco_value) in enumerate(zip(raw_values, aligned_values, glyco_values)):
        raw = _sequence_value(raw_value, keep_gaps=False)
        aligned = _sequence_value(aligned_value, keep_gaps=True)
        glyco = _sequence_value(glyco_value, keep_gaps=False)

        raw_len = len(raw)
        aligned_len = len(aligned)
        glyco_len = len(glyco)
        gap_count = aligned.count("-")
        glyco_marker_count = glyco.count("O")
        motif_positions = _nxs_motif_positions(raw)

        offset = 0
        base_values = [
            raw_len,
            aligned_len,
            gap_count,
            gap_count / max(1, aligned_len),
            aligned_len - gap_count,
            glyco_len,
            glyco_marker_count,
            glyco_marker_count / max(1, glyco_len),
            len(motif_positions),
            len(motif_positions) / max(1, raw_len),
        ]
        features[row, offset:offset + len(base_values)] = np.asarray(base_values, dtype=np.float32)
        offset += len(base_values)

        for start, end in _bin_slices(aligned_len, region_bins):
            chunk = aligned[start:end]
            features[row, offset] = chunk.count("-") / max(1, len(chunk))
            offset += 1

        for start, end in _bin_slices(glyco_len, region_bins):
            chunk = glyco[start:end]
            features[row, offset] = chunk.count("O") / max(1, len(chunk))
            offset += 1

        motif_bins = np.zeros(region_bins, dtype=np.float32)
        for pos in motif_positions:
            bin_idx = min(region_bins - 1, int(pos * region_bins / max(1, raw_len)))
            motif_bins[bin_idx] += 1.0
        features[row, offset:offset + region_bins] = motif_bins

    info = {
        "enabled": True,
        "aligned_col": aligned_col,
        "glyco_col": glyco_col,
        "aligned_col_present": aligned_col in data.columns,
        "glyco_col_present": glyco_col in data.columns,
        "region_bins": region_bins,
    }
    return features, names, info


def _fit_env_aux_stats(aux: np.ndarray, train_pos: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    train_aux = aux[train_pos] if aux.size else aux
    mean = train_aux.mean(axis=0) if train_aux.size else np.zeros((aux.shape[1],), dtype=np.float32)
    std = train_aux.std(axis=0) if train_aux.size else np.ones((aux.shape[1],), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {
        "feature_names": list(names),
        "mean": mean.astype(np.float32),
        "std": std,
    }


def _apply_env_aux_stats(aux: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    if aux.shape[1] == 0:
        return aux.astype(np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((aux - mean) / std).astype(np.float32)


def _build_feature_matrices(
    data: pd.DataFrame,
    args: argparse.Namespace,
    *,
    cache_tag: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    env, env_cache = _extract_feature_matrix(data, args.env_seq_col, args.env_model, args, cache_tag=cache_tag)
    vh, vh_cache = _extract_feature_matrix(data, args.vh_seq_col, args.vh_model, args, cache_tag=cache_tag)
    vl, vl_cache = _extract_feature_matrix(data, args.vl_seq_col, args.vl_model, args, cache_tag=cache_tag)
    return env, vh, vl, {
        "env": str(env_cache),
        "vh": str(vh_cache),
        "vl": str(vl_cache),
    }


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    y_mean: float,
    y_std: float,
    *,
    classification_prob_threshold: float = 0.5,
    is_token: bool = False,
    desc: str = "",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ids: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for _batch in tqdm(loader, desc=desc, leave=False, disable=not desc):
            if is_token:
                if len(_batch) == 12:
                    batch_id, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, y, env_pool, vh_pool, vl_pool = _batch
                    pred_cls, pred_reg, *_ = model(
                        env_pool.to(device), vh_pool.to(device), vl_pool.to(device),
                        env_tok.to(device), vh_tok.to(device), vl_tok.to(device),
                        env_mask.to(device), vh_mask.to(device), vl_mask.to(device),
                        ab_id.to(device),
                    )
                else:
                    batch_id, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, y = _batch
                    pred_cls, pred_reg, *_ = model(
                        env_tok.to(device), vh_tok.to(device), vl_tok.to(device),
                        env_mask.to(device), vh_mask.to(device), vl_mask.to(device),
                        ab_id.to(device),
                    )
            else:
                batch_id, batch_ab_id, env, vh, vl, y = _batch
                pred_cls, pred_reg, *_ = model(
                    env.to(device), vh.to(device), vl.to(device), batch_ab_id.to(device),
                )
            out = _select_task_prediction(task, pred_cls, pred_reg)
            ids.append(batch_id.numpy())
            ys.append(y.numpy())
            preds.append(out.detach().cpu().numpy())
    id_np = np.concatenate(ids)
    y_np = np.concatenate(ys)
    pred_np = np.concatenate(preds)
    if task == "classification":
        metrics = _classification_metrics(
            y_np.astype(np.int64),
            pred_np,
            prob_threshold=classification_prob_threshold,
        )
    else:
        y_real = y_np * y_std + y_mean
        pred_real = pred_np * y_std + y_mean
        metrics = _regression_metrics(y_real, pred_real)
    return metrics, id_np, y_np, pred_np


def _log_epoch(task: str, epoch: int, epochs: int, train_loss: float, val_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> None:
    if task == "classification":
        columns = ["acc", "f1_bin", "f1_macro", "auc", "ap"]

        def fmt(metrics: Dict[str, float]) -> List[str]:
            return [
                _fmt_float(metrics["accuracy"]),
                _fmt_float(metrics["f1_binary"]),
                _fmt_float(metrics["f1_macro"]),
                _fmt_float(metrics.get("roc_auc", float("nan"))),
                _fmt_float(metrics["avg_precision"]),
            ]
    else:
        columns = ["mae", "rmse", "pearson_r", "spearman_r"]

        def fmt(metrics: Dict[str, float]) -> List[str]:
            return [
                _fmt_float(metrics["mae"], 5),
                _fmt_float(metrics["rmse"], 5),
                _fmt_float(metrics.get("pearson_r", float("nan"))),
                _fmt_float(metrics["spearman_r"]),
            ]

    _print_epoch_table(
        epoch,
        epochs,
        train_loss,
        columns,
        [("val", fmt(val_metrics)), ("test", fmt(test_metrics))],
    )


def _make_loader(
    ids: np.ndarray,
    ab_ids: np.ndarray,
    env: np.ndarray,
    vh: np.ndarray,
    vl: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = FeatureDataset(ids, ab_ids, env, vh, vl, y)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )


def _make_multitask_loader(
    ids: np.ndarray,
    ab_ids: np.ndarray,
    env: np.ndarray,
    vh: np.ndarray,
    vl: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    reg_mask: np.ndarray,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = MultiTaskFeatureDataset(ids, ab_ids, env, vh, vl, y_cls, y_reg, reg_mask)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )


def _multitask_supervised_loss(
    pred_cls: torch.Tensor,
    y_cls: torch.Tensor,
    pred_reg: torch.Tensor,
    y_reg: torch.Tensor,
    reg_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.use_focal_loss:
        cls_loss = focal_bce_with_logits(
            pred_cls,
            y_cls.float(),
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
    else:
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, y_cls.float())
    reg_mask_bool = reg_mask > 0.5
    if torch.any(reg_mask_bool):
        reg_loss = F.huber_loss(pred_reg[reg_mask_bool].float(), y_reg[reg_mask_bool].float(), delta=args.huber_delta)
    else:
        reg_loss = pred_reg.new_zeros(())
    total = args.lambda_cls * cls_loss + args.lambda_reg * reg_loss
    return total, cls_loss, reg_loss


def _evaluate_multitask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
    args: argparse.Namespace,
    *,
    classification_prob_threshold: float = 0.5,
    is_token: bool = False,
    desc: str = "",
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ids: List[np.ndarray] = []
    y_cls_list: List[np.ndarray] = []
    y_reg_list: List[np.ndarray] = []
    reg_mask_list: List[np.ndarray] = []
    pred_cls_list: List[np.ndarray] = []
    pred_reg_list: List[np.ndarray] = []
    with torch.no_grad():
        for _batch in tqdm(loader, desc=desc, leave=False, disable=not desc):
            if is_token:
                if len(_batch) == 14:
                    batch_id, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, y_cls, y_reg, reg_mask, env_pool, vh_pool, vl_pool = _batch
                    pred_cls, pred_reg, *_ = model(
                        env_pool.to(device), vh_pool.to(device), vl_pool.to(device),
                        env_tok.to(device), vh_tok.to(device), vl_tok.to(device),
                        env_mask.to(device), vh_mask.to(device), vl_mask.to(device),
                        ab_id.to(device),
                    )
                else:
                    batch_id, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, y_cls, y_reg, reg_mask = _batch
                    pred_cls, pred_reg, *_ = model(
                        env_tok.to(device), vh_tok.to(device), vl_tok.to(device),
                        env_mask.to(device), vh_mask.to(device), vl_mask.to(device),
                        ab_id.to(device),
                    )
            else:
                batch_id, batch_ab_id, env, vh, vl, y_cls, y_reg, reg_mask = _batch
                pred_cls, pred_reg, *_ = model(
                    env.to(device), vh.to(device), vl.to(device), batch_ab_id.to(device),
                )
            ids.append(batch_id.numpy())
            y_cls_list.append(y_cls.numpy())
            y_reg_list.append(y_reg.numpy())
            reg_mask_list.append(reg_mask.numpy())
            pred_cls_list.append(pred_cls.detach().cpu().numpy())
            pred_reg_list.append(pred_reg.detach().cpu().numpy())

    id_np = np.concatenate(ids)
    y_cls_np = np.concatenate(y_cls_list).astype(np.float32)
    y_reg_np = np.concatenate(y_reg_list).astype(np.float32)
    reg_mask_np = np.concatenate(reg_mask_list).astype(np.float32)
    pred_cls_np = np.concatenate(pred_cls_list).astype(np.float32)
    pred_reg_np = np.concatenate(pred_reg_list).astype(np.float32)

    cls_metrics = _classification_metrics(
        y_cls_np.astype(np.int64),
        pred_cls_np,
        prob_threshold=classification_prob_threshold,
    )

    reg_mask_bool = reg_mask_np > 0.5
    if np.any(reg_mask_bool):
        y_reg_real = y_reg_np[reg_mask_bool] * y_std + y_mean
        pred_reg_real = pred_reg_np[reg_mask_bool] * y_std + y_mean
        reg_metrics = _regression_metrics(y_reg_real, pred_reg_real)
    else:
        reg_metrics = _empty_regression_metrics()

    total_loss, cls_loss, reg_loss = _multitask_supervised_loss(
        torch.as_tensor(pred_cls_np, dtype=torch.float32),
        torch.as_tensor(y_cls_np, dtype=torch.float32),
        torch.as_tensor(pred_reg_np, dtype=torch.float32),
        torch.as_tensor(y_reg_np, dtype=torch.float32),
        torch.as_tensor(reg_mask_np, dtype=torch.float32),
        args,
    )
    metrics: Dict[str, Any] = {
        "classification": cls_metrics,
        "regression": reg_metrics,
        "supervised_loss": float(total_loss.item()),
        "classification_loss": float(cls_loss.item()),
        "regression_loss": float(reg_loss.item()),
        "regression_num_samples": int(reg_mask_bool.sum()),
    }
    return metrics, id_np, y_cls_np, pred_cls_np, y_reg_np, pred_reg_np, reg_mask_np


def _log_epoch_multitask(
    epoch: int,
    epochs: int,
    train_loss: float,
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
) -> None:
    columns = ["cls_f1", "cls_auc", "reg_rmse", "reg_pearson", "sup_loss"]

    def fmt(metrics: Dict[str, Any]) -> List[str]:
        cls = metrics.get("classification", {})
        reg = metrics.get("regression", {})
        return [
            _fmt_float(cls.get("f1_binary", float("nan"))),
            _fmt_float(cls.get("roc_auc", float("nan"))),
            _fmt_float(reg.get("rmse", float("nan")), 5),
            _fmt_float(reg.get("pearson_r", float("nan"))),
            _fmt_float(metrics.get("supervised_loss", float("nan")), 5),
        ]

    _print_epoch_table(
        epoch,
        epochs,
        train_loss,
        columns,
        [("val", fmt(val_metrics)), ("test", fmt(test_metrics))],
    )


def _prediction_frame_multitask(
    ids: np.ndarray,
    y_cls: np.ndarray,
    pred_cls: np.ndarray,
    y_reg: np.ndarray,
    pred_reg: np.ndarray,
    reg_mask: np.ndarray,
    y_mean: float,
    y_std: float,
    *,
    source: Optional[pd.DataFrame] = None,
    split_name: str = "",
    classification_prob_threshold: float = 0.5,
) -> pd.DataFrame:
    reg_mask_bool = reg_mask > 0.5
    y_reg_real = np.full(y_reg.shape, np.nan, dtype=np.float32)
    pred_reg_real = np.full(pred_reg.shape, np.nan, dtype=np.float32)
    if np.any(reg_mask_bool):
        y_reg_real[reg_mask_bool] = y_reg[reg_mask_bool] * y_std + y_mean
        pred_reg_real[reg_mask_bool] = pred_reg[reg_mask_bool] * y_std + y_mean

    prob = _logits_to_prob(pred_cls)
    frame = pd.DataFrame(
        {
            "id": ids,
            "y_true_cls": y_cls.astype(np.int64),
            "y_pred_cls_logit": pred_cls,
            "y_prob": prob,
            "y_pred_cls_label": (prob >= classification_prob_threshold).astype(np.int64),
            "reg_available": reg_mask_bool.astype(np.int64),
            "y_true_reg_norm": np.where(reg_mask_bool, y_reg, np.nan),
            "y_pred_reg_norm": np.where(reg_mask_bool, pred_reg, np.nan),
            "y_true_reg_real": y_reg_real,
            "y_pred_reg_real": pred_reg_real,
        }
    )
    if split_name:
        frame.insert(0, "split", split_name)
    if source is None:
        return frame
    return _attach_prediction_metadata(frame, source)


def _flatten_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_metrics(full_key, value))
        else:
            out[full_key] = float(value)
    return out


def _attach_prediction_metadata(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Join prediction rows to source metadata by sample id.

    Evaluation loaders (especially token bucket samplers) may emit batches in an
    order different from ``source`` row order; never align with positional concat.
    """
    if "id" not in frame.columns:
        raise KeyError("Prediction frame must include an 'id' column to attach metadata.")

    meta = source.reset_index(drop=True).copy()
    join_col: Optional[str] = None
    if "__display_id" in meta.columns:
        join_col = "__display_id"
    elif "__row_id" in meta.columns:
        join_col = "__row_id"

    public_cols = [col for col in meta.columns if not col.startswith("__")]

    if join_col is not None:
        meta_keys = meta[[join_col] + public_cols].rename(columns={join_col: "id"})
        meta_keys["id"] = meta_keys["id"].astype(np.int64)
        if meta_keys["id"].duplicated().any():
            dup = meta_keys["id"][meta_keys["id"].duplicated()].unique().tolist()[:5]
            raise ValueError(f"Duplicate prediction join ids in source metadata: {dup}")

        eval_ids = frame["id"].astype(np.int64)
        if len(eval_ids) != len(np.unique(eval_ids)):
            raise ValueError("Duplicate ids in prediction frame.")

        merged = frame.merge(meta_keys, on="id", how="left", validate="one_to_one")
        join_verify_col = "idx" if "idx" in public_cols else public_cols[0]
        missing_mask = merged[join_verify_col].isna()
        if missing_mask.any():
            missing_ids = merged.loc[missing_mask, "id"].astype(int).tolist()[:5]
            raise ValueError(f"Missing metadata for prediction ids: {missing_ids}")

        leading = ["split"] if "split" in frame.columns else []
        pred_cols = [col for col in frame.columns if col not in leading]
        ordered = leading + public_cols + pred_cols
        return merged[ordered].sort_values("id", kind="stable").reset_index(drop=True)

    metadata = meta[public_cols] if public_cols else pd.DataFrame()
    if len(metadata) != len(frame):
        raise ValueError(
            f"Prediction metadata length mismatch: {len(metadata)} rows for {len(frame)} predictions"
        )
    return pd.concat([metadata, frame], axis=1)


def _prediction_frame(
    ids: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    task: str,
    y_mean: float,
    y_std: float,
    *,
    source: Optional[pd.DataFrame] = None,
    split_name: str = "",
    classification_prob_threshold: float = 0.5,
) -> pd.DataFrame:
    frame = pd.DataFrame({"id": ids, "y_true": y, "y_pred_logit_or_norm": pred})
    if task == "classification":
        prob = _logits_to_prob(pred)
        frame["y_prob"] = prob
        frame["y_pred_label"] = (prob >= classification_prob_threshold).astype(np.int64)
    else:
        frame["y_true_real"] = y * y_std + y_mean
        frame["y_pred_real"] = pred * y_std + y_mean

    if split_name:
        frame.insert(0, "split", split_name)
    if source is None:
        return frame
    return _attach_prediction_metadata(frame, source)


def _independent_targets(data: pd.DataFrame, args: argparse.Namespace, split: SplitData) -> np.ndarray:
    y_raw = data["__target"].to_numpy()
    if args.task == "classification":
        class_map = split.class_label_map or {}
        keep_values = set(class_map.keys())
        if not set(np.unique(y_raw)).issubset(keep_values):
            bad = sorted(set(np.unique(y_raw)) - keep_values)
            raise ValueError(f"Independent labels contain classes unseen during training: {bad}")
        return np.asarray([class_map[v] for v in y_raw], dtype=np.float32)
    return ((y_raw.astype(np.float32) - split.y_mean) / split.y_std).astype(np.float32)


def _independent_targets_multitask(
    data: pd.DataFrame,
    args: argparse.Namespace,
    split: SplitData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "__target_cls" not in data.columns or "__target_reg" not in data.columns:
        raise KeyError("Multitask independent evaluation expects __target_cls and __target_reg columns.")
    y_cls_raw = data["__target_cls"].to_numpy()
    class_map = split.class_label_map or {}
    keep_values = set(class_map.keys())
    if not set(np.unique(y_cls_raw)).issubset(keep_values):
        bad = sorted(set(np.unique(y_cls_raw)) - keep_values)
        raise ValueError(f"Independent labels contain classes unseen during training: {bad}")
    y_cls = np.asarray([class_map[v] for v in y_cls_raw], dtype=np.float32)

    y_reg_raw = pd.to_numeric(data["__target_reg"], errors="coerce").to_numpy(dtype=np.float64)
    reg_mask = np.isfinite(y_reg_raw)
    y_reg = np.zeros(len(data), dtype=np.float32)
    if np.any(reg_mask):
        y_reg[reg_mask] = ((y_reg_raw[reg_mask] - split.y_mean) / split.y_std).astype(np.float32)
    return y_cls, y_reg, reg_mask.astype(np.float32)


def _build_model(env_dim: int, vh_dim: int, vl_dim: int, args: argparse.Namespace, device: torch.device) -> PAVE:
    return PAVE(
        in_env=env_dim,
        in_vh=vh_dim,
        in_vl=vl_dim,
        num_ab_embeddings=args.num_ab_id_embeddings,
        proj_dim=args.proj_dim,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
        interaction_rank=args.interaction_rank,
        num_res_blocks=args.num_res_blocks,
        head_hidden_mult=args.head_hidden_mult,
    ).to(device)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return _unwrap_model(model).state_dict()


def _load_model_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    _unwrap_model(model).load_state_dict(state_dict)


def _maybe_wrap_data_parallel(model: nn.Module, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if device.type != "cuda" or args.cpu or not getattr(args, "data_parallel", True):
        return model
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return model
    CONSOLE.print(f"[cyan]Using DataParallel on {gpu_count} CUDA devices.[/]")
    return nn.DataParallel(model)


def _make_bucket_sampler(env_toks, vh_toks, vl_toks, batch_size: int, shuffle: bool) -> BucketBatchSampler:
    """Build a BucketBatchSampler from combined env+vh+vl token lengths."""
    env_lens = _get_store_lengths(env_toks)
    vh_lens = _get_store_lengths(vh_toks)
    vl_lens = _get_store_lengths(vl_toks)
    total_lens = [e + h + l for e, h, l in zip(env_lens, vh_lens, vl_lens)]
    return BucketBatchSampler(total_lens, batch_size, shuffle=shuffle)


def _make_token_loader(
    ids: np.ndarray,
    ab_ids: np.ndarray,
    env_toks: List[np.ndarray],
    vh_toks: List[np.ndarray],
    vl_toks: List[np.ndarray],
    y: np.ndarray,
    args: argparse.Namespace,
    env_pool: Optional[np.ndarray] = None,
    vh_pool: Optional[np.ndarray] = None,
    vl_pool: Optional[np.ndarray] = None,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = TokenFeatureDataset(
        ids, ab_ids, env_toks, vh_toks, vl_toks, y,
        env_pool=env_pool, vh_pool=vh_pool, vl_pool=vl_pool,
    )
    batch_size = _effective_token_batch_size(args)
    num_workers = _effective_token_num_workers(args)
    pin = torch.cuda.is_available() and not args.cpu
    prefetch = int(getattr(args, "token_prefetch_factor", 4)) if num_workers > 0 else None
    if getattr(args, "token_bucket_sort", True):
        sampler = _make_bucket_sampler(env_toks, vh_toks, vl_toks, batch_size, shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin,
            collate_fn=_token_pad_collate_fn,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=_token_pad_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch,
    )


def _make_multitask_token_loader(
    ids: np.ndarray,
    ab_ids: np.ndarray,
    env_toks: List[np.ndarray],
    vh_toks: List[np.ndarray],
    vl_toks: List[np.ndarray],
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    reg_mask: np.ndarray,
    args: argparse.Namespace,
    env_pool: Optional[np.ndarray] = None,
    vh_pool: Optional[np.ndarray] = None,
    vl_pool: Optional[np.ndarray] = None,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = MultiTaskTokenFeatureDataset(
        ids, ab_ids, env_toks, vh_toks, vl_toks, y_cls, y_reg, reg_mask,
        env_pool=env_pool, vh_pool=vh_pool, vl_pool=vl_pool,
    )
    batch_size = _effective_token_batch_size(args)
    num_workers = _effective_token_num_workers(args)
    pin = torch.cuda.is_available() and not args.cpu
    prefetch = int(getattr(args, "token_prefetch_factor", 4)) if num_workers > 0 else None
    if getattr(args, "token_bucket_sort", True):
        sampler = _make_bucket_sampler(env_toks, vh_toks, vl_toks, batch_size, shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin,
            collate_fn=_multitask_token_pad_collate_fn,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=_multitask_token_pad_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch,
    )


def _build_cross_attn_model(
    env_dim: int,
    vh_dim: int,
    vl_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> PAVECrossAttn:
    return PAVECrossAttn(
        in_env=env_dim,
        in_vh=vh_dim,
        in_vl=vl_dim,
        num_ab_embeddings=args.num_ab_id_embeddings,
        proj_dim=args.proj_dim,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_layers=args.cross_attn_layers,
        interaction_rank=args.interaction_rank,
        num_res_blocks=args.num_res_blocks,
        head_hidden_mult=args.head_hidden_mult,
    ).to(device)


def _build_hybrid_cross_attn_model(
    env_dim: int,
    vh_dim: int,
    vl_dim: int,
    tok_env_dim: int,
    tok_vh_dim: int,
    tok_vl_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> PAVEHybridCrossAttn:
    return PAVEHybridCrossAttn(
        in_env=env_dim,
        in_vh=vh_dim,
        in_vl=vl_dim,
        tok_env_dim=tok_env_dim,
        tok_vh_dim=tok_vh_dim,
        tok_vl_dim=tok_vl_dim,
        num_ab_embeddings=args.num_ab_id_embeddings,
        proj_dim=args.proj_dim,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_layers=args.cross_attn_layers,
        interaction_rank=args.interaction_rank,
        num_res_blocks=args.num_res_blocks,
        head_hidden_mult=args.head_hidden_mult,
        token_gate_hidden=getattr(args, "hybrid_token_gate_hidden", 0),
    ).to(device)


def _pave_forward(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    is_token: bool,
) -> Tuple[torch.Tensor, ...]:
    """Unpack batch, move to device, run model forward pass.

    Returns (pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair, y).
    """
    if is_token:
        if len(batch) == 12:
            _, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, y, env_pool, vh_pool, vl_pool = batch
            out = model(
                env_pool.to(device, non_blocking=True),
                vh_pool.to(device, non_blocking=True),
                vl_pool.to(device, non_blocking=True),
                env_tok.to(device, non_blocking=True),
                vh_tok.to(device, non_blocking=True),
                vl_tok.to(device, non_blocking=True),
                env_mask.to(device, non_blocking=True),
                vh_mask.to(device, non_blocking=True),
                vl_mask.to(device, non_blocking=True),
                ab_id.to(device, non_blocking=True),
            )
        else:
            _, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, y = batch
            out = model(
                env_tok.to(device, non_blocking=True),
                vh_tok.to(device, non_blocking=True),
                vl_tok.to(device, non_blocking=True),
                env_mask.to(device, non_blocking=True),
                vh_mask.to(device, non_blocking=True),
                vl_mask.to(device, non_blocking=True),
                ab_id.to(device, non_blocking=True),
            )
    else:
        _, ab_id, env, vh, vl, y = batch
        out = model(
            env.to(device, non_blocking=True),
            vh.to(device, non_blocking=True),
            vl.to(device, non_blocking=True),
            ab_id.to(device, non_blocking=True),
        )
    pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair = out[:7]
    return pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair, y.to(device, non_blocking=True)


def _pave_forward_multitask(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    is_token: bool,
) -> Tuple[torch.Tensor, ...]:
    """Unpack batch and run forward for multitask mode.

    Returns (pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair, y_cls, y_reg, reg_mask).
    """
    if is_token:
        if len(batch) == 14:
            _, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, y_cls, y_reg, reg_mask, env_pool, vh_pool, vl_pool = batch
            out = model(
                env_pool.to(device, non_blocking=True),
                vh_pool.to(device, non_blocking=True),
                vl_pool.to(device, non_blocking=True),
                env_tok.to(device, non_blocking=True),
                vh_tok.to(device, non_blocking=True),
                vl_tok.to(device, non_blocking=True),
                env_mask.to(device, non_blocking=True),
                vh_mask.to(device, non_blocking=True),
                vl_mask.to(device, non_blocking=True),
                ab_id.to(device, non_blocking=True),
            )
        else:
            _, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, y_cls, y_reg, reg_mask = batch
            out = model(
                env_tok.to(device, non_blocking=True),
                vh_tok.to(device, non_blocking=True),
                vl_tok.to(device, non_blocking=True),
                env_mask.to(device, non_blocking=True),
                vh_mask.to(device, non_blocking=True),
                vl_mask.to(device, non_blocking=True),
                ab_id.to(device, non_blocking=True),
            )
    else:
        _, ab_id, env, vh, vl, y_cls, y_reg, reg_mask = batch
        out = model(
            env.to(device, non_blocking=True),
            vh.to(device, non_blocking=True),
            vl.to(device, non_blocking=True),
            ab_id.to(device, non_blocking=True),
        )
    pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair = out[:7]
    return (
        pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair,
        y_cls.to(device, non_blocking=True),
        y_reg.to(device, non_blocking=True),
        reg_mask.to(device, non_blocking=True),
    )


def _standardize_env_for_split(
    env_base: np.ndarray,
    env_aux: Optional[np.ndarray],
    env_aux_names: Sequence[str],
    split: SplitData,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    if env_aux is None or env_aux.shape[1] == 0:
        return env_base.astype(np.float32), None
    stats = _fit_env_aux_stats(env_aux, split.train_pos, env_aux_names)
    env_all = np.concatenate([env_base, _apply_env_aux_stats(env_aux, stats)], axis=1).astype(np.float32)
    return env_all, stats


def _train_single_fold(
    fold_index: int,
    split: SplitData,
    data: pd.DataFrame,
    env_all: np.ndarray,
    vh_all: np.ndarray,
    vl_all: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
    *,
    env_tok_all: Optional[List[np.ndarray]] = None,
    vh_tok_all: Optional[List[np.ndarray]] = None,
    vl_tok_all: Optional[List[np.ndarray]] = None,
) -> Tuple[Dict[str, Any], nn.Module, Dict[str, Any]]:
    model_arch = _selected_model_arch(args)
    is_token = env_tok_all is not None
    is_hybrid = model_arch == "hybrid_cross_attn"
    set_seed(args.seed + fold_index)
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_pos_fit, train_balance_info = _resolve_train_positions_for_training(split, data, args, fold_index)
    y_train_fit = _align_train_labels(split.train_pos, split.y_train, train_pos_fit)

    train_ids = data.iloc[train_pos_fit]["__display_id"].to_numpy(dtype=np.int64)
    val_ids = data.iloc[split.val_pos]["__display_id"].to_numpy(dtype=np.int64)
    test_ids = data.iloc[split.test_pos]["__display_id"].to_numpy(dtype=np.int64)
    train_ab_ids = data.iloc[train_pos_fit]["__ab_id_index"].to_numpy(dtype=np.int64)
    val_ab_ids = data.iloc[split.val_pos]["__ab_id_index"].to_numpy(dtype=np.int64)
    test_ab_ids = data.iloc[split.test_pos]["__ab_id_index"].to_numpy(dtype=np.int64)

    if is_token:
        assert env_tok_all is not None and vh_tok_all is not None and vl_tok_all is not None
        dl_train = _make_token_loader(
            train_ids, train_ab_ids,
            env_tok_all.subset(train_pos_fit),
            vh_tok_all.subset(train_pos_fit),
            vl_tok_all.subset(train_pos_fit),
            y_train_fit, args,
            env_pool=env_all[train_pos_fit] if is_hybrid else None,
            vh_pool=vh_all[train_pos_fit] if is_hybrid else None,
            vl_pool=vl_all[train_pos_fit] if is_hybrid else None,
            shuffle=True,
        )
        dl_val = _make_token_loader(
            val_ids, val_ab_ids,
            env_tok_all.subset(split.val_pos),
            vh_tok_all.subset(split.val_pos),
            vl_tok_all.subset(split.val_pos),
            split.y_val, args,
            env_pool=env_all[split.val_pos] if is_hybrid else None,
            vh_pool=vh_all[split.val_pos] if is_hybrid else None,
            vl_pool=vl_all[split.val_pos] if is_hybrid else None,
            shuffle=False,
        )
        dl_test = _make_token_loader(
            test_ids, test_ab_ids,
            env_tok_all.subset(split.test_pos),
            vh_tok_all.subset(split.test_pos),
            vl_tok_all.subset(split.test_pos),
            split.y_test, args,
            env_pool=env_all[split.test_pos] if is_hybrid else None,
            vh_pool=vh_all[split.test_pos] if is_hybrid else None,
            vl_pool=vl_all[split.test_pos] if is_hybrid else None,
            shuffle=False,
        )
        tok_env_dim = int(env_tok_all[0].shape[1])
        tok_vh_dim = int(vh_tok_all[0].shape[1])
        tok_vl_dim = int(vl_tok_all[0].shape[1])
        if is_hybrid:
            model = _build_hybrid_cross_attn_model(
                env_all.shape[1], vh_all.shape[1], vl_all.shape[1],
                tok_env_dim, tok_vh_dim, tok_vl_dim, args, device,
            )
        else:
            model = _build_cross_attn_model(tok_env_dim, tok_vh_dim, tok_vl_dim, args, device)
    else:
        dl_train = _make_loader(
            train_ids, train_ab_ids,
            env_all[train_pos_fit], vh_all[train_pos_fit], vl_all[train_pos_fit],
            y_train_fit, args, shuffle=True,
        )
        dl_val = _make_loader(
            val_ids, val_ab_ids,
            env_all[split.val_pos], vh_all[split.val_pos], vl_all[split.val_pos],
            split.y_val, args, shuffle=False,
        )
        dl_test = _make_loader(
            test_ids, test_ab_ids,
            env_all[split.test_pos], vh_all[split.test_pos], vl_all[split.test_pos],
            split.y_test, args, shuffle=False,
        )
        model = _build_model(env_all.shape[1], vh_all.shape[1], vl_all.shape[1], args, device)
    model = _maybe_wrap_data_parallel(model, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("-inf") if args.task == "classification" else float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0
    history: List[Dict[str, float]] = []

    _print_fold_header(
        fold_index,
        len(train_pos_fit),
        len(split.val_pos),
        len(split.test_pos),
        train_groups=_count_unique_groups(data, split.train_pos, args.group_col),
        val_groups=_count_unique_groups(data, split.val_pos, args.group_col),
        test_groups=_count_unique_groups(data, split.test_pos, args.group_col),
        train_raw_size=len(split.train_pos),
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        _loss_sum = 0.0
        _loss_count = 0
        pbar = tqdm(dl_train, desc=f"Fold {fold_index:02d} Epoch {epoch:03d}/{args.epochs:03d}", leave=False)
        for _batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair, y = _pave_forward(model, _batch, device, is_token)
            pred = _select_task_prediction(args.task, pred_cls, pred_reg)
            if args.task == "classification":
                if args.use_focal_loss:
                    pred_loss = focal_bce_with_logits(
                        pred,
                        y.float(),
                        alpha=args.focal_alpha,
                        gamma=args.focal_gamma,
                    )
                else:
                    pred_loss = F.binary_cross_entropy_with_logits(pred, y.float())
                va_loss = _virus_antibody_loss_classification(
                    z_env,
                    z_ab,
                    y,
                    args.temperature,
                    args.use_hard_negative,
                    args.hard_negative_topk,
                )
                sup_loss = _supcon_loss_classification(pair, y, args.temperature)
            else:
                pred_loss = F.huber_loss(pred.float(), y.float(), delta=args.huber_delta)
                va_loss = _virus_antibody_loss_regression(z_env, z_ab, y, args.temperature)
                sup_loss = _supcon_loss_regression(pair, y, args.supcon_reg_sigma, args.temperature)

            heavy_light_loss = _info_nce_diag(z_vh, z_vl, args.temperature)
            loss = (
                pred_loss
                + args.lambda_hl * heavy_light_loss
                + args.lambda_va * va_loss
                + args.lambda_supcon * sup_loss
            )
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            _loss_sum += float(loss.detach().cpu().item())
            _loss_count += 1
            pbar.set_postfix(loss=f"{_loss_sum / _loss_count:.4f}")

        train_loss = _loss_sum / _loss_count if _loss_count > 0 else 0.0
        val_metrics, _, _, _ = _evaluate(model, dl_val, device, args.task, split.y_mean, split.y_std, is_token=is_token, desc="Val  ")
        test_metrics, _, _, _ = _evaluate(model, dl_test, device, args.task, split.y_mean, split.y_std, is_token=is_token, desc="Test ")
        if not args.silent_epoch_metrics:
            _log_epoch(args.task, epoch, args.epochs, train_loss, val_metrics, test_metrics)

        row: Dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        row.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
        history.append(row)

        val_key, larger_is_better = _single_task_early_stop_value(args.task, val_metrics, args)
        improved = (val_key > best_val + 1e-12) if larger_is_better else (val_key < best_val - 1e-12)
        if improved:
            best_val = float(val_key)
            best_state = {k: v.detach().cpu().clone() for k, v in _model_state_dict(model).items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is not None:
        _load_model_state_dict(model, best_state)

    val_metrics, val_ids_out, val_y, val_pred = _evaluate(model, dl_val, device, args.task, split.y_mean, split.y_std, is_token=is_token, desc="Val  ")
    test_metrics, test_ids_out, test_y, test_pred = _evaluate(model, dl_test, device, args.task, split.y_mean, split.y_std, is_token=is_token, desc="Test ")

    classification_prob_threshold = 0.5
    val_threshold_objective_score = float("nan")
    threshold_objective = "fixed_0.5"
    val_metrics_at_0_5: Optional[Dict[str, float]] = None
    if args.task == "classification":
        val_y_int = val_y.astype(np.int64)
        val_prob = _logits_to_prob(val_pred)
        val_metrics_at_0_5 = dict(val_metrics)
        if args.val_prob_threshold_tuning and len(np.unique(val_y_int)) > 1:
            classification_prob_threshold, val_threshold_objective_score = find_best_threshold(
                val_y_int,
                val_prob,
                metric=args.classification_threshold_metric,
            )
            threshold_objective = args.classification_threshold_metric
        val_metrics = _classification_metrics(val_y_int, val_pred, prob_threshold=classification_prob_threshold)
        test_metrics = _classification_metrics(
            test_y.astype(np.int64),
            test_pred,
            prob_threshold=classification_prob_threshold,
        )

    checkpoint_path = fold_dir / "best_model.pt"
    val_predictions_path = fold_dir / "val_predictions.csv"
    test_predictions_path = fold_dir / "test_predictions.csv"
    torch.save(_model_state_dict(model), checkpoint_path)
    _prediction_frame(
        val_ids_out,
        val_y,
        val_pred,
        args.task,
        split.y_mean,
        split.y_std,
        source=data.iloc[split.val_pos],
        split_name=f"fold_{fold_index}_val",
        classification_prob_threshold=classification_prob_threshold,
    ).to_csv(val_predictions_path, index=False)
    _prediction_frame(
        test_ids_out,
        test_y,
        test_pred,
        args.task,
        split.y_mean,
        split.y_std,
        source=data.iloc[split.test_pos],
        split_name=f"fold_{fold_index}_test",
        classification_prob_threshold=classification_prob_threshold,
    ).to_csv(test_predictions_path, index=False)
    with (fold_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    summary = {
        "fold": fold_index,
        "checkpoint": str(checkpoint_path),
        "history": str(fold_dir / "history.json"),
        "prediction_files": {
            "val": str(val_predictions_path),
            "test": str(test_predictions_path),
        },
        "split_sizes": {
            "train_raw": int(len(split.train_pos)),
            "train_balanced": int(len(train_pos_fit)),
            "train": int(len(train_pos_fit)),
            "val": int(len(split.val_pos)),
            "test": int(len(split.test_pos)),
        },
        "train_balance_by_antibody": train_balance_info,
        "y_mean": float(split.y_mean),
        "y_std": float(split.y_std),
        "best_val_score": float(best_val),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    test_prediction: Dict[str, Any] = {
        "fold": fold_index,
        "ids": test_ids_out,
        "y": test_y,
        "pred": test_pred,
        "y_mean": split.y_mean,
        "y_std": split.y_std,
    }
    if args.task == "classification":
        summary["classification_prob_threshold"] = classification_prob_threshold
        summary["classification_threshold_tuning"] = {
            "enabled": bool(args.val_prob_threshold_tuning),
            "objective_metric": threshold_objective,
            "val_objective_score_at_best_threshold": val_threshold_objective_score,
            "val_metrics_at_prob_0_5": val_metrics_at_0_5,
        }
        test_prediction["classification_prob_threshold"] = classification_prob_threshold
    return summary, model, test_prediction


def _train_single_fold_multitask(
    fold_index: int,
    split: SplitData,
    data: pd.DataFrame,
    env_all: np.ndarray,
    vh_all: np.ndarray,
    vl_all: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
    *,
    env_tok_all: Optional[List[np.ndarray]] = None,
    vh_tok_all: Optional[List[np.ndarray]] = None,
    vl_tok_all: Optional[List[np.ndarray]] = None,
) -> Tuple[Dict[str, Any], nn.Module, Dict[str, Any]]:
    model_arch = _selected_model_arch(args)
    is_token = env_tok_all is not None
    is_hybrid = model_arch == "hybrid_cross_attn"
    if split.y_cls_train is None or split.y_reg_train is None or split.reg_mask_train is None:
        raise ValueError("Multitask split is missing classification/regression targets.")

    set_seed(args.seed + fold_index)
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_pos_fit, train_balance_info = _resolve_train_positions_for_training(split, data, args, fold_index)
    y_cls_train_fit = _align_train_labels(split.train_pos, split.y_cls_train, train_pos_fit)
    y_reg_train_fit = _align_train_labels(split.train_pos, split.y_reg_train, train_pos_fit)
    reg_mask_train_fit = _align_train_labels(split.train_pos, split.reg_mask_train, train_pos_fit)

    train_ids = data.iloc[train_pos_fit]["__display_id"].to_numpy(dtype=np.int64)
    val_ids = data.iloc[split.val_pos]["__display_id"].to_numpy(dtype=np.int64)
    test_ids = data.iloc[split.test_pos]["__display_id"].to_numpy(dtype=np.int64)
    train_ab_ids = data.iloc[train_pos_fit]["__ab_id_index"].to_numpy(dtype=np.int64)
    val_ab_ids = data.iloc[split.val_pos]["__ab_id_index"].to_numpy(dtype=np.int64)
    test_ab_ids = data.iloc[split.test_pos]["__ab_id_index"].to_numpy(dtype=np.int64)

    y_cls_val = split.y_cls_val if split.y_cls_val is not None else split.y_val
    y_reg_val = split.y_reg_val if split.y_reg_val is not None else np.zeros_like(split.y_val)
    reg_mask_val = split.reg_mask_val if split.reg_mask_val is not None else np.zeros_like(split.y_val)
    y_cls_test = split.y_cls_test if split.y_cls_test is not None else split.y_test
    y_reg_test = split.y_reg_test if split.y_reg_test is not None else np.zeros_like(split.y_test)
    reg_mask_test = split.reg_mask_test if split.reg_mask_test is not None else np.zeros_like(split.y_test)

    if is_token:
        assert env_tok_all is not None and vh_tok_all is not None and vl_tok_all is not None
        dl_train = _make_multitask_token_loader(
            train_ids, train_ab_ids,
            env_tok_all.subset(train_pos_fit),
            vh_tok_all.subset(train_pos_fit),
            vl_tok_all.subset(train_pos_fit),
            y_cls_train_fit, y_reg_train_fit, reg_mask_train_fit,
            args,
            env_pool=env_all[train_pos_fit] if is_hybrid else None,
            vh_pool=vh_all[train_pos_fit] if is_hybrid else None,
            vl_pool=vl_all[train_pos_fit] if is_hybrid else None,
            shuffle=True,
        )
        dl_val = _make_multitask_token_loader(
            val_ids, val_ab_ids,
            env_tok_all.subset(split.val_pos),
            vh_tok_all.subset(split.val_pos),
            vl_tok_all.subset(split.val_pos),
            y_cls_val, y_reg_val, reg_mask_val,
            args,
            env_pool=env_all[split.val_pos] if is_hybrid else None,
            vh_pool=vh_all[split.val_pos] if is_hybrid else None,
            vl_pool=vl_all[split.val_pos] if is_hybrid else None,
            shuffle=False,
        )
        dl_test = _make_multitask_token_loader(
            test_ids, test_ab_ids,
            env_tok_all.subset(split.test_pos),
            vh_tok_all.subset(split.test_pos),
            vl_tok_all.subset(split.test_pos),
            y_cls_test, y_reg_test, reg_mask_test,
            args,
            env_pool=env_all[split.test_pos] if is_hybrid else None,
            vh_pool=vh_all[split.test_pos] if is_hybrid else None,
            vl_pool=vl_all[split.test_pos] if is_hybrid else None,
            shuffle=False,
        )
        tok_env_dim = int(env_tok_all[0].shape[1])
        tok_vh_dim = int(vh_tok_all[0].shape[1])
        tok_vl_dim = int(vl_tok_all[0].shape[1])
        if is_hybrid:
            model = _build_hybrid_cross_attn_model(
                env_all.shape[1], vh_all.shape[1], vl_all.shape[1],
                tok_env_dim, tok_vh_dim, tok_vl_dim, args, device,
            )
        else:
            model = _build_cross_attn_model(tok_env_dim, tok_vh_dim, tok_vl_dim, args, device)
    else:
        dl_train = _make_multitask_loader(
            train_ids, train_ab_ids,
            env_all[train_pos_fit], vh_all[train_pos_fit], vl_all[train_pos_fit],
            y_cls_train_fit, y_reg_train_fit, reg_mask_train_fit,
            args, shuffle=True,
        )
        dl_val = _make_multitask_loader(
            val_ids, val_ab_ids,
            env_all[split.val_pos], vh_all[split.val_pos], vl_all[split.val_pos],
            y_cls_val, y_reg_val, reg_mask_val,
            args, shuffle=False,
        )
        dl_test = _make_multitask_loader(
            test_ids, test_ab_ids,
            env_all[split.test_pos], vh_all[split.test_pos], vl_all[split.test_pos],
            y_cls_test, y_reg_test, reg_mask_test,
            args, shuffle=False,
        )
        model = _build_model(env_all.shape[1], vh_all.shape[1], vl_all.shape[1], args, device)
    model = _maybe_wrap_data_parallel(model, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0
    history: List[Dict[str, float]] = []

    _print_fold_header(
        fold_index,
        len(train_pos_fit),
        len(split.val_pos),
        len(split.test_pos),
        train_groups=_count_unique_groups(data, split.train_pos, args.group_col),
        val_groups=_count_unique_groups(data, split.val_pos, args.group_col),
        test_groups=_count_unique_groups(data, split.test_pos, args.group_col),
        train_raw_size=len(split.train_pos),
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        _loss_sum = 0.0
        _loss_count = 0
        pbar = tqdm(dl_train, desc=f"Fold {fold_index:02d} Epoch {epoch:03d}/{args.epochs:03d}", leave=False)
        for _batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            pred_cls, pred_reg, z_env, z_vh, z_vl, z_ab, pair, y_cls, y_reg, reg_mask = _pave_forward_multitask(
                model, _batch, device, is_token
            )
            pred_loss, _, _ = _multitask_supervised_loss(pred_cls, y_cls, pred_reg, y_reg, reg_mask, args)

            va_cls = _virus_antibody_loss_classification(
                z_env,
                z_ab,
                y_cls,
                args.temperature,
                args.use_hard_negative,
                args.hard_negative_topk,
            )
            sup_cls = _supcon_loss_classification(pair, y_cls, args.temperature)

            reg_mask_bool = reg_mask > 0.5
            if torch.any(reg_mask_bool):
                va_reg = _virus_antibody_loss_regression(
                    z_env[reg_mask_bool],
                    z_ab[reg_mask_bool],
                    y_reg[reg_mask_bool],
                    args.temperature,
                )
                sup_reg = _supcon_loss_regression(
                    pair[reg_mask_bool],
                    y_reg[reg_mask_bool],
                    args.supcon_reg_sigma,
                    args.temperature,
                )
            else:
                va_reg = z_env.new_zeros(())
                sup_reg = z_env.new_zeros(())

            va_loss = args.lambda_cls * va_cls + args.lambda_reg * va_reg
            sup_loss = args.lambda_cls * sup_cls + args.lambda_reg * sup_reg
            heavy_light_loss = _info_nce_diag(z_vh, z_vl, args.temperature)
            loss = pred_loss + args.lambda_hl * heavy_light_loss + args.lambda_va * va_loss + args.lambda_supcon * sup_loss
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            _loss_sum += float(loss.detach().cpu().item())
            _loss_count += 1
            pbar.set_postfix(loss=f"{_loss_sum / _loss_count:.4f}")

        train_loss = _loss_sum / _loss_count if _loss_count > 0 else 0.0
        val_metrics, *_ = _evaluate_multitask(model, dl_val, device, split.y_mean, split.y_std, args, is_token=is_token, desc="Val  ")
        test_metrics, *_ = _evaluate_multitask(model, dl_test, device, split.y_mean, split.y_std, args, is_token=is_token, desc="Test ")
        if not args.silent_epoch_metrics:
            _log_epoch_multitask(epoch, args.epochs, train_loss, val_metrics, test_metrics)

        row: Dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}
        row.update(_flatten_metrics("val", val_metrics))
        row.update(_flatten_metrics("test", test_metrics))
        history.append(row)

        val_key = float(val_metrics.get("supervised_loss", float("inf")))
        improved = val_key < best_val - 1e-12
        if improved:
            best_val = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in _model_state_dict(model).items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is not None:
        _load_model_state_dict(model, best_state)

    (
        val_metrics,
        val_ids_out,
        val_y_cls,
        val_pred_cls,
        val_y_reg,
        val_pred_reg,
        val_reg_mask,
    ) = _evaluate_multitask(model, dl_val, device, split.y_mean, split.y_std, args, is_token=is_token, desc="Val  ")
    (
        test_metrics,
        test_ids_out,
        test_y_cls,
        test_pred_cls,
        test_y_reg,
        test_pred_reg,
        test_reg_mask,
    ) = _evaluate_multitask(model, dl_test, device, split.y_mean, split.y_std, args, is_token=is_token, desc="Test ")

    classification_prob_threshold = 0.5
    val_threshold_objective_score = float("nan")
    threshold_objective = "fixed_0.5"
    val_cls_metrics_at_0_5 = dict(val_metrics["classification"])
    if args.val_prob_threshold_tuning and len(np.unique(val_y_cls.astype(np.int64))) > 1:
        classification_prob_threshold, val_threshold_objective_score = find_best_threshold(
            val_y_cls.astype(np.int64),
            _logits_to_prob(val_pred_cls),
            metric=args.classification_threshold_metric,
        )
        threshold_objective = args.classification_threshold_metric

    val_metrics["classification"] = _classification_metrics(
        val_y_cls.astype(np.int64),
        val_pred_cls,
        prob_threshold=classification_prob_threshold,
    )
    test_metrics["classification"] = _classification_metrics(
        test_y_cls.astype(np.int64),
        test_pred_cls,
        prob_threshold=classification_prob_threshold,
    )

    checkpoint_path = fold_dir / "best_model.pt"
    val_predictions_path = fold_dir / "val_predictions.csv"
    test_predictions_path = fold_dir / "test_predictions.csv"
    torch.save(_model_state_dict(model), checkpoint_path)
    _prediction_frame_multitask(
        val_ids_out,
        val_y_cls,
        val_pred_cls,
        val_y_reg,
        val_pred_reg,
        val_reg_mask,
        split.y_mean,
        split.y_std,
        source=data.iloc[split.val_pos],
        split_name=f"fold_{fold_index}_val",
        classification_prob_threshold=classification_prob_threshold,
    ).to_csv(val_predictions_path, index=False)
    _prediction_frame_multitask(
        test_ids_out,
        test_y_cls,
        test_pred_cls,
        test_y_reg,
        test_pred_reg,
        test_reg_mask,
        split.y_mean,
        split.y_std,
        source=data.iloc[split.test_pos],
        split_name=f"fold_{fold_index}_test",
        classification_prob_threshold=classification_prob_threshold,
    ).to_csv(test_predictions_path, index=False)
    with (fold_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    summary = {
        "fold": fold_index,
        "checkpoint": str(checkpoint_path),
        "history": str(fold_dir / "history.json"),
        "prediction_files": {
            "val": str(val_predictions_path),
            "test": str(test_predictions_path),
        },
        "split_sizes": {
            "train_raw": int(len(split.train_pos)),
            "train_balanced": int(len(train_pos_fit)),
            "train": int(len(train_pos_fit)),
            "val": int(len(split.val_pos)),
            "test": int(len(split.test_pos)),
        },
        "train_balance_by_antibody": train_balance_info,
        "y_mean": float(split.y_mean),
        "y_std": float(split.y_std),
        "best_val_score": float(best_val),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "classification_prob_threshold": classification_prob_threshold,
        "classification_threshold_tuning": {
            "enabled": bool(args.val_prob_threshold_tuning),
            "objective_metric": threshold_objective,
            "val_objective_score_at_best_threshold": val_threshold_objective_score,
            "val_metrics_at_prob_0_5": val_cls_metrics_at_0_5,
        },
    }
    test_prediction: Dict[str, Any] = {
        "fold": fold_index,
        "ids": test_ids_out,
        "y_cls": test_y_cls,
        "pred_cls": test_pred_cls,
        "y_reg": test_y_reg,
        "pred_reg": test_pred_reg,
        "reg_mask": test_reg_mask,
        "y_mean": split.y_mean,
        "y_std": split.y_std,
        "classification_prob_threshold": classification_prob_threshold,
    }
    return summary, model, test_prediction


def _evaluate_independent_fold(
    fold_index: int,
    model: nn.Module,
    split: SplitData,
    ind_data: pd.DataFrame,
    env_ind_base: np.ndarray,
    vh_ind: np.ndarray,
    vl_ind: np.ndarray,
    env_aux_ind: Optional[np.ndarray],
    env_aux_stats: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    *,
    eval_output_dir: Optional[Path] = None,
    classification_prob_threshold: float = 0.5,
    env_ind_toks: Optional[List[np.ndarray]] = None,
    vh_ind_toks: Optional[List[np.ndarray]] = None,
    vl_ind_toks: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    is_token = env_ind_toks is not None
    is_hybrid = _selected_model_arch(args) == "hybrid_cross_attn"
    out_dir = eval_output_dir if eval_output_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    y_ind = _independent_targets(ind_data, args, split)
    ind_ids = ind_data["__display_id"].to_numpy(dtype=np.int64)
    ind_ab_ids = ind_data["__ab_id_index"].to_numpy(dtype=np.int64)
    if is_token:
        assert env_ind_toks is not None and vh_ind_toks is not None and vl_ind_toks is not None
        env_ind = env_ind_base.astype(np.float32)
        if env_aux_ind is not None and env_aux_ind.shape[1] > 0 and env_aux_stats is not None:
            env_ind = np.concatenate([env_ind, _apply_env_aux_stats(env_aux_ind, env_aux_stats)], axis=1).astype(np.float32)
        dl_ind = _make_token_loader(
            ind_ids, ind_ab_ids, env_ind_toks, vh_ind_toks, vl_ind_toks, y_ind, args,
            env_pool=env_ind if is_hybrid else None,
            vh_pool=vh_ind if is_hybrid else None,
            vl_pool=vl_ind if is_hybrid else None,
            shuffle=False,
        )
    else:
        env_ind = env_ind_base.astype(np.float32)
        if env_aux_ind is not None and env_aux_ind.shape[1] > 0 and env_aux_stats is not None:
            env_ind = np.concatenate([env_ind, _apply_env_aux_stats(env_aux_ind, env_aux_stats)], axis=1).astype(np.float32)
        dl_ind = _make_loader(ind_ids, ind_ab_ids, env_ind, vh_ind, vl_ind, y_ind, args, shuffle=False)
    ind_metrics, ind_ids_out, ind_y, ind_pred = _evaluate(
        model,
        dl_ind,
        device,
        args.task,
        split.y_mean,
        split.y_std,
        classification_prob_threshold=classification_prob_threshold,
        is_token=is_token,
    )
    independent_predictions_path = out_dir / f"independent_test_predictions_fold{fold_index}.csv"
    _prediction_frame(
        ind_ids_out,
        ind_y,
        ind_pred,
        args.task,
        split.y_mean,
        split.y_std,
        source=ind_data,
        split_name=f"fold_{fold_index}_independent_test",
        classification_prob_threshold=classification_prob_threshold,
    ).to_csv(independent_predictions_path, index=False)
    out: Dict[str, Any] = {
        "fold": fold_index,
        "metrics": ind_metrics,
        "prediction_file": str(independent_predictions_path),
        "ids": ind_ids_out,
        "y": ind_y,
        "pred": ind_pred,
        "y_mean": split.y_mean,
        "y_std": split.y_std,
    }
    if args.task == "classification":
        out["classification_prob_threshold"] = classification_prob_threshold
    return out


def _evaluate_independent_fold_multitask(
    fold_index: int,
    model: nn.Module,
    split: SplitData,
    ind_data: pd.DataFrame,
    env_ind_base: np.ndarray,
    vh_ind: np.ndarray,
    vl_ind: np.ndarray,
    env_aux_ind: Optional[np.ndarray],
    env_aux_stats: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    *,
    eval_output_dir: Optional[Path] = None,
    classification_prob_threshold: float = 0.5,
    env_ind_toks: Optional[List[np.ndarray]] = None,
    vh_ind_toks: Optional[List[np.ndarray]] = None,
    vl_ind_toks: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    is_token = env_ind_toks is not None
    is_hybrid = _selected_model_arch(args) == "hybrid_cross_attn"
    out_dir = eval_output_dir if eval_output_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    y_cls_ind, y_reg_ind, reg_mask_ind = _independent_targets_multitask(ind_data, args, split)
    ind_ids = ind_data["__display_id"].to_numpy(dtype=np.int64)
    ind_ab_ids = ind_data["__ab_id_index"].to_numpy(dtype=np.int64)
    if is_token:
        assert env_ind_toks is not None and vh_ind_toks is not None and vl_ind_toks is not None
        env_ind = env_ind_base.astype(np.float32)
        if env_aux_ind is not None and env_aux_ind.shape[1] > 0 and env_aux_stats is not None:
            env_ind = np.concatenate([env_ind, _apply_env_aux_stats(env_aux_ind, env_aux_stats)], axis=1).astype(np.float32)
        dl_ind = _make_multitask_token_loader(
            ind_ids, ind_ab_ids, env_ind_toks, vh_ind_toks, vl_ind_toks,
            y_cls_ind, y_reg_ind, reg_mask_ind, args,
            env_pool=env_ind if is_hybrid else None,
            vh_pool=vh_ind if is_hybrid else None,
            vl_pool=vl_ind if is_hybrid else None,
            shuffle=False,
        )
    else:
        env_ind = env_ind_base.astype(np.float32)
        if env_aux_ind is not None and env_aux_ind.shape[1] > 0 and env_aux_stats is not None:
            env_ind = np.concatenate([env_ind, _apply_env_aux_stats(env_aux_ind, env_aux_stats)], axis=1).astype(np.float32)
        dl_ind = _make_multitask_loader(
            ind_ids, ind_ab_ids, env_ind, vh_ind, vl_ind,
            y_cls_ind, y_reg_ind, reg_mask_ind, args, shuffle=False,
        )
    ind_metrics, ind_ids_out, ind_y_cls, ind_pred_cls, ind_y_reg, ind_pred_reg, ind_reg_mask = _evaluate_multitask(
        model,
        dl_ind,
        device,
        split.y_mean,
        split.y_std,
        args,
        classification_prob_threshold=classification_prob_threshold,
        is_token=is_token,
    )
    independent_predictions_path = out_dir / f"independent_test_predictions_fold{fold_index}.csv"
    _prediction_frame_multitask(
        ind_ids_out,
        ind_y_cls,
        ind_pred_cls,
        ind_y_reg,
        ind_pred_reg,
        ind_reg_mask,
        split.y_mean,
        split.y_std,
        source=ind_data,
        split_name=f"fold_{fold_index}_independent_test",
        classification_prob_threshold=classification_prob_threshold,
    ).to_csv(independent_predictions_path, index=False)
    return {
        "fold": fold_index,
        "metrics": ind_metrics,
        "prediction_file": str(independent_predictions_path),
        "ids": ind_ids_out,
        "y_cls": ind_y_cls,
        "pred_cls": ind_pred_cls,
        "y_reg": ind_y_reg,
        "pred_reg": ind_pred_reg,
        "reg_mask": ind_reg_mask,
        "y_mean": split.y_mean,
        "y_std": split.y_std,
        "classification_prob_threshold": classification_prob_threshold,
    }


def _write_independent_ensemble(
    fold_predictions: Sequence[Dict[str, Any]],
    ind_data: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
    *,
    eval_output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if not fold_predictions:
        return {}, {}
    out_dir = eval_output_dir if eval_output_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(fold_predictions[0]["ids"])

    if args.task == "classification":
        y_true = np.asarray(fold_predictions[0]["y"], dtype=np.int64)
        logits = np.stack([np.asarray(item["pred"], dtype=np.float64) for item in fold_predictions], axis=0)
        probs = _logits_to_prob(logits)
        thresholds = [float(item.get("classification_prob_threshold", 0.5)) for item in fold_predictions]
        labels = np.stack(
            [(probs[i] >= thresholds[i]).astype(np.int64) for i in range(len(fold_predictions))],
            axis=0,
        )
        avg_prob = probs.mean(axis=0)
        avg_label = (avg_prob >= 0.5).astype(np.int64)
        vote_fraction = labels.mean(axis=0)
        vote_label = (vote_fraction >= 0.5).astype(np.int64)
        average_metrics = _classification_metrics_from_prob(y_true, avg_prob, avg_label)
        vote_metrics = _classification_metrics_from_prob(y_true, vote_fraction, vote_label)

        frame = pd.DataFrame(
            {
                "split": "independent_test_ensemble",
                "id": ids,
                "y_true": y_true,
                "average_y_prob": avg_prob,
                "average_y_pred_label": avg_label,
                "vote_fraction_positive": vote_fraction,
                "vote_y_pred_label": vote_label,
            }
        )
        for fold_offset, item in enumerate(fold_predictions, start=1):
            frame[f"fold{fold_offset}_logit"] = np.asarray(item["pred"], dtype=np.float64)
            frame[f"fold{fold_offset}_prob"] = probs[fold_offset - 1]
            frame[f"fold{fold_offset}_label"] = labels[fold_offset - 1]
        ensemble_path = out_dir / "independent_test_predictions_ensemble.csv"
        _attach_prediction_metadata(frame, ind_data).to_csv(ensemble_path, index=False)
        return (
            {
                "average": average_metrics,
                "vote": vote_metrics,
            },
            {"independent_test_ensemble": str(ensemble_path)},
        )

    target_by_id = ind_data.set_index(ind_data["__display_id"].astype(np.int64))["__target"]
    y_true_real = target_by_id.reindex(ids).to_numpy(dtype=np.float32)
    fold_real_preds = np.stack(
        [
            np.asarray(item["pred"], dtype=np.float64) * float(item["y_std"]) + float(item["y_mean"])
            for item in fold_predictions
        ],
        axis=0,
    )
    avg_real = fold_real_preds.mean(axis=0)
    ensemble_metrics = _regression_metrics(y_true_real, avg_real)
    frame = pd.DataFrame(
        {
            "split": "independent_test_ensemble",
            "id": ids,
            "y_true_real": y_true_real,
            "y_pred_real": avg_real,
        }
    )
    for fold_offset in range(fold_real_preds.shape[0]):
        frame[f"fold{fold_offset + 1}_y_pred_real"] = fold_real_preds[fold_offset]
    ensemble_path = out_dir / "independent_test_predictions_ensemble.csv"
    _attach_prediction_metadata(frame, ind_data).to_csv(ensemble_path, index=False)
    return ({"average": ensemble_metrics}, {"independent_test_ensemble": str(ensemble_path)})


def _write_independent_ensemble_multitask(
    fold_predictions: Sequence[Dict[str, Any]],
    ind_data: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
    *,
    eval_output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if not fold_predictions:
        return {}, {}
    out_dir = eval_output_dir if eval_output_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = np.asarray(fold_predictions[0]["ids"])

    y_cls_true = np.asarray(fold_predictions[0]["y_cls"], dtype=np.int64)
    cls_logits = np.stack([np.asarray(item["pred_cls"], dtype=np.float64) for item in fold_predictions], axis=0)
    cls_probs = _logits_to_prob(cls_logits)
    thresholds = [float(item.get("classification_prob_threshold", 0.5)) for item in fold_predictions]
    cls_labels = np.stack(
        [(cls_probs[i] >= thresholds[i]).astype(np.int64) for i in range(len(fold_predictions))],
        axis=0,
    )
    avg_cls_prob = cls_probs.mean(axis=0)
    avg_cls_label = (avg_cls_prob >= 0.5).astype(np.int64)
    vote_fraction = cls_labels.mean(axis=0)
    vote_cls_label = (vote_fraction >= 0.5).astype(np.int64)

    reg_mask = np.asarray(fold_predictions[0]["reg_mask"], dtype=np.float32) > 0.5
    y_reg_true_real = np.full(reg_mask.shape, np.nan, dtype=np.float64)
    avg_reg_real = np.full(reg_mask.shape, np.nan, dtype=np.float64)
    reg_metrics = _empty_regression_metrics()
    if np.any(reg_mask):
        y_reg_true_real[reg_mask] = (
            np.asarray(fold_predictions[0]["y_reg"], dtype=np.float64)[reg_mask] * float(fold_predictions[0]["y_std"])
            + float(fold_predictions[0]["y_mean"])
        )
        reg_real_preds = np.stack(
            [
                np.asarray(item["pred_reg"], dtype=np.float64) * float(item["y_std"]) + float(item["y_mean"])
                for item in fold_predictions
            ],
            axis=0,
        )
        avg_reg_real[reg_mask] = reg_real_preds.mean(axis=0)[reg_mask]
        reg_metrics = _regression_metrics(y_reg_true_real[reg_mask], avg_reg_real[reg_mask])

    frame = pd.DataFrame(
        {
            "split": "independent_test_ensemble",
            "id": ids,
            "y_true_cls": y_cls_true,
            "average_y_prob": avg_cls_prob,
            "average_y_pred_label": avg_cls_label,
            "vote_fraction_positive": vote_fraction,
            "vote_y_pred_label": vote_cls_label,
            "reg_available": reg_mask.astype(np.int64),
            "y_true_reg_real": y_reg_true_real,
            "average_y_pred_reg_real": avg_reg_real,
        }
    )
    for fold_offset, item in enumerate(fold_predictions, start=1):
        frame[f"fold{fold_offset}_cls_logit"] = np.asarray(item["pred_cls"], dtype=np.float64)
        frame[f"fold{fold_offset}_cls_prob"] = cls_probs[fold_offset - 1]
        frame[f"fold{fold_offset}_cls_label"] = cls_labels[fold_offset - 1]
        frame[f"fold{fold_offset}_reg_pred_real"] = (
            np.asarray(item["pred_reg"], dtype=np.float64) * float(item["y_std"]) + float(item["y_mean"])
        )
    ensemble_path = out_dir / "independent_test_predictions_ensemble.csv"
    _attach_prediction_metadata(frame, ind_data).to_csv(ensemble_path, index=False)
    return (
        {
            "classification": {
                "average": _classification_metrics_from_prob(y_cls_true, avg_cls_prob, avg_cls_label),
                "vote": _classification_metrics_from_prob(y_cls_true, vote_fraction, vote_cls_label),
            },
            "regression": {"average": reg_metrics},
        },
        {"independent_test_ensemble": str(ensemble_path)},
    )


def _summarize_metric_distribution(fold_metrics: Sequence[Dict[str, Any]], metric_key: str) -> Dict[str, Dict[str, Any]]:
    metric_names = sorted({name for item in fold_metrics for name in item.get(metric_key, {})})
    summary: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray(
            [float(item.get(metric_key, {}).get(name, float("nan"))) for item in fold_metrics],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[name] = {"mean": float("nan"), "std": float("nan"), "num_folds": 0}
        else:
            summary[name] = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "num_folds": int(finite.size),
            }
    return summary


def _summarize_cv_test_results(
    fold_predictions: Sequence[Dict[str, Any]],
    fold_metrics: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not fold_predictions:
        return {}

    if args.task == "classification":
        y_blocks: List[np.ndarray] = []
        prob_blocks: List[np.ndarray] = []
        pred_blocks: List[np.ndarray] = []
        for item in fold_predictions:
            y_blocks.append(np.asarray(item["y"], dtype=np.int64))
            logits = np.asarray(item["pred"], dtype=np.float64)
            prob = _logits_to_prob(logits)
            t = float(item.get("classification_prob_threshold", 0.5))
            pred_blocks.append((prob >= t).astype(np.int64))
            prob_blocks.append(prob)
        y_true = np.concatenate(y_blocks, axis=0)
        prob = np.concatenate(prob_blocks, axis=0)
        pred_label = np.concatenate(pred_blocks, axis=0)
        pooled_metrics = _classification_metrics_from_prob(y_true, prob, pred_label=pred_label)
    else:
        y_true_parts = []
        pred_parts = []
        for item in fold_predictions:
            y_true_parts.append(np.asarray(item["y"], dtype=np.float64) * float(item["y_std"]) + float(item["y_mean"]))
            pred_parts.append(np.asarray(item["pred"], dtype=np.float64) * float(item["y_std"]) + float(item["y_mean"]))
        y_true = np.concatenate(y_true_parts, axis=0)
        pred = np.concatenate(pred_parts, axis=0)
        pooled_metrics = _regression_metrics(y_true, pred)

    return {
        "pooled_out_of_fold_metrics": pooled_metrics,
        "per_fold_test_metrics": _summarize_metric_distribution(fold_metrics, "test_metrics"),
        "num_samples": int(sum(len(item["y"]) for item in fold_predictions)),
        "num_folds": int(len(fold_predictions)),
    }


def _summarize_cv_test_results_multitask(
    fold_predictions: Sequence[Dict[str, Any]],
    fold_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not fold_predictions:
        return {}

    y_cls_blocks: List[np.ndarray] = []
    prob_blocks: List[np.ndarray] = []
    pred_blocks: List[np.ndarray] = []
    y_reg_true_parts: List[np.ndarray] = []
    y_reg_pred_parts: List[np.ndarray] = []
    reg_sample_count = 0
    for item in fold_predictions:
        y_cls_blocks.append(np.asarray(item["y_cls"], dtype=np.int64))
        logits = np.asarray(item["pred_cls"], dtype=np.float64)
        prob = _logits_to_prob(logits)
        t = float(item.get("classification_prob_threshold", 0.5))
        pred_blocks.append((prob >= t).astype(np.int64))
        prob_blocks.append(prob)

        reg_mask = np.asarray(item["reg_mask"], dtype=np.float32) > 0.5
        if np.any(reg_mask):
            y_reg_true_parts.append(
                np.asarray(item["y_reg"], dtype=np.float64)[reg_mask] * float(item["y_std"]) + float(item["y_mean"])
            )
            y_reg_pred_parts.append(
                np.asarray(item["pred_reg"], dtype=np.float64)[reg_mask] * float(item["y_std"]) + float(item["y_mean"])
            )
            reg_sample_count += int(reg_mask.sum())

    y_cls_true = np.concatenate(y_cls_blocks, axis=0)
    cls_prob = np.concatenate(prob_blocks, axis=0)
    cls_pred = np.concatenate(pred_blocks, axis=0)
    pooled_cls_metrics = _classification_metrics_from_prob(y_cls_true, cls_prob, pred_label=cls_pred)
    pooled_reg_metrics = (
        _regression_metrics(np.concatenate(y_reg_true_parts), np.concatenate(y_reg_pred_parts))
        if y_reg_true_parts
        else _empty_regression_metrics()
    )

    cls_fold_summary = _summarize_metric_distribution(
        [{"test_metrics": item.get("test_metrics", {}).get("classification", {})} for item in fold_metrics],
        "test_metrics",
    )
    reg_fold_summary = _summarize_metric_distribution(
        [{"test_metrics": item.get("test_metrics", {}).get("regression", {})} for item in fold_metrics],
        "test_metrics",
    )
    return {
        "pooled_out_of_fold_metrics": {
            "classification": pooled_cls_metrics,
            "regression": pooled_reg_metrics,
        },
        "per_fold_test_metrics": {
            "classification": cls_fold_summary,
            "regression": reg_fold_summary,
        },
        "num_samples": int(sum(len(item["y_cls"]) for item in fold_predictions)),
        "num_regression_samples": int(reg_sample_count),
        "num_folds": int(len(fold_predictions)),
    }


def train_one(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    model_arch = _selected_model_arch(args)
    use_token_inputs = model_arch in {"cross_attn", "hybrid_cross_attn"}
    args.model_arch = model_arch
    if args.ab_model:
        args.vh_model = args.ab_model
        args.vl_model = args.ab_model

    known_models = set(list_registered_models())
    for role, model_name in (("env_model", args.env_model), ("vh_model", args.vh_model), ("vl_model", args.vl_model)):
        backend_model_name, _, display_model_name = _resolve_model_spec(model_name, args)
        if backend_model_name not in known_models:
            raise KeyError(
                f"{role}={display_model_name!r} resolves to backend {backend_model_name!r}, "
                "but that backend is not registered. Use --list_models or --list_protein_models."
            )

    data_path = _resolve_training_data_path(args)
    data = _prepare_dataframe(args, data_path, independent=False)
    antibody_id_info: Dict[str, Any] = {"enabled": False}
    antibody_name_to_raw_id: Dict[str, int] = {}
    if args.use_antibody_id_feature:
        data, antibody_id_info, antibody_name_to_raw_id = _attach_antibody_id_feature(
            data,
            args,
            independent=False,
        )
        args.num_ab_id_embeddings = int(data["__ab_id_index"].max()) + 1
    else:
        data = data.copy()
        data["__ab_id_index"] = 0
        args.num_ab_id_embeddings = 1
    splits = _group_kfold_splits(data, args) if args.cv_scheme == "group_kfold" else [_split_data(data, args)]

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / f"{run_id}_{args.task}_PAVE"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_base, vh_all, vl_all, cache_paths = _build_feature_matrices(data, args, cache_tag=f"{args.task}_train")
    env_embedding_dim = int(env_base.shape[1])

    # Token-level features for cross-attention mode
    env_tok_all: Optional[List[np.ndarray]] = None
    vh_tok_all: Optional[List[np.ndarray]] = None
    vl_tok_all: Optional[List[np.ndarray]] = None
    env_ind_toks: Optional[List[np.ndarray]] = None
    vh_ind_toks: Optional[List[np.ndarray]] = None
    vl_ind_toks: Optional[List[np.ndarray]] = None
    if use_token_inputs:
        env_tok_all, vh_tok_all, vl_tok_all, _ = _build_token_feature_matrices(
            data, args, cache_tag=f"{args.task}_train"
        )
    env_aux: Optional[np.ndarray] = None
    env_aux_names: Sequence[str] = []
    env_aux_parts: List[np.ndarray] = []
    env_aux_name_parts: List[str] = []
    env_aux_info: Dict[str, Any] = {"enabled": False, "dim": 0}
    if args.use_env_aux_features:
        env_aux_base, env_aux_base_names, env_aux_info = _build_env_aux_matrix(data, args)
        env_aux_parts.append(env_aux_base)
        env_aux_name_parts.extend(env_aux_base_names)
        env_aux_info.update(
            {
                "dim": int(env_aux_base.shape[1]),
                "feature_names": list(env_aux_base_names),
                "base_env_embedding_dim": env_embedding_dim,
            }
        )
    epitope_pos, epitope_pos_names, epitope_pos_info = _build_epitope_pos_matrix(data, args)
    if epitope_pos.shape[1] > 0:
        env_aux_parts.append(epitope_pos)
        env_aux_name_parts.extend(epitope_pos_names)
    epitope_token, epitope_token_names, epitope_token_info, epitope_token_cache = _build_epitope_token_matrix(
        data,
        args,
        cache_tag=f"{args.task}_train",
    )
    if epitope_token.shape[1] > 0:
        env_aux_parts.append(epitope_token)
        env_aux_name_parts.extend(epitope_token_names)
    if env_aux_parts:
        env_aux = np.concatenate(env_aux_parts, axis=1).astype(np.float32)
        env_aux_names = env_aux_name_parts
    total_env_aux_dim = int(env_aux.shape[1]) if env_aux is not None else 0
    env_aux_info["augmented_env_dim"] = int(env_embedding_dim + total_env_aux_dim)
    env_aux_info["combined_env_aux_dim"] = total_env_aux_dim

    independent_cohorts: List[Dict[str, Any]] = []
    if args.eval_independent:
        ind_paths = [Path(p).expanduser().resolve() for p in args.independent_json]
        if args.task == "regression":
            default_pair = [Path(DEFAULT_INDEPENDENT_JSON).resolve(), Path(DEFAULT_VRC601_INDEPENDENT_JSON).resolve()]
            if ind_paths == default_pair and len(ind_paths) == 2:
                ind_paths = [default_pair[0]]
                CONSOLE.print(
                    "[yellow]Regression independent eval: CH505 only (VRC601 JSON has no default regression target; "
                    "add labels or pass a custom --independent_json list).[/yellow]"
                )
        if args.task == "multitask" and len(ind_paths) > 1:
            CONSOLE.print(
                "[yellow]Multitask training: independent evaluation uses only the first "
                f"--independent_json; skipping {', '.join(str(p) for p in ind_paths[1:])}[/yellow]"
            )
            ind_paths = ind_paths[:1]
        for cohort_index, ind_path in enumerate(ind_paths):
            cohort_stem = ind_path.stem
            ind_eval_dir = run_dir if cohort_index == 0 else (run_dir / f"independent_{cohort_stem}")
            if cohort_index > 0:
                ind_eval_dir.mkdir(parents=True, exist_ok=True)
            cache_tag_ind = (
                f"{args.task}_independent"
                if cohort_index == 0
                else f"{args.task}_independent_{cohort_stem}"
            )
            ind_data_c = _prepare_dataframe(args, ind_path, independent=True)
            cohort_ab_info: Dict[str, Any] = {"enabled": False}
            if args.use_antibody_id_feature:
                ind_data_c, cohort_ab_info, antibody_name_to_raw_id = _attach_antibody_id_feature(
                    ind_data_c,
                    args,
                    independent=True,
                    known_name_to_raw_id=antibody_name_to_raw_id,
                )
                args.num_ab_id_embeddings = max(
                    int(args.num_ab_id_embeddings),
                    int(ind_data_c["__ab_id_index"].max()) + 1,
                )
                antibody_id_info["num_embedding_tokens"] = int(args.num_ab_id_embeddings)
            else:
                ind_data_c = ind_data_c.copy()
                ind_data_c["__ab_id_index"] = 0
            env_ind_b, vh_i, vl_i, ind_cache_paths_c = _build_feature_matrices(
                ind_data_c,
                args,
                cache_tag=cache_tag_ind,
            )
            env_aux_ind_c: Optional[np.ndarray] = None
            env_aux_ind_info_c: Dict[str, Any] = {}
            epitope_pos_ind_info_c: Dict[str, Any] = {}
            epitope_token_ind_info_c: Dict[str, Any] = {}
            epitope_token_ind_cache_c: Optional[Path] = None
            ind_aux_parts_c: List[np.ndarray] = []
            if args.use_env_aux_features:
                env_aux_ind_base_c, _, env_aux_ind_info_c = _build_env_aux_matrix(ind_data_c, args)
                ind_aux_parts_c.append(env_aux_ind_base_c)
            epitope_pos_ind_c, _, epitope_pos_ind_info_c = _build_epitope_pos_matrix(ind_data_c, args)
            if epitope_pos_ind_c.shape[1] > 0:
                ind_aux_parts_c.append(epitope_pos_ind_c)
            epitope_token_ind_c, _, epitope_token_ind_info_c, epitope_token_ind_cache_c = _build_epitope_token_matrix(
                ind_data_c,
                args,
                cache_tag=cache_tag_ind,
            )
            if epitope_token_ind_c.shape[1] > 0:
                ind_aux_parts_c.append(epitope_token_ind_c)
            if ind_aux_parts_c:
                env_aux_ind_c = np.concatenate(ind_aux_parts_c, axis=1).astype(np.float32)
            env_ind_toks_c: Optional[List[np.ndarray]] = None
            vh_ind_toks_c: Optional[List[np.ndarray]] = None
            vl_ind_toks_c: Optional[List[np.ndarray]] = None
            if use_token_inputs:
                env_ind_toks_c, vh_ind_toks_c, vl_ind_toks_c, _ = _build_token_feature_matrices(
                    ind_data_c, args, cache_tag=cache_tag_ind
                )
            independent_cohorts.append(
                {
                    "name": cohort_stem,
                    "path": ind_path,
                    "eval_dir": ind_eval_dir,
                    "ind_data": ind_data_c,
                    "env_ind_base": env_ind_b,
                    "vh_ind": vh_i,
                    "vl_ind": vl_i,
                    "env_aux_ind": env_aux_ind_c,
                    "ind_cache_paths": ind_cache_paths_c,
                    "env_aux_ind_info": env_aux_ind_info_c,
                    "epitope_pos_ind_info": epitope_pos_ind_info_c,
                    "epitope_token_ind_info": epitope_token_ind_info_c,
                    "epitope_token_ind_cache": epitope_token_ind_cache_c,
                    "independent_antibody_id_info": cohort_ab_info,
                    "env_ind_toks": env_ind_toks_c,
                    "vh_ind_toks": vh_ind_toks_c,
                    "vl_ind_toks": vl_ind_toks_c,
                    "fold_predictions": [],
                    "fold_summaries": [],
                    "ensemble_metrics": {},
                    "ensemble_prediction_files": {},
                }
            )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    _print_training_feature_summary(
        args,
        env_base=env_base,
        vh_all=vh_all,
        vl_all=vl_all,
        env_aux=env_aux,
        epitope_pos=epitope_pos,
        epitope_token=epitope_token,
        env_aux_info=env_aux_info,
        env_tok_all=env_tok_all,
        vh_tok_all=vh_tok_all,
        vl_tok_all=vl_tok_all,
    )
    fold_metrics: List[Dict[str, Any]] = []
    fold_checkpoints: List[str] = []
    fold_prediction_files: Dict[str, Dict[str, str]] = {}
    fold_test_predictions: List[Dict[str, Any]] = []

    for fold_index, split in enumerate(splits, start=1):
        env_all, env_aux_stats = _standardize_env_for_split(env_base, env_aux, env_aux_names, split, args)
        fold_dir = run_dir / f"fold_{fold_index}"
        if args.task == "multitask":
            fold_summary, model, fold_test_prediction = _train_single_fold_multitask(
                fold_index,
                split,
                data,
                env_all,
                vh_all,
                vl_all,
                args,
                device,
                fold_dir,
                env_tok_all=env_tok_all,
                vh_tok_all=vh_tok_all,
                vl_tok_all=vl_tok_all,
            )
        else:
            fold_summary, model, fold_test_prediction = _train_single_fold(
                fold_index,
                split,
                data,
                env_all,
                vh_all,
                vl_all,
                args,
                device,
                fold_dir,
                env_tok_all=env_tok_all,
                vh_tok_all=vh_tok_all,
                vl_tok_all=vl_tok_all,
            )
        fold_test_predictions.append(fold_test_prediction)
        fold_summary["env_aux_features"] = {
            **env_aux_info,
            "standardized_with_fold_train_split": bool(env_aux_stats is not None),
        }
        fold_metrics.append(fold_summary)
        fold_checkpoints.append(str(fold_summary["checkpoint"]))
        fold_prediction_files[f"fold_{fold_index}"] = fold_summary["prediction_files"]

        if args.eval_independent and independent_cohorts:
            ind_thr = float(fold_summary.get("classification_prob_threshold", 0.5)) if args.task in {"classification", "multitask"} else 0.5
            fold_pf = fold_prediction_files[f"fold_{fold_index}"]
            fold_pf.setdefault("independent_tests", {})
            for coh in independent_cohorts:
                if coh["env_ind_base"] is None or coh["vh_ind"] is None or coh["vl_ind"] is None:
                    continue
                if args.task == "multitask":
                    ind_fold = _evaluate_independent_fold_multitask(
                        fold_index,
                        model,
                        split,
                        coh["ind_data"],
                        coh["env_ind_base"],
                        coh["vh_ind"],
                        coh["vl_ind"],
                        coh["env_aux_ind"],
                        env_aux_stats,
                        args,
                        device,
                        run_dir,
                        eval_output_dir=coh["eval_dir"],
                        classification_prob_threshold=ind_thr,
                        env_ind_toks=coh["env_ind_toks"],
                        vh_ind_toks=coh["vh_ind_toks"],
                        vl_ind_toks=coh["vl_ind_toks"],
                    )
                else:
                    ind_fold = _evaluate_independent_fold(
                        fold_index,
                        model,
                        split,
                        coh["ind_data"],
                        coh["env_ind_base"],
                        coh["vh_ind"],
                        coh["vl_ind"],
                        coh["env_aux_ind"],
                        env_aux_stats,
                        args,
                        device,
                        run_dir,
                        eval_output_dir=coh["eval_dir"],
                        classification_prob_threshold=ind_thr,
                        env_ind_toks=coh["env_ind_toks"],
                        vh_ind_toks=coh["vh_ind_toks"],
                        vl_ind_toks=coh["vl_ind_toks"],
                    )
                coh["fold_predictions"].append(ind_fold)
                coh["fold_summaries"].append(
                    {
                        "fold": fold_index,
                        "metrics": ind_fold["metrics"],
                        "prediction_file": ind_fold["prediction_file"],
                    }
                )
                fold_pf["independent_tests"][coh["name"]] = ind_fold["prediction_file"]
            if independent_cohorts and independent_cohorts[0]["fold_predictions"]:
                fold_pf["independent_test"] = independent_cohorts[0]["fold_predictions"][-1]["prediction_file"]

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_independent_metrics: Dict[str, Any] = {}
    all_ensemble_prediction_files: Dict[str, str] = {}
    if args.eval_independent and independent_cohorts:
        for cohort_index, coh in enumerate(independent_cohorts):
            if not coh["fold_predictions"]:
                continue
            if args.task == "multitask":
                ens, pred_files = _write_independent_ensemble_multitask(
                    coh["fold_predictions"],
                    coh["ind_data"],
                    args,
                    run_dir,
                    eval_output_dir=coh["eval_dir"],
                )
            else:
                ens, pred_files = _write_independent_ensemble(
                    coh["fold_predictions"],
                    coh["ind_data"],
                    args,
                    run_dir,
                    eval_output_dir=coh["eval_dir"],
                )
            coh["ensemble_metrics"] = ens
            coh["ensemble_prediction_files"] = pred_files
            for k, v in pred_files.items():
                out_key = k if cohort_index == 0 else f"{k}__{coh['name']}"
                all_ensemble_prediction_files[out_key] = v
            if cohort_index == 0:
                ensemble_independent_metrics = ens

    cv_test_summary = (
        _summarize_cv_test_results_multitask(fold_test_predictions, fold_metrics)
        if args.task == "multitask"
        else _summarize_cv_test_results(fold_test_predictions, fold_metrics, args)
    )

    output: Dict[str, Any] = {
        "task": args.task,
        "run_dir": str(run_dir),
        "args": dict(vars(args)),
        "data_json": str(data_path),
        "dataset_variant": str(getattr(args, "dataset_variant", "default")),
        "label_col": args.label_col,
        "reg_label_col": args.reg_label_col if args.task == "multitask" else None,
        "group_col": args.group_col,
        "balance_train_by_antibody": bool(getattr(args, "balance_train_by_antibody", False)),
        "cv_scheme": args.cv_scheme,
        "num_folds": int(args.num_folds),
        "sequence_columns": {
            "env": args.env_seq_col,
            "env_aligned": args.env_aligned_col,
            "env_glyco": args.env_glyco_col,
            "vh": args.vh_seq_col,
            "vl": args.vl_seq_col,
        },
        "models": {
            "env": args.env_model,
            "vh": args.vh_model,
            "vl": args.vl_model,
        },
        "model_specs": {
            role: {
                "display_name": display_name,
                "backend_model_name": backend_name,
                "model_dir": model_dir,
            }
            for role, (backend_name, model_dir, display_name) in {
                "env": _resolve_model_spec(args.env_model, args),
                "vh": _resolve_model_spec(args.vh_model, args),
                "vl": _resolve_model_spec(args.vl_model, args),
            }.items()
        },
        "embedding_backend": args.embedding_backend,
        "embedding_cache": cache_paths,
        "training_device": {
            "device": str(device),
            "data_parallel": bool(device.type == "cuda" and args.data_parallel and torch.cuda.device_count() > 1),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() and not args.cpu else 0,
        },
        "antibody_id_feature": antibody_id_info,
        "env_aux_features": {
            **env_aux_info,
            "standardized_per_fold_train_split": bool(env_aux is not None and env_aux.shape[1] > 0),
        },
        "epitope_pos_features": epitope_pos_info,
        "epitope_token_features": {
            **epitope_token_info,
            "cache": str(epitope_token_cache) if epitope_token_cache is not None else None,
        },
        "fold_checkpoints": fold_checkpoints,
        "prediction_files": {
            "folds": fold_prediction_files,
            **all_ensemble_prediction_files,
        },
        "hyperparams": {
            "lambda_hl": args.lambda_hl,
            "lambda_va": args.lambda_va,
            "lambda_supcon": args.lambda_supcon,
            "temperature": args.temperature,
            "supcon_reg_sigma": args.supcon_reg_sigma,
            "proj_dim": args.proj_dim,
            "fusion_hidden": args.fusion_hidden,
            "dropout": args.dropout,
            "interaction_rank": args.interaction_rank,
            "num_res_blocks": args.num_res_blocks,
            "head_hidden_mult": args.head_hidden_mult,
            "use_env_aux_features": args.use_env_aux_features,
            "env_aux_region_bins": args.env_aux_region_bins,
            "use_epitope_pos_features": args.use_epitope_pos_features,
            "use_epitope_token_features": args.use_epitope_token_features,
            "epitope_token_pool": args.epitope_token_pool,
            "use_antibody_id_feature": args.use_antibody_id_feature,
            "lambda_cls": args.lambda_cls,
            "lambda_reg": args.lambda_reg,
            "val_prob_threshold_tuning": args.val_prob_threshold_tuning,
            "classification_threshold_metric": args.classification_threshold_metric,
            "use_focal_loss": args.use_focal_loss,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
        },
        "fold_metrics": fold_metrics,
        "cv_test_summary": cv_test_summary,
    }

    if args.eval_independent and independent_cohorts:
        first_cohort = independent_cohorts[0]
        output["independent_json"] = str(first_cohort["path"])
        output["independent_json_list"] = [str(c["path"]) for c in independent_cohorts]
        output["independent_embedding_cache"] = first_cohort["ind_cache_paths"]
        output["independent_embedding_caches"] = {c["name"]: c["ind_cache_paths"] for c in independent_cohorts}
        output["independent_fold_metrics"] = first_cohort["fold_summaries"]
        output["ensemble_independent_metrics"] = ensemble_independent_metrics
        independent_evaluations: List[Dict[str, Any]] = []
        for c in independent_cohorts:
            record: Dict[str, Any] = {
                "name": c["name"],
                "independent_json": str(c["path"]),
                "eval_output_dir": str(c["eval_dir"]),
                "fold_metrics": c["fold_summaries"],
                "ensemble_metrics": c["ensemble_metrics"],
                "ensemble_prediction_files": c["ensemble_prediction_files"],
                "embedding_cache": c["ind_cache_paths"],
                "antibody_id_feature": c["independent_antibody_id_info"],
            }
            if args.task == "classification" and "ch505" in str(c["name"]).lower():
                ens_path = (c.get("ensemble_prediction_files") or {}).get("independent_test_ensemble")
                if ens_path:
                    record["ch505_grouped_accuracy"] = _ch505_ensemble_grouped_accuracy(ens_path)
            independent_evaluations.append(record)
        output["independent_evaluations"] = independent_evaluations
        output["antibody_id_feature"]["independent"] = first_cohort["independent_antibody_id_info"]
        env_aux_ind_f = first_cohort["env_aux_ind"]
        env_aux_ind_info_f = first_cohort["env_aux_ind_info"]
        env_ind_base_f = first_cohort["env_ind_base"]
        if env_aux_ind_f is not None and env_aux_ind_f.shape[1] > 0:
            output["env_aux_features"]["independent"] = {
                "aligned_col_present": env_aux_ind_info_f.get("aligned_col_present", False),
                "glyco_col_present": env_aux_ind_info_f.get("glyco_col_present", False),
                "combined_env_aux_dim": int(env_aux_ind_f.shape[1]),
                "augmented_env_dim": int(
                    (env_ind_base_f.shape[1] if env_ind_base_f is not None else 0) + env_aux_ind_f.shape[1]
                ),
            }
        epitope_pos_ind_info_f = first_cohort["epitope_pos_ind_info"]
        if epitope_pos_ind_info_f:
            output["epitope_pos_features"]["independent"] = epitope_pos_ind_info_f
        epitope_token_ind_info_f = first_cohort["epitope_token_ind_info"]
        epitope_token_ind_cache_f = first_cohort["epitope_token_ind_cache"]
        if epitope_token_ind_info_f:
            output["epitope_token_features"]["independent"] = {
                **epitope_token_ind_info_f,
                "cache": str(epitope_token_ind_cache_f) if epitope_token_ind_cache_f is not None else None,
            }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    _print_final_output(output)

# python train_pave.py --task regression
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PAVE on JSON virus-antibody datasets.")
    parser.add_argument("--task", choices=["classification", "regression", "multitask"], required=True)
    parser.add_argument(
        "--data_json",
        type=str,
        default="",
        help=(
            "JSON dataset path. When set, overrides --dataset_variant. "
            "Otherwise defaults depend on --task and --dataset_variant."
        ),
    )
    parser.add_argument(
        "--dataset_variant",
        choices=list(DATASET_VARIANT_CHOICES),
        default="default",
        help=(
            "Training JSON variant when --data_json is not set: "
            "'default' -> virus_antibody_*_dataset.json; "
            "'clean' -> virus_antibody_*_dataset_clean.json."
        ),
    )
    parser.add_argument("--label_col", type=str, default="", help="Target column. Defaults: binary_label_50/log_ic50_exact.")
    parser.add_argument("--reg_label_col", type=str, default="log_ic50_exact", help="Regression target column for multitask training.")
    parser.add_argument(
        "--group_col",
        type=str,
        default="virus_id",
        choices=["virus_id", "ab_id", "study_id", "virus_year", "random"],
        help=(
            "Column used to keep an entity intact across train/val/test (no leakage). "
            "'virus_id' -> unseen virus; 'ab_id' -> unseen antibody; "
            "'study_id' -> unseen study/source; 'virus_year' -> temporal split; "
            "'random' -> ignore grouping and split rows randomly."
        ),
    )
    parser.add_argument("--exact_only", action="store_true", default=True)
    parser.add_argument("--include_censored", dest="exact_only", action="store_false")
    parser.add_argument("--limit_rows", type=int, default=0)

    parser.add_argument("--env_seq_col", type=str, default="env_seq")
    parser.add_argument(
        "--token_env_seq_col",
        type=str,
        default="",
        help=(
            "Sequence column to use for the Env token embeddings when "
            "--use_token_cross_attention is set.  Defaults to --env_seq_col "
            "(i.e. 'env_seq').  Set to 'env_seq_aligned' to embed the "
            "HXB2-aligned Env sequence instead."
        ),
    )
    parser.add_argument("--env_aligned_col", type=str, default="env_seq_aligned")
    parser.add_argument("--env_glyco_col", type=str, default="env_seq_glyco")
    parser.add_argument("--use_env_aux_features", action="store_true", default=False)
    parser.add_argument("--no_env_aux_features", dest="use_env_aux_features", action="store_false")
    parser.add_argument("--env_aux_region_bins", type=int, default=4)
    parser.add_argument("--use_epitope_pos_features", action="store_true", default=False)
    parser.add_argument("--no_epitope_pos_features", dest="use_epitope_pos_features", action="store_false")
    parser.add_argument("--epitope_feature_csv", type=str, default=str(DEFAULT_EPITOPE_FEATURE_CSV))
    parser.add_argument("--alignment_fasta", type=str, default=str(DEFAULT_ALIGNMENT_FASTA))
    parser.add_argument("--epitope_col", type=str, default="ab_epitope_simple")
    parser.add_argument("--default_epitope_region", type=str, default="CD4bs")
    parser.add_argument("--use_epitope_token_features", action="store_true", default=True)
    parser.add_argument("--no_epitope_token_features", dest="use_epitope_token_features", action="store_false")
    parser.add_argument("--epitope_token_pool", choices=["mean", "max"], default="mean")
    parser.add_argument("--vh_seq_col", type=str, default="vh_seq")
    parser.add_argument("--vl_seq_col", type=str, default="vl_seq")
    parser.add_argument("--strip_stop", action="store_true", default=True)
    parser.add_argument("--keep_stop", dest="strip_stop", action="store_false")
    parser.add_argument("--strip_gaps", action="store_true", default=True)
    parser.add_argument("--keep_gaps", dest="strip_gaps", action="store_false")

    protein_model_help = "Supported protein language models: " + ", ".join(PROTEIN_LANGUAGE_MODELS)
    parser.add_argument("--env_model", type=str, default="ESMC-600M", help=protein_model_help)
    parser.add_argument("--vh_model", type=str, default="Antiberta2-H", help=protein_model_help)
    parser.add_argument("--vl_model", type=str, default="Antiberta2-L", help=protein_model_help)
    parser.add_argument("--ab_model", type=str, default="", help=f"Set both --vh_model and --vl_model. {protein_model_help}")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--list_protein_models", action="store_true")
    parser.add_argument("--check_conda_env", action="store_true")
    parser.add_argument("--strict_conda_env", action="store_true")
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument(
        "--embedding_num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use for embedding extraction (0 = use all available, default: 0).",
    )
    parser.add_argument("--embedding_backend", choices=["pave", "hash"], default="pave")
    parser.add_argument("--hash_embedding_dim", type=int, default=128)
    parser.add_argument("--embedding_pool_override", type=str, default="")
    parser.add_argument("--embedding_max_length", type=int, default=0)
    parser.add_argument("--token_pool", choices=["mean", "first", "last"], default="mean")
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--rebuild_cache", action="store_true")

    parser.add_argument("--eval_independent", action="store_true", default=True)
    parser.add_argument("--no_eval_independent", dest="eval_independent", action="store_false")
    parser.add_argument(
        "--independent_json",
        nargs="+",
        default=[str(p) for p in DEFAULT_INDEPENDENT_JSONS],
        help=(
            "One or more independent-test JSON files (same protocol as CH505: "
            "env/vh/vl sequences, aligned Env optional, classification label column or ic50-derived binary). "
            "Default: CH505 validation + VRC601 validation. Outputs after the first cohort go under "
            "run_dir/independent_<stem>/."
        ),
    )
    parser.add_argument("--ind_label_col", type=str, default="")
    parser.add_argument("--ind_reg_label_col", type=str, default="")
    parser.add_argument("--ind_limit_rows", type=int, default=0)
    parser.add_argument("--use_antibody_id_feature", action="store_true", default=False)
    parser.add_argument("--no_antibody_id_feature", dest="use_antibody_id_feature", action="store_false")
    parser.add_argument("--antibody_id_mapping_json", type=str, default=str(DEFAULT_ANTIBODY_ID_MAPPING_JSON))

    parser.add_argument("--output_root", type=str, default=str(ROOT / "results" / "pave" / "runs"))
    parser.add_argument("--cv_scheme", choices=["group_kfold", "single"], default="group_kfold")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance_train_by_antibody",
        action="store_true",
        default=False,
        help=(
            "Within each CV fold, upsample the minority class per antibody on the training split only "
            "(Sensitive/Resistant counts equalized). Val/test unchanged. Classification and multitask only."
        ),
    )
    parser.add_argument(
        "--no_balance_train_by_antibody",
        dest="balance_train_by_antibody",
        action="store_false",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--data_parallel",
        "--data-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use torch.nn.DataParallel across all visible CUDA devices when more than one GPU is available. "
            "Use --no-data-parallel to force single-GPU training."
        ),
    )

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument(
        "--token_batch_size",
        type=int,
        default=256,
        help=(
            "Per-batch sample cap for token cross-attention loaders. "
            "Uses min(batch_size, token_batch_size); keeps full-length tokens without truncation."
        ),
    )
    parser.add_argument(
        "--token_num_workers",
        type=int,
        default=32,
        help=(
            "DataLoader workers for token cross-attention loaders. "
            "Default 0 avoids worker/prefetch RAM spikes with full-length token embeddings."
        ),
    )
    parser.add_argument(
        "--token_bucket_sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Group samples by similar total token length (env+vh+vl) to minimise padding waste. "
            "Shuffles batch ORDER each epoch so training diversity is preserved. "
            "Typically raises GPU utilisation from <20%% to >60%%. "
            "Use --no-token_bucket_sort to disable."
        ),
    )
    parser.add_argument(
        "--token_max_cache_shards",
        type=int,
        default=256,
        help=(
            "Maximum number of shards to keep in the ShardedTokenStore LRU cache per store. "
            "Each shard holds shard_size (default 256) token-embedding sequences. "
            "Increase for faster I/O (fewer disk re-reads); decrease to reduce RAM usage. "
            "Default 32: with fp16+proj256 each shard is ~3 MB, so 32 shards × 3 modalities ≈ 288 MB."
        ),
    )
    parser.add_argument(
        "--token_prefetch_factor",
        type=int,
        default=4,
        help=(
            "DataLoader prefetch_factor for token cross-attention loaders (only used when "
            "token_num_workers > 0).  Each worker pre-loads this many batches ahead, hiding "
            "shard I/O latency between epochs.  Default 4; reduce to 2 if RAM is tight."
        ),
    )
    parser.add_argument(
        "--token_shard_proj_dim",
        type=int,
        default=64,
        help=(
            "Project each token embedding to this dimension with a fixed random Gaussian matrix "
            "before writing shard files.  Reduces shard size proportionally (e.g. ESMC-600M "
            "1280→256 is a 5× saving).  Combined with fp16 storage this gives ~10× RAM reduction. "
            "Set 0 to disable projection and keep the original model dimension."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--fusion_hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--interaction_rank", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--head_hidden_mult", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--lambda_hl", type=float, default=0.003)
    parser.add_argument("--lambda_va", type=float, default=0.003)
    parser.add_argument("--lambda_supcon", type=float, default=0.01)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--supcon_reg_sigma", type=float, default=0.5)
    parser.add_argument("--huber_delta", type=float, default=0.5)
    parser.add_argument("--use_hard_negative", action="store_true")
    parser.add_argument("--hard_negative_topk", type=int, default=8)
    parser.add_argument("--silent_epoch_metrics", action="store_true")
    parser.add_argument("--val_prob_threshold_tuning", action="store_true", default=True)
    parser.add_argument("--no_val_prob_threshold_tuning", dest="val_prob_threshold_tuning", action="store_false")
    parser.add_argument(
        "--classification_threshold_metric",
        choices=["f1", "mcc"],
        default="f1",
        help="Objective on the validation set when tuning the classification probability threshold.",
    )
    parser.add_argument(
        "--early_stop_metric_cls",
        choices=["acc", "auc", "f1"],
        default="auc",
        help="Early-stopping metric for classification tasks.",
    )
    parser.add_argument(
        "--early_stop_metric_reg",
        choices=["r2", "R", "rmse"],
        default="rmse",
        help="Early-stopping metric for regression tasks (R maps to pearson_r).",
    )
    parser.add_argument("--use_focal_loss", action="store_true", default=True, help="Use focal loss for the classification BCE head (default: on).")
    parser.add_argument("--no_focal_loss", dest="use_focal_loss", action="store_false", help="Use plain BCE for the classification head.")
    parser.add_argument("--focal_alpha", type=float, default=0.75, help="Focal loss alpha (weight for positive class).")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (focusing parameter).")

    # Token-level cross-attention
    parser.add_argument(
        "--use_token_cross_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use token-level cross-attention (PAVECrossAttn) instead of pooled PAVE. "
             "Antibody tokens attend to Env tokens via multi-head cross-attention. "
             "Use --no-use-token-cross-attention to disable.",
    )
    parser.add_argument(
        "--model_arch",
        choices=["auto", "pave", "cross_attn", "hybrid_cross_attn"],
        default="hybrid_cross_attn",
        help=(
            "Model architecture selector. auto preserves legacy behavior "
            "(cross_attn when --use_token_cross_attention is on, else pave)."
        ),
    )
    parser.add_argument(
        "--cross_attn_heads",
        type=int,
        default=8,
        help="Number of attention heads in CrossAttentionEncoder (default: 8).",
    )
    parser.add_argument(
        "--cross_attn_layers",
        type=int,
        default=2,
        help="Number of cross-attention + FFN layers in CrossAttentionEncoder (default: 2).",
    )
    parser.add_argument(
        "--hybrid_token_gate_hidden",
        type=int,
        default=0,
        help="Hidden size for hybrid token gate MLP (0 -> fusion_hidden).",
    )
    parser.add_argument(
        "--cross_attn_max_env_len",
        type=int,
        default=0,
        help="Maximum Env token sequence length (0 = no truncation, default: 0).",
    )
    parser.add_argument(
        "--cross_attn_max_ab_len",
        type=int,
        default=0,
        help="Maximum antibody (VH+VL concatenated) token sequence length (0 = no truncation, default: 0).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        _print_registered_models(list_registered_models())
        return
    if args.list_protein_models:
        registered = set(list_registered_models())
        rows: List[Tuple[str, str, str]] = []
        for name in PROTEIN_LANGUAGE_MODELS:
            preset = PROTEIN_MODEL_PRESETS[name]
            backend = preset["model_name"]
            suffix = "" if backend in registered else " (backend not registered)"
            rows.append((name, f"{backend}{suffix}", preset["model_dir"]))
        _print_protein_models_table(rows)
        return
    if not args.label_col:
        args.label_col = "binary_label_50" if args.task in {"classification", "multitask"} else "log_ic50_exact"
    train_one(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run PAVE inference with a saved 5-fold checkpoint ensemble on JSON cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_pave as tp

DEFAULT_CHECKPOINT_ROOT = (
    ROOT / "results" / "pave" / "focal_alpha_sweep_hybrid" / "focal_alpha0p75"
)
DEFAULT_DATASETS: Tuple[Path, ...] = (
    ROOT / "data" / "no_label" / "ch505_beast_ancestral_validation.json",
    ROOT / "data" / "no_label" / "ch505_beast_posterior_anchor_validation.json",
    ROOT / "data" / "no_label" / "ch505_validation_full.json",
    ROOT / "data" / "no_label" / "vrc601_validation_expanded_exact_available.json",
)


def _resolve_run_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "metrics.json").is_file():
        return path
    candidates = sorted(
        (p.parent for p in path.glob("*/metrics.json") if (p.parent / "fold_1" / "best_model.pt").is_file()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No PAVE run with metrics.json and fold_*/best_model.pt found under {path}"
        )
    return candidates[0]


def _namespace_from_saved_args(payload: Dict[str, Any]) -> argparse.Namespace:
    args_dict = dict(payload.get("args") or {})
    if not args_dict:
        raise ValueError("metrics.json is missing saved training args")
    args = argparse.Namespace(**args_dict)
    if not getattr(args, "data_json", None):
        args.data_json = str(payload.get("data_json") or "")
    if not getattr(args, "dataset_variant", None):
        args.dataset_variant = str(payload.get("dataset_variant") or "default")
    return args


def _load_training_context(run_dir: Path) -> Tuple[argparse.Namespace, Dict[str, Any], List[Dict[str, Any]]]:
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    args = _namespace_from_saved_args(metrics)
    fold_metrics = list(metrics.get("fold_metrics") or [])
    if not fold_metrics:
        raise ValueError(f"No fold_metrics in {metrics_path}")
    return args, metrics, fold_metrics


def _prepare_inference_frame(
    args: argparse.Namespace,
    dataset_path: Path,
    *,
    limit_rows: int = 0,
    require_ic50_label: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Prepare cohort rows for inference.

    Unlike training/independent eval in train_pave, this keeps rows with missing
    ic50 labels so unlabeled cohorts still receive predictions. Rows without a
    usable ic50 get a dummy __target=0.0 for the forward pass only.
    """
    data = tp._load_json_records(dataset_path)
    if "virus_id" not in data.columns and "virus" in data.columns:
        data = data.copy()
        data["virus_id"] = data["virus"].astype(str)

    n_input = len(data)
    data = data.reset_index(drop=True).copy()
    data["__row_id"] = np.arange(len(data), dtype=np.int64)
    if "idx" in data.columns:
        data["__display_id"] = (
            pd.to_numeric(data["idx"], errors="coerce").fillna(data["__row_id"]).astype(np.int64)
        )
    else:
        data["__display_id"] = data["__row_id"].astype(np.int64)

    required = [args.env_seq_col, args.vh_seq_col, args.vl_seq_col]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required sequence columns in {dataset_path}: {missing}")
    for col in required:
        data[col] = [
            tp._clean_sequence(v, strip_stop=args.strip_stop, strip_gaps=args.strip_gaps)
            for v in data[col].tolist()
        ]
    seq_ok = (
        (data[args.env_seq_col] != "")
        & (data[args.vh_seq_col] != "")
        & (data[args.vl_seq_col] != "")
    )
    n_missing_seq = int((~seq_ok).sum())
    data = data[seq_ok].copy()

    if "ic50" in data.columns:
        binary_targets = [tp._ic50_to_binary_label50(v) for v in data["ic50"].tolist()]
    else:
        binary_targets = [float("nan")] * len(data)
    label_available = np.array(
        [v is not None and not (isinstance(v, float) and np.isnan(v)) for v in binary_targets],
        dtype=bool,
    )
    data["ic50_label_available"] = label_available
    targets = np.asarray(binary_targets, dtype=np.float64)
    n_missing_ic50 = int((~label_available).sum())
    if require_ic50_label:
        data = data[label_available].copy()
        targets = targets[label_available]
    else:
        # Dummy label for forward pass; y_true is corrected after prediction export.
        targets = np.where(np.isfinite(targets), targets, 0.0)

    if args.task != "classification":
        raise ValueError(f"infer_pave currently supports classification runs only, got {args.task!r}")
    data["__target"] = targets.astype(np.float64)

    if limit_rows > 0:
        data = data.head(int(limit_rows)).copy()

    data = data.reset_index(drop=True)
    data["__row_id"] = np.arange(len(data), dtype=np.int64)
    if data.empty:
        raise ValueError(f"No usable records remain after filtering {dataset_path}")

    stats = {
        "input_rows": n_input,
        "dropped_missing_sequences": n_missing_seq,
        "dropped_missing_ic50_labels": n_missing_ic50 if require_ic50_label else 0,
        "rows_with_ic50_label": int(data["ic50_label_available"].sum()),
        "rows_without_ic50_label_included": int((~data["ic50_label_available"]).sum()),
        "output_rows": len(data),
    }
    return data, stats


def _load_cached_matrix(cache_path: Path, row_ids: np.ndarray) -> np.ndarray:
    payload = torch.load(cache_path, map_location="cpu")
    features = torch.as_tensor(payload["features"], dtype=torch.float32).numpy()
    cached_ids = torch.as_tensor(payload["row_ids"], dtype=torch.long).numpy()
    if not np.array_equal(cached_ids, row_ids):
        raise ValueError(
            f"Row ids in cache {cache_path} do not match the prepared training dataframe"
        )
    return features


def _build_training_feature_bundle(
    args: argparse.Namespace,
    train_data: pd.DataFrame,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    model_arch = tp._selected_model_arch(args)
    use_token_inputs = model_arch in {"cross_attn", "hybrid_cross_attn"}
    train_cache_tag = f"{args.task}_train"
    row_ids = train_data["__row_id"].to_numpy(dtype=np.int64)

    embedding_cache = metrics.get("embedding_cache") or {}
    env_cache_path = Path(embedding_cache["env"]) if embedding_cache.get("env") else None
    if env_cache_path and env_cache_path.is_file():
        env_base = _load_cached_matrix(env_cache_path, row_ids)
        vh_all = _load_cached_matrix(Path(embedding_cache["vh"]), row_ids)
        vl_all = _load_cached_matrix(Path(embedding_cache["vl"]), row_ids)
        cache_paths = {k: str(v) for k, v in embedding_cache.items()}
        tp.CONSOLE.print("[green]Loaded training pooled embeddings from metrics.json cache.[/]")
    else:
        env_base, vh_all, vl_all, cache_paths = tp._build_feature_matrices(
            train_data, args, cache_tag=train_cache_tag
        )

    env_tok_all = vh_tok_all = vl_tok_all = None
    if use_token_inputs:
        # Online UniCAV inference only needs training token feature dimensions to
        # rebuild the fold model. Loading every training token shard adds several
        # GB of artifact weight and is not used in the independent forward pass.
        tok_dim = int(getattr(args, "token_shard_proj_dim", 0) or 0)
        if tok_dim <= 0:
            env_tok_all, vh_tok_all, vl_tok_all, _ = tp._build_token_feature_matrices(
                train_data, args, cache_tag=train_cache_tag
            )
        else:
            dummy = np.zeros((1, tok_dim), dtype=np.float16)
            env_tok_all = [dummy]
            vh_tok_all = [dummy]
            vl_tok_all = [dummy]

    env_aux_parts: List[np.ndarray] = []
    env_aux_name_parts: List[str] = []
    if args.use_env_aux_features:
        env_aux_base, env_aux_base_names, _ = tp._build_env_aux_matrix(train_data, args)
        env_aux_parts.append(env_aux_base)
        env_aux_name_parts.extend(env_aux_base_names)
    epitope_pos, epitope_pos_names, _ = tp._build_epitope_pos_matrix(train_data, args)
    if epitope_pos.shape[1] > 0:
        env_aux_parts.append(epitope_pos)
        env_aux_name_parts.extend(epitope_pos_names)
    epitope_token, epitope_token_names, _, _ = tp._build_epitope_token_matrix(
        train_data, args, cache_tag=train_cache_tag
    )
    if epitope_token.shape[1] > 0:
        env_aux_parts.append(epitope_token)
        env_aux_name_parts.extend(epitope_token_names)
    env_aux = (
        np.concatenate(env_aux_parts, axis=1).astype(np.float32)
        if env_aux_parts
        else None
    )

    return {
        "env_base": env_base,
        "vh_all": vh_all,
        "vl_all": vl_all,
        "env_aux": env_aux,
        "env_aux_names": env_aux_name_parts,
        "env_tok_all": env_tok_all,
        "vh_tok_all": vh_tok_all,
        "vl_tok_all": vl_tok_all,
        "cache_paths": cache_paths,
        "use_token_inputs": use_token_inputs,
        "model_arch": model_arch,
    }


def _build_inference_feature_bundle(
    args: argparse.Namespace,
    ind_data: pd.DataFrame,
    cache_tag: str,
    train_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    use_token_inputs = bool(train_bundle["use_token_inputs"])
    env_ind_base, vh_ind, vl_ind, ind_cache_paths = tp._build_feature_matrices(
        ind_data, args, cache_tag=cache_tag
    )
    ind_aux_parts: List[np.ndarray] = []
    if args.use_env_aux_features:
        env_aux_ind_base, _, _ = tp._build_env_aux_matrix(ind_data, args)
        ind_aux_parts.append(env_aux_ind_base)
    epitope_pos_ind, _, _ = tp._build_epitope_pos_matrix(ind_data, args)
    if epitope_pos_ind.shape[1] > 0:
        ind_aux_parts.append(epitope_pos_ind)
    epitope_token_ind, _, _, _ = tp._build_epitope_token_matrix(ind_data, args, cache_tag=cache_tag)
    if epitope_token_ind.shape[1] > 0:
        ind_aux_parts.append(epitope_token_ind)
    env_aux_ind = (
        np.concatenate(ind_aux_parts, axis=1).astype(np.float32)
        if ind_aux_parts
        else None
    )

    env_ind_toks = vh_ind_toks = vl_ind_toks = None
    if use_token_inputs:
        env_ind_toks, vh_ind_toks, vl_ind_toks, _ = tp._build_token_feature_matrices(
            ind_data, args, cache_tag=cache_tag
        )

    return {
        "env_ind_base": env_ind_base,
        "vh_ind": vh_ind,
        "vl_ind": vl_ind,
        "env_aux_ind": env_aux_ind,
        "env_ind_toks": env_ind_toks,
        "vh_ind_toks": vh_ind_toks,
        "vl_ind_toks": vl_ind_toks,
        "ind_cache_paths": ind_cache_paths,
    }


def _load_fold_model(
    fold_index: int,
    split: tp.SplitData,
    train_bundle: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> nn.Module:
    env_all, _ = tp._standardize_env_for_split(
        train_bundle["env_base"],
        train_bundle["env_aux"],
        train_bundle["env_aux_names"],
        split,
        args,
    )
    vh_all = train_bundle["vh_all"]
    vl_all = train_bundle["vl_all"]
    model_arch = train_bundle["model_arch"]
    is_hybrid = model_arch == "hybrid_cross_attn"
    is_token = train_bundle["use_token_inputs"]

    if is_token:
        assert train_bundle["env_tok_all"] is not None
        tok_env_dim = int(train_bundle["env_tok_all"][0].shape[1])
        tok_vh_dim = int(train_bundle["vh_tok_all"][0].shape[1])
        tok_vl_dim = int(train_bundle["vl_tok_all"][0].shape[1])
        if is_hybrid:
            model = tp._build_hybrid_cross_attn_model(
                env_all.shape[1],
                vh_all.shape[1],
                vl_all.shape[1],
                tok_env_dim,
                tok_vh_dim,
                tok_vl_dim,
                args,
                device,
            )
        else:
            model = tp._build_cross_attn_model(tok_env_dim, tok_vh_dim, tok_vl_dim, args, device)
    else:
        model = tp._build_model(env_all.shape[1], vh_all.shape[1], vl_all.shape[1], args, device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model = tp._maybe_wrap_data_parallel(model, args, device)
    tp._load_model_state_dict(model, state_dict)
    model.eval()
    return model


def _fix_prediction_exports(output_dir: Path, ind_data: pd.DataFrame) -> None:
    """Mark y_true as missing where ic50 was not available in the source cohort."""
    label_by_id = ind_data.set_index(ind_data["__display_id"].astype(np.int64))[
        "ic50_label_available"
    ]
    for path in sorted(output_dir.glob("independent_test_predictions*.csv")):
        frame = pd.read_csv(path)
        if "id" not in frame.columns or "y_true" not in frame.columns:
            continue
        frame["ic50_label_available"] = frame["id"].map(label_by_id).fillna(False).astype(bool)
        frame.loc[~frame["ic50_label_available"], "y_true"] = np.nan
        frame.to_csv(path, index=False)


def infer_dataset(
    args: argparse.Namespace,
    run_dir: Path,
    fold_metrics: Sequence[Dict[str, Any]],
    train_data: pd.DataFrame,
    train_bundle: Dict[str, Any],
    dataset_path: Path,
    output_dir: Path,
    device: torch.device,
    *,
    limit_rows: int = 0,
    require_ic50_label: bool = False,
) -> Dict[str, Any]:
    cohort_stem = dataset_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    ind_data, row_stats = _prepare_inference_frame(
        args,
        dataset_path,
        limit_rows=limit_rows,
        require_ic50_label=require_ic50_label,
    )
    tp.CONSOLE.print(
        f"  rows: input={row_stats['input_rows']} "
        f"-> after_seq_filter={row_stats['output_rows']} "
        f"(dropped_missing_seq={row_stats['dropped_missing_sequences']}, "
        f"with_ic50_label={row_stats.get('rows_with_ic50_label', row_stats['output_rows'])})"
    )
    if args.use_antibody_id_feature:
        ind_data, _, _ = tp._attach_antibody_id_feature(
            ind_data,
            args,
            independent=True,
        )
    else:
        ind_data = ind_data.copy()
        ind_data["__ab_id_index"] = 0
    cache_tag = f"{args.task}_infer_{cohort_stem}"
    infer_bundle = _build_inference_feature_bundle(args, ind_data, cache_tag, train_bundle)

    splits = (
        tp._group_kfold_splits(train_data, args)
        if args.cv_scheme == "group_kfold"
        else [tp._split_data(train_data, args)]
    )
    if len(splits) != len(fold_metrics):
        raise ValueError(
            f"Expected {len(fold_metrics)} folds from metrics.json, got {len(splits)} splits"
        )

    fold_predictions: List[Dict[str, Any]] = []
    per_fold_outputs: List[Dict[str, Any]] = []

    for fold_index, (split, fold_info) in enumerate(zip(splits, fold_metrics), start=1):
        checkpoint_path = Path(fold_info["checkpoint"])
        if not checkpoint_path.is_file():
            checkpoint_path = run_dir / f"fold_{fold_index}" / "best_model.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold_index}: {checkpoint_path}")

        model = _load_fold_model(
            fold_index, split, train_bundle, args, device, checkpoint_path
        )
        env_all, env_aux_stats = tp._standardize_env_for_split(
            train_bundle["env_base"],
            train_bundle["env_aux"],
            train_bundle["env_aux_names"],
            split,
            args,
        )
        classification_prob_threshold = float(
            fold_info.get("classification_prob_threshold", 0.5)
        )
        ind_fold = tp._evaluate_independent_fold(
            fold_index,
            model,
            split,
            ind_data,
            infer_bundle["env_ind_base"],
            infer_bundle["vh_ind"],
            infer_bundle["vl_ind"],
            infer_bundle["env_aux_ind"],
            env_aux_stats,
            args,
            device,
            run_dir,
            eval_output_dir=output_dir,
            classification_prob_threshold=classification_prob_threshold,
            env_ind_toks=infer_bundle["env_ind_toks"],
            vh_ind_toks=infer_bundle["vh_ind_toks"],
            vl_ind_toks=infer_bundle["vl_ind_toks"],
        )
        fold_predictions.append(ind_fold)
        per_fold_outputs.append(
            {
                "fold": fold_index,
                "checkpoint": str(checkpoint_path),
                "classification_prob_threshold": classification_prob_threshold,
                "metrics": ind_fold.get("metrics"),
                "prediction_file": ind_fold.get("prediction_file"),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_metrics, ensemble_files = tp._write_independent_ensemble(
        fold_predictions,
        ind_data,
        args,
        run_dir,
        eval_output_dir=output_dir,
    )
    _fix_prediction_exports(output_dir, ind_data)

    summary = {
        "dataset": str(dataset_path),
        "cohort": cohort_stem,
        "num_records": int(len(ind_data)),
        "row_stats": row_stats,
        "output_dir": str(output_dir),
        "ensemble_metrics": ensemble_metrics,
        "ensemble_prediction_files": ensemble_files,
        "fold_outputs": per_fold_outputs,
        "inference_cache_paths": infer_bundle.get("ind_cache_paths"),
    }
    summary_path = output_dir / "inference_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def run_inference(
    checkpoint_root: Path,
    datasets: Sequence[Path],
    *,
    output_root: Optional[Path] = None,
    limit_rows: int = 0,
    cpu: bool = False,
    require_ic50_label: bool = False,
) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(checkpoint_root)
    args, metrics, fold_metrics = _load_training_context(run_dir)
    if cpu:
        args.cpu = True

    train_path = tp._resolve_training_data_path(args)
    train_data = tp._prepare_dataframe(args, train_path, independent=False)
    if args.use_antibody_id_feature:
        train_data, _, _ = tp._attach_antibody_id_feature(train_data, args, independent=False)
        args.num_ab_id_embeddings = int(train_data["__ab_id_index"].max()) + 1
    else:
        train_data = train_data.copy()
        train_data["__ab_id_index"] = 0
        args.num_ab_id_embeddings = 1

    train_bundle = _build_training_feature_bundle(args, train_data, metrics)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    out_root = (output_root or (run_dir / "inference_no_label")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tp.CONSOLE.print(f"[bold]Run dir[/]: {run_dir}")
    tp.CONSOLE.print(f"[bold]Task[/]: {args.task} | arch: {train_bundle['model_arch']}")
    tp.CONSOLE.print(f"[bold]Output[/]: {out_root}")

    cohort_summaries: List[Dict[str, Any]] = []
    for dataset_path in datasets:
        dataset_path = dataset_path.expanduser().resolve()
        cohort_dir = out_root / dataset_path.stem
        tp.CONSOLE.print(f"\n[cyan]Inferring[/] {dataset_path.name} -> {cohort_dir}")
        summary = infer_dataset(
            args,
            run_dir,
            fold_metrics,
            train_data,
            train_bundle,
            dataset_path,
            cohort_dir,
            device,
            limit_rows=limit_rows,
            require_ic50_label=require_ic50_label,
        )
        cohort_summaries.append(summary)
        ens_path = summary.get("ensemble_prediction_files", {}).get("independent_test_ensemble")
        if ens_path:
            tp.CONSOLE.print(f"  ensemble predictions: {ens_path}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "task": args.task,
        "training_data_json": str(train_path),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "output_root": str(out_root),
        "device": str(device),
        "datasets": [str(p) for p in datasets],
        "cohorts": cohort_summaries,
        "training_metrics_snapshot": {
            "cv_test_summary": metrics.get("cv_test_summary"),
            "ensemble_independent_metrics": metrics.get("ensemble_independent_metrics"),
        },
    }
    manifest_path = out_root / "inference_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    tp.CONSOLE.print(f"\n[green]Done.[/] Manifest: {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PAVE 5-fold ensemble inference on JSON cohorts using saved checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Training run directory, or parent directory containing a run with metrics.json.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Directory for inference outputs (default: <run_dir>/inference_no_label).",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        action="append",
        default=[],
        help="JSON cohort to score (repeatable). Defaults to the data/no_label CH505 + VRC601 files.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Optional smoke-test limit per dataset (0 = all rows).",
    )
    parser.add_argument(
        "--require-ic50-label",
        action="store_true",
        help="Drop rows without a parseable ic50 label (legacy eval behaviour).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser


def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()
    datasets = [Path(p) for p in cli.dataset_json] if cli.dataset_json else list(DEFAULT_DATASETS)
    output_root = Path(cli.output_root).expanduser().resolve() if cli.output_root else None
    run_inference(
        Path(cli.checkpoint_root),
        datasets,
        output_root=output_root,
        limit_rows=int(cli.limit_rows),
        cpu=bool(cli.cpu),
        require_ic50_label=bool(cli.require_ic50_label),
    )


if __name__ == "__main__":
    main()

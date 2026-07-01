from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

SPACE_ROOT = Path(__file__).resolve().parent
LOGO_PATH = SPACE_ROOT / "logo.png"
EXAMPLES_PATH = SPACE_ROOT / "examples.json"
ANTIBODY_MAP_PATH = SPACE_ROOT / "antibody_sequence_map.json"


def _logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _load_antibody_sequence_map() -> dict[str, dict[str, Any]]:
    if not ANTIBODY_MAP_PATH.exists():
        return {}
    payload = json.loads(ANTIBODY_MAP_PATH.read_text(encoding="utf-8"))
    antibodies = payload.get("antibodies") or {}
    return {
        str(name): record
        for name, record in antibodies.items()
        if isinstance(record, dict)
    }


ANTIBODY_SEQUENCE_MAP = _load_antibody_sequence_map()
ANTIBODY_CHOICES = sorted(ANTIBODY_SEQUENCE_MAP)


def _copy_or_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _artifact_root() -> Path:
    repo_id = os.environ.get("PAVE_ARTIFACT_REPO", "").strip()
    if repo_id:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type=os.environ.get("PAVE_ARTIFACT_REPO_TYPE", "model"),
                local_dir=os.environ.get("PAVE_ARTIFACT_DIR") or None,
            )
        )
    local = os.environ.get("PAVE_ARTIFACT_DIR", "").strip()
    if local:
        return Path(local).expanduser().resolve()
    _raise_missing_artifacts()
    return SPACE_ROOT


def _raise_missing_artifacts() -> None:
    raise gr.Error(
        "UniCAV model artifacts are not configured. Set PAVE_ARTIFACT_REPO to the "
        "Hugging Face model repo that contains data/, model_weight/, results/pave/"
        "embedding_cache/, and results/pave/focal_alpha_sweep_hybrid/focal_alpha0p75/."
    )


def _patch_train_pave_model_roots(model_weight_root: Path) -> None:
    import train_pave as tp

    tp.MODEL_WEIGHT_ROOT = model_weight_root
    for name, cfg in tp.PROTEIN_MODEL_PRESETS.items():
        if name.startswith("ESMC-600M"):
            cfg["model_dir"] = str(model_weight_root / "esmc-600m-2024-12")
        elif name.startswith("ESMC-300M"):
            cfg["model_dir"] = str(model_weight_root / "esmc-300m-2024-12")
        elif name.startswith("Antiberta2"):
            cfg["model_dir"] = str(model_weight_root / "antiberta2")


def _rewrite_token_cache_metadata(cache_dir: Path) -> None:
    import torch

    for meta_path in cache_dir.glob("*_tokens_*.pt"):
        try:
            payload = torch.load(meta_path, map_location="cpu")
        except Exception:
            continue
        shard_paths = payload.get("shard_paths") if isinstance(payload, dict) else None
        if not shard_paths:
            continue
        rewritten = []
        changed = False
        for raw in shard_paths:
            old = Path(str(raw))
            candidate = cache_dir / old.parent.name / old.name
            if candidate.exists():
                rewritten.append(str(candidate))
                changed = changed or str(candidate) != str(raw)
            else:
                rewritten.append(str(raw))
        if changed:
            payload["shard_paths"] = rewritten
            torch.save(payload, meta_path)


def _patch_metrics(run_dir: Path, artifact: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    data_json = artifact / "data" / "virus_antibody_classification_dataset.json"
    cache_dir = artifact / "results" / "pave" / "embedding_cache"

    metrics["data_json"] = str(data_json)
    args = dict(metrics.get("args") or {})
    args["data_json"] = str(data_json)
    args["cache_dir"] = str(cache_dir)
    args["output_root"] = str(run_dir.parent)
    if (artifact / "data" / "Env_HXB2_feature.csv").exists():
        args["epitope_feature_csv"] = str(artifact / "data" / "Env_HXB2_feature.csv")
    if (artifact / "data" / "ch505_aligned_to_training.fasta").exists():
        args["alignment_fasta"] = str(artifact / "data" / "ch505_aligned_to_training.fasta")
    metrics["args"] = args

    cache = dict(metrics.get("embedding_cache") or {})
    for key, value in list(cache.items()):
        if value:
            cache[key] = str(cache_dir / Path(str(value)).name)
    metrics["embedding_cache"] = cache

    for i, fold in enumerate(metrics.get("fold_metrics") or [], start=1):
        fold["checkpoint"] = str(run_dir / f"fold_{i}" / "best_model.pt")
        pred_files = fold.get("prediction_files") or {}
        for split_name, raw in list(pred_files.items()):
            pred_files[split_name] = str(run_dir / f"fold_{i}" / Path(str(raw)).name)

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if cache_dir.exists():
        _rewrite_token_cache_metadata(cache_dir)
        _alias_train_epitope_cache(cache_dir, data_json, args)


def _alias_train_epitope_cache(cache_dir: Path, data_json: Path, args_dict: dict[str, Any]) -> None:
    import argparse
    import train_pave as tp

    candidates = sorted(cache_dir.glob("classification_train_epitope_token_ESMC-600M_*.pt"))
    if not candidates:
        return
    args = argparse.Namespace(**args_dict)
    data = tp._prepare_dataframe(args, data_json, independent=False)
    _backend_model_name, _model_dir, display_model_name = tp._resolve_model_spec(args.env_model, args)
    expected = tp._epitope_token_cache_path(
        data,
        args,
        cache_tag=f"{args.task}_train",
        display_model_name=display_model_name,
    )
    if expected.exists():
        return
    expected.parent.mkdir(parents=True, exist_ok=True)
    _copy_or_link(candidates[0], expected)


def _prepare_runtime() -> tuple[Path, Path]:
    artifact = _artifact_root()
    runtime = Path(os.environ.get("PAVE_RUNTIME_DIR", SPACE_ROOT / "_runtime")).resolve()
    runtime.mkdir(parents=True, exist_ok=True)

    # If the Space repo contains only this deploy folder, link the project files
    # from the artifact. If the full project repo is copied into the Space, these
    # calls are no-ops.
    for name in ["models", "train_pave.py", "infer_pave.py"]:
        src = artifact / name
        if src.exists():
            _copy_or_link(src, SPACE_ROOT / name)

    for name in ["data", "model_weight", "DNAFM"]:
        src = artifact / name
        if src.exists():
            _copy_or_link(src, runtime / name)

    model_weight_root = runtime / "model_weight"
    if not model_weight_root.exists():
        model_weight_root = runtime / "DNAFM" / "model_weight"
    if not model_weight_root.exists():
        local_default = Path("/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight")
        if local_default.exists():
            model_weight_root = local_default
    _patch_train_pave_model_roots(model_weight_root)

    src_cache = artifact / "results" / "pave" / "embedding_cache"
    dst_cache = runtime / "results" / "pave" / "embedding_cache"
    dst_cache.mkdir(parents=True, exist_ok=True)
    if src_cache.exists():
        for item in src_cache.iterdir():
            dst = dst_cache / item.name
            if dst.exists() or dst.is_symlink():
                continue
            if item.is_dir() or item.stat().st_size > 5 * 1024 * 1024:
                _copy_or_link(item, dst)
            else:
                shutil.copy2(item, dst)

    src_checkpoint_root = artifact / "results" / "pave" / "focal_alpha_sweep_hybrid" / "focal_alpha0p75"
    checkpoint_root = runtime / "results" / "pave" / "focal_alpha_sweep_hybrid" / "focal_alpha0p75"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    src_run_dirs = sorted(src_checkpoint_root.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not src_run_dirs:
        raise FileNotFoundError(f"No metrics.json found under {src_checkpoint_root}")
    src_run_dir = src_run_dirs[0].parent
    run_dir = checkpoint_root / src_run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_run_dir / "metrics.json", run_dir / "metrics.json")
    for item in src_run_dir.iterdir():
        if item.name == "metrics.json":
            continue
        _copy_or_link(item, run_dir / item.name)

    run_dirs = sorted(checkpoint_root.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No metrics.json found under {checkpoint_root}")
    run_dir = run_dirs[0].parent
    _patch_metrics(run_dir, runtime)
    return runtime, checkpoint_root


ARTIFACT_ROOT: Path | None = None
CHECKPOINT_ROOT: Path | None = None


def _get_runtime() -> tuple[Path, Path]:
    global ARTIFACT_ROOT, CHECKPOINT_ROOT
    if ARTIFACT_ROOT is None or CHECKPOINT_ROOT is None:
        ARTIFACT_ROOT, CHECKPOINT_ROOT = _prepare_runtime()
    return ARTIFACT_ROOT, CHECKPOINT_ROOT


def _clean_text(value: str | None) -> str:
    return (value or "").strip()


def _forward_cross_attention(
    model: Any,
    batch: Any,
    device: Any,
    *,
    is_hybrid: bool,
) -> list[Any]:
    import torch

    with torch.no_grad():
        if is_hybrid:
            (
                _ids,
                ab_id,
                env_tok,
                env_mask,
                vh_tok,
                vh_mask,
                vl_tok,
                vl_mask,
                _y,
                env_pool,
                vh_pool,
                vl_pool,
            ) = batch
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
                return_cross_attn_weights=True,
                cross_attn_average_heads=True,
            )
        else:
            _ids, ab_id, env_tok, env_mask, vh_tok, vh_mask, vl_tok, vl_mask, _y = batch
            out = model(
                env_tok.to(device, non_blocking=True),
                vh_tok.to(device, non_blocking=True),
                vl_tok.to(device, non_blocking=True),
                env_mask.to(device, non_blocking=True),
                vh_mask.to(device, non_blocking=True),
                vl_mask.to(device, non_blocking=True),
                ab_id.to(device, non_blocking=True),
                return_cross_attn_weights=True,
                cross_attn_average_heads=True,
            )
    return list(out[-1])


def _render_attention_plot(
    attention: np.ndarray,
    *,
    env_len: int,
    vh_len: int,
    vl_len: int,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    query_len = int(vh_len + vl_len)
    env_len = int(env_len)
    attention = np.asarray(attention, dtype=np.float32)[:query_len, :env_len]
    if attention.size == 0:
        raise ValueError("No attention values were available for this sample.")

    env_profile = attention.mean(axis=0)
    env_positions = np.arange(env_len)
    ab_positions = np.arange(query_len)
    chain_labels = np.where(ab_positions < vh_len, "VH", "VL")
    chain_positions = np.where(ab_positions < vh_len, ab_positions + 1, ab_positions - vh_len + 1)
    hover = np.empty((query_len, env_len), dtype=object)
    for i in range(query_len):
        for j in range(env_len):
            hover[i, j] = (
                f"{chain_labels[i]} position {int(chain_positions[i])}<br>"
                f"Env position {j + 1}<br>"
                f"Attention {attention[i, j]:.5f}"
            )

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.24, 0.76],
        vertical_spacing=0.035,
        shared_xaxes=False,
        shared_yaxes=False,
    )
    fig.add_trace(
        go.Scatter(
            x=env_positions,
            y=env_profile,
            mode="lines",
            line={"color": "#1d7282", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(29,114,130,0.18)",
            name="Env attention",
            hovertemplate="Env position %{x}<br>Mean attention %{y:.5f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=env_positions,
            y=ab_positions,
            z=attention,
            customdata=hover,
            colorscale=[
                [0.0, "#f6f8fb"],
                [0.25, "#d6e4ee"],
                [0.5, "#8fb4c7"],
                [0.75, "#3f7f98"],
                [1.0, "#113b55"],
            ],
            colorbar={
                "title": "Cross-attention",
                "len": 0.56,
                "y": 0.38,
                "yanchor": "middle",
                "thickness": 16,
            },
            hovertemplate="%{customdata}<extra></extra>",
            name="Cross-attention",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=vh_len - 0.5, line_color="#ffffff", line_width=1.5, row=2, col=1)
    fig.add_annotation(
        x=0,
        y=max(vh_len / 2 - 0.5, 0),
        text="VH",
        showarrow=False,
        font={"color": "#ffffff", "size": 12},
        xref="x2",
        yref="y2",
        xanchor="left",
    )
    fig.add_annotation(
        x=0,
        y=vh_len + max(vl_len / 2 - 0.5, 0),
        text="VL",
        showarrow=False,
        font={"color": "#ffffff", "size": 12},
        xref="x2",
        yref="y2",
        xanchor="left",
    )
    fig.update_xaxes(title_text="Env token position", range=[0, max(env_len - 1, 1)], row=2, col=1)
    fig.update_yaxes(
        title_text="Antibody token position",
        range=[max(query_len - 0.5, 0.5), -0.5],
        row=2,
        col=1,
    )
    fig.update_xaxes(showticklabels=False, range=[0, max(env_len - 1, 1)], row=1, col=1)
    env_ymax = max(float(np.nanmax(env_profile)) * 1.12, 1e-6)
    fig.update_yaxes(title_text="Env mean", range=[0, env_ymax], row=1, col=1)
    fig.update_layout(
        title={
            "text": (
                "UniCAV interactive cross-attention<br>"
                "<span style='font-size:12px;color:#617083'>Averaged across attention layers and ensemble folds; "
                "antibody VH/VL tokens attend to Env tokens.</span>"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        height=620,
        margin={"l": 72, "r": 96, "t": 78, "b": 58},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        showlegend=False,
    )
    return fig


def _build_attention_plot(dataset_path: Path, checkpoint_root: Path, *, force_cpu: bool):
    import torch
    import infer_pave
    import train_pave as tp

    run_dir = infer_pave._resolve_run_dir(checkpoint_root)
    args, metrics, fold_metrics = infer_pave._load_training_context(run_dir)
    if force_cpu:
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

    train_bundle = infer_pave._build_training_feature_bundle(args, train_data, metrics)
    if not train_bundle["use_token_inputs"]:
        return None
    model_arch = train_bundle["model_arch"]
    is_hybrid = model_arch == "hybrid_cross_attn"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ind_data, _row_stats = infer_pave._prepare_inference_frame(args, dataset_path, limit_rows=1)
    if args.use_antibody_id_feature:
        ind_data, _, _ = tp._attach_antibody_id_feature(ind_data, args, independent=True)
    else:
        ind_data = ind_data.copy()
        ind_data["__ab_id_index"] = 0

    infer_bundle = infer_pave._build_inference_feature_bundle(
        args,
        ind_data,
        f"{args.task}_attention_{dataset_path.stem}",
        train_bundle,
    )
    splits = (
        tp._group_kfold_splits(train_data, args)
        if args.cv_scheme == "group_kfold"
        else [tp._split_data(train_data, args)]
    )

    env_tok = infer_bundle["env_ind_toks"][0]
    vh_tok = infer_bundle["vh_ind_toks"][0]
    vl_tok = infer_bundle["vl_ind_toks"][0]
    env_len = int(env_tok.shape[0])
    vh_len = int(vh_tok.shape[0])
    vl_len = int(vl_tok.shape[0])

    ind_ids = ind_data["__display_id"].to_numpy(dtype=np.int64)
    ind_ab_ids = ind_data["__ab_id_index"].to_numpy(dtype=np.int64)
    y_ind = tp._independent_targets(ind_data, args, splits[0])

    fold_attn: list[np.ndarray] = []
    for fold_index, (split, fold_info) in enumerate(zip(splits, fold_metrics), start=1):
        checkpoint_path = Path(fold_info["checkpoint"])
        if not checkpoint_path.is_file():
            checkpoint_path = run_dir / f"fold_{fold_index}" / "best_model.pt"
        model = infer_pave._load_fold_model(fold_index, split, train_bundle, args, device, checkpoint_path)
        env_ind = infer_bundle["env_ind_base"].astype(np.float32)
        _env_all, env_aux_stats = tp._standardize_env_for_split(
            train_bundle["env_base"],
            train_bundle["env_aux"],
            train_bundle["env_aux_names"],
            split,
            args,
        )
        if infer_bundle["env_aux_ind"] is not None and infer_bundle["env_aux_ind"].shape[1] > 0 and env_aux_stats is not None:
            env_ind = np.concatenate(
                [env_ind, tp._apply_env_aux_stats(infer_bundle["env_aux_ind"], env_aux_stats)],
                axis=1,
            ).astype(np.float32)
        dl = tp._make_token_loader(
            ind_ids,
            ind_ab_ids,
            infer_bundle["env_ind_toks"],
            infer_bundle["vh_ind_toks"],
            infer_bundle["vl_ind_toks"],
            y_ind,
            args,
            env_pool=env_ind if is_hybrid else None,
            vh_pool=infer_bundle["vh_ind"] if is_hybrid else None,
            vl_pool=infer_bundle["vl_ind"] if is_hybrid else None,
            shuffle=False,
        )
        batch = next(iter(dl))
        layers = _forward_cross_attention(model, batch, device, is_hybrid=is_hybrid)
        layer_arrays = []
        for weights in layers:
            arr = weights.detach().cpu().numpy()
            if arr.ndim == 4:
                arr = arr.mean(axis=1)
            layer_arrays.append(arr[0, : vh_len + vl_len, :env_len])
        fold_attn.append(np.mean(layer_arrays, axis=0))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _render_attention_plot(
        np.mean(fold_attn, axis=0),
        env_len=env_len,
        vh_len=vh_len,
        vl_len=vl_len,
    )


def _example_source_root() -> Path:
    local = os.environ.get("PAVE_ARTIFACT_DIR", "").strip()
    if local:
        return Path(local).expanduser().resolve()
    return SPACE_ROOT


def _load_example_record(index: int) -> tuple[str, str, str, str, str, str, str, str]:
    candidates = [
        EXAMPLES_PATH,
        _example_source_root() / "data" / "virus_antibody_classification_dataset.json",
        _example_source_root() / "data" / "no_label" / "ch505_validation_full.json",
    ]
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            break
    if not rows:
        raise gr.Error("No local example dataset found.")

    complete = [
        row
        for row in rows
        if _clean_text(row.get("env_seq")) and _clean_text(row.get("vh_seq")) and _clean_text(row.get("vl_seq"))
    ]
    if len(complete) <= index:
        raise gr.Error(f"Example {index + 1} is unavailable.")
    row = complete[index]
    return (
        _clean_text(row.get("virus") or row.get("Virus") or f"example_virus_{index + 1}"),
        _clean_text(row.get("antibody") or row.get("Antibody") or f"example_antibody_{index + 1}"),
        _clean_text(row.get("lineage") or row.get("ab_lineage")),
        _clean_text(row.get("antibody_full") or row.get("Antibody")),
        _clean_text(row.get("ab_epitope_simple")),
        _clean_text(row.get("env_seq")),
        _clean_text(row.get("vh_seq")),
        _clean_text(row.get("vl_seq")),
    )


def _load_antibody_record(selected: str | None) -> tuple[str, str, str, str, str, str]:
    name = _clean_text(selected)
    record = ANTIBODY_SEQUENCE_MAP.get(name)
    if not record:
        return "", "", "", "", "", ""
    antibody_name = _clean_text(record.get("antibody")) or name
    return (
        antibody_name,
        _clean_text(record.get("ab_lineage")),
        _clean_text(record.get("antibody_full")) or antibody_name,
        _clean_text(record.get("ab_epitope_simple")),
        _clean_text(record.get("vh_seq")),
        _clean_text(record.get("vl_seq")),
    )


def predict_single(
    env_seq: str,
    vh_seq: str,
    vl_seq: str,
    virus: str,
    antibody: str,
    lineage: str,
    antibody_full: str,
    ab_epitope_simple: str,
    force_cpu: bool,
):
    import pandas as pd

    try:
        record = {
            "env_seq": _clean_text(env_seq),
            "vh_seq": _clean_text(vh_seq),
            "vl_seq": _clean_text(vl_seq),
            "virus": _clean_text(virus) or "input_virus",
            "antibody": _clean_text(antibody) or "input_antibody",
            "lineage": _clean_text(lineage),
            "antibody_full": _clean_text(antibody_full),
            "ab_epitope_simple": _clean_text(ab_epitope_simple),
        }
        missing = [key for key in ("env_seq", "vh_seq", "vl_seq") if not record[key]]
        if missing:
            raise ValueError("Missing required sequence field(s): " + ", ".join(missing))

        import infer_pave
        _artifact_root, checkpoint_root = _get_runtime()

        with tempfile.TemporaryDirectory(prefix="pave_space_") as tmp:
            tmpdir = Path(tmp)
            dataset_path = tmpdir / "input.json"
            dataset_path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
            output_root = tmpdir / "outputs"
            manifest = infer_pave.run_inference(
                checkpoint_root,
                [dataset_path],
                output_root=output_root,
                limit_rows=0,
                cpu=bool(force_cpu),
            )
            cohort = manifest["cohorts"][0]
            ensemble_path = Path(cohort["ensemble_prediction_files"]["independent_test_ensemble"])
            frame = pd.read_csv(ensemble_path)
            keep = [
                c
                for c in [
                    "id",
                    "virus",
                    "antibody",
                    "lineage",
                    "antibody_full",
                    "average_y_prob",
                    "average_y_pred_label",
                    "vote_fraction_positive",
                    "vote_y_pred_label",
                ]
                if c in frame.columns
            ]
            result = frame[keep].copy()
            attention_plot = _build_attention_plot(
                dataset_path,
                checkpoint_root,
                force_cpu=bool(force_cpu),
            )
            if "average_y_prob" in result.columns:
                prob = float(result.loc[0, "average_y_prob"])
                label = int(result.loc[0, "average_y_pred_label"])
                summary = (
                    f"Predicted neutralisation probability: {prob:.4f}\n"
                    f"Predicted label: {label} ({'non-neutralising/resistant' if label == 1 else 'neutralising/sensitive'})"
                )
            else:
                summary = "Prediction completed."
            return summary, result, attention_plot
    except Exception:
        details = traceback.format_exc()
        print(details, flush=True)
        return (
            "Prediction failed. Full traceback:\n\n" + details,
            pd.DataFrame(),
            None,
        )


UNICAV_CSS = """
:root {
  --unicav-ink: #18212f;
  --unicav-muted: #617083;
  --unicav-line: #d9e2ec;
  --unicav-surface: #ffffff;
  --unicav-soft: #f4f7fa;
  --unicav-signal: #1d7282;
  --unicav-accent: #b84c45;
}
.gradio-container {
  width: min(1180px, calc(100vw - 48px)) !important;
  max-width: 1180px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-left: 24px !important;
  padding-right: 24px !important;
  color: var(--unicav-ink);
}
body {
  background: var(--unicav-soft) !important;
}
.unicav-hero {
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 24px 0 12px;
  border-bottom: 1px solid var(--unicav-line);
  margin-bottom: 18px;
}
.unicav-logo {
  width: 78px;
  height: 78px;
  object-fit: contain;
  flex: 0 0 auto;
}
.unicav-title-stack {
  min-width: 0;
}
.unicav-brand {
  font-size: 34px;
  line-height: 1.1;
  font-weight: 760;
  letter-spacing: 0;
  margin: 0 0 8px;
}
.unicav-subtitle {
  max-width: 840px;
  color: var(--unicav-muted);
  font-size: 15px;
  line-height: 1.55;
  margin: 0;
}
.unicav-section h3 {
  font-size: 15px;
  margin: 18px 0 8px;
  color: var(--unicav-ink);
}
.unicav-note {
  color: var(--unicav-muted);
  font-size: 13px;
  line-height: 1.45;
  margin: 0 0 8px;
}
button.primary {
  background: var(--unicav-signal) !important;
  border-color: var(--unicav-signal) !important;
}
@media (max-width: 640px) {
  .gradio-container {
    width: min(100vw, calc(100vw - 24px)) !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
  }
  .unicav-hero {
    gap: 12px;
    align-items: flex-start;
  }
  .unicav-logo {
    width: 58px;
    height: 58px;
  }
  .unicav-brand {
    font-size: 28px;
  }
}
"""


with gr.Blocks(title="UniCAV") as demo:
    gr.HTML(
        f"""
        <div class="unicav-hero">
          <img class="unicav-logo" src="{_logo_data_uri()}" alt="UniCAV logo">
          <div class="unicav-title-stack">
            <h1 class="unicav-brand">UniCAV</h1>
            <p class="unicav-subtitle">
              HIV-1 Env and antibody VH/VL neutralisation predictor. Enter one Env-antibody pair,
              optionally add sample metadata, then run the 5-fold ensemble prediction.
            </p>
          </div>
        </div>
        """
    )
    gr.Markdown("### Sample Metadata", elem_classes=["unicav-section"])
    gr.HTML('<p class="unicav-note">Optional fields. They help identify the result row but are not required for prediction.</p>')
    with gr.Row():
        virus = gr.Textbox(label="Virus / sample name", placeholder="input_virus", scale=1)
        antibody = gr.Textbox(label="Antibody", placeholder="input_antibody", scale=1)
    with gr.Row():
        lineage = gr.Textbox(label="Lineage", placeholder="optional", scale=1)
        antibody_full = gr.Textbox(label="Antibody full name", placeholder="optional", scale=1)
        ab_epitope_simple = gr.Textbox(
            label="Epitope",
            placeholder="CD4bs / V3 / V1V2 / MPER / fusion_peptide",
            scale=1,
        )

    gr.Markdown("### Sequence Input", elem_classes=["unicav-section"])
    gr.HTML('<p class="unicav-note">Required fields. Paste amino-acid sequences only; spaces, line breaks and stop symbols are cleaned before inference.</p>')
    antibody_preset = gr.Dropdown(
        label="Antibody sequence preset",
        choices=ANTIBODY_CHOICES,
        value=None,
        filterable=True,
        allow_custom_value=False,
        info="Only antibodies with a unique Antibody-to-VH/VL mapping are listed.",
    )
    env_seq = gr.Textbox(
        label="Env sequence",
        lines=8,
        placeholder="Paste HIV Env amino-acid sequence",
    )
    with gr.Row():
        vh_seq = gr.Textbox(
            label="VH sequence",
            lines=5,
            placeholder="Paste antibody heavy-chain VH amino-acid sequence",
        )
        vl_seq = gr.Textbox(
            label="VL sequence",
            lines=5,
            placeholder="Paste antibody light-chain VL amino-acid sequence",
        )
    with gr.Row():
        ex1 = gr.Button("Example 1", variant="secondary")
        ex2 = gr.Button("Example 2", variant="secondary")
        ex3 = gr.Button("Example 3", variant="secondary")

    gr.Markdown("### Prediction", elem_classes=["unicav-section"])
    with gr.Row():
        cpu = gr.Checkbox(value=False, label="Force CPU")
        run = gr.Button("Predict Neutralisation", variant="primary")
    summary = gr.Textbox(label="Prediction Summary", lines=12)
    table = gr.Dataframe(label="Prediction Details")
    attention_plot = gr.Plot(
        label="Attention Mechanism",
    )
    example_outputs = [virus, antibody, lineage, antibody_full, ab_epitope_simple, env_seq, vh_seq, vl_seq]
    ex1.click(lambda: _load_example_record(0), inputs=[], outputs=example_outputs)
    ex2.click(lambda: _load_example_record(1), inputs=[], outputs=example_outputs)
    ex3.click(lambda: _load_example_record(2), inputs=[], outputs=example_outputs)
    antibody_preset.change(
        _load_antibody_record,
        inputs=[antibody_preset],
        outputs=[antibody, lineage, antibody_full, ab_epitope_simple, vh_seq, vl_seq],
    )
    run.click(
        predict_single,
        inputs=[env_seq, vh_seq, vl_seq, virus, antibody, lineage, antibody_full, ab_epitope_simple, cpu],
        outputs=[summary, table, attention_plot],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"},
        css=UNICAV_CSS,
        show_error=True,
    )

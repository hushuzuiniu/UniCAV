# UniCAV Hugging Face Space

This document describes the Gradio Space wrapper for the UniCAV
`focal_alpha0p75/20260520-071018_classification_PAVE` 5-fold ensemble.

If deploying this repository as a Hugging Face Space, copy the following metadata
into the Space repository `README.md` front matter:

```yaml
---
title: UniCAV
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
---
```

## Recommended Deployment Layout

Use two Hugging Face repos:

1. A **Space repo** containing this `app.py`, `requirements.txt`, and the project
   Python source files needed by inference (`train_pave.py`, `infer_pave.py`,
   minimal `models/`).
2. A private **Model repo** named `<your-hf-username>/UniCAV-artifacts` containing only the
   runtime artifacts:

```text
data/virus_antibody_classification_dataset.json
data/Env_HXB2_feature.csv
data/ch505_aligned_to_training.fasta
results/pave/focal_alpha_sweep_hybrid/focal_alpha0p75/20260520-071018_classification_PAVE/metrics.json
results/pave/focal_alpha_sweep_hybrid/focal_alpha0p75/20260520-071018_classification_PAVE/fold_1/best_model.pt
results/pave/focal_alpha_sweep_hybrid/focal_alpha0p75/20260520-071018_classification_PAVE/fold_2/best_model.pt
results/pave/focal_alpha_sweep_hybrid/focal_alpha0p75/20260520-071018_classification_PAVE/fold_3/best_model.pt
results/pave/focal_alpha_sweep_hybrid/focal_alpha0p75/20260520-071018_classification_PAVE/fold_4/best_model.pt
results/pave/focal_alpha_sweep_hybrid/focal_alpha0p75/20260520-071018_classification_PAVE/fold_5/best_model.pt
results/pave/embedding_cache/classification_train_env_seq_ESMC-600M_bf6c195abcce91fce70e.pt
results/pave/embedding_cache/classification_train_vh_seq_Antiberta2-H_d873654efd8cc9349e5b.pt
results/pave/embedding_cache/classification_train_vl_seq_Antiberta2-L_d82a9965bf06f3222c07.pt
results/pave/embedding_cache/classification_train_epitope_token_ESMC-600M_6e76c151a32e9d09e8f3.pt
model_weight/esmc-600m-2024-12/
model_weight/antiberta2/
```

Do not upload `results/pave/embedding_cache/*_tokens_*_shards/` for this Space.
UniCAV computes token embeddings for the single input at inference time and only
uses the training pooled/epitope caches listed above.

Set this Space variable:

```text
PAVE_ARTIFACT_REPO=<your-hf-username>/UniCAV-artifacts
PAVE_ARTIFACT_REPO_TYPE=model
```

If the artifact repo is private, add a Space secret:

```text
HF_TOKEN=<read token with access to the artifact repo>
```

The three example buttons are bundled in this Space repo and do not require the
artifact repo. Prediction does require the artifact repo above.

## Input Format

The Space accepts one Env-antibody pair through direct text inputs. Required
fields:

```text
Env sequence
VH sequence
VL sequence
```

Optional metadata fields:

```text
Virus / sample name
Antibody
Lineage
Antibody full name
Epitope
```

The output table includes:

```text
average_y_prob
average_y_pred_label
vote_fraction_positive
vote_y_pred_label
```

`average_y_pred_label = 1` means predicted non-neutralising/resistant.

## Hardware Note

This model uses ESMC-600M + Antiberta2 + a hybrid cross-attention head. A free CPU
Space will be very slow and may not have enough disk/RAM for the backbone and
token caches. Use a GPU Space with persistent storage, or export a slimmer
inference bundle before public deployment.

## Local Smoke Test

From the project root:

```bash
PAVE_ARTIFACT_DIR=/path/to/artifact/root \
python app.py
```

Then paste one Env sequence, one VH sequence, and one VL sequence into the
interface, or click one of the three example buttons.

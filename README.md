# UniCAV

Code, source data, and supplementary tables for:

**Paired Env-antibody representation learning predicts HIV-1 neutralization and
traces antibody escape across viral evolution**

UniCAV predicts HIV-1 neutralization outcomes for paired Env-antibody sequences.
The model combines Env representations from ESM-C, antibody VH/VL
representations from AntiBERTa2, and a hybrid pooled-feature plus token
cross-attention architecture. The repository also includes the source data and
plotting scripts used to reproduce the manuscript figures and supplementary
figures.

## Repository Layout

- `app.py` - Gradio demo entry point.
- `train_pave.py` - UniCAV model training utilities and architecture code.
- `infer_pave.py` - 5-fold checkpoint ensemble inference on JSON cohorts.
- `models/` - protein language model wrappers and base model definitions.
- `scripts/` - embedding helper scripts.
- `requirements.txt` - Python runtime dependencies for the demo/model package.
- `HUGGINGFACE_SPACE.md` - deployment notes for the online Hugging Face Space.
- `figures/` - manuscript and supplementary figure plotting scripts.
- `figures/source_data/` - Excel/CSV/TSV/tree source data used by the figure
  scripts.
- `figures/*_panels/` - generated figure panel PDFs.
- `tables/` - supplementary tables distributed with the manuscript.

## Demo And Model Artifacts

- Interactive demo: <https://huggingface.co/spaces/<your-hf-username>/UniCAV>
- Runtime artifacts are expected from the Hugging Face model repository
  `<your-hf-username>/UniCAV-artifacts`.
- The demo loads artifacts through:

```text
PAVE_ARTIFACT_REPO=<your-hf-username>/UniCAV-artifacts
PAVE_ARTIFACT_REPO_TYPE=model
```

See `HUGGINGFACE_SPACE.md` for the exact Hugging Face Space deployment layout and
artifact file list.

## Environment

The demo/model package is described in `requirements.txt`. Python 3.10+ is
recommended.

Install the demo dependencies with:

```bash
pip install -r requirements.txt
```

Run the local demo with:

```bash
PAVE_ARTIFACT_DIR=/path/to/artifact/root python app.py
```

For figure regeneration, install the Python/R packages required by the
individual plotting scripts and run commands from the repository root, for
example:

```bash
python figures/Figure2.py
python figures/Figure3.py
Rscript figures/Figure4.R
python figures/Figure4.py
python figures/Figure5.py
python figures/Figure6.py
python figures/FigureS7.py
```

Some figures are split across Python and R scripts. Generated panel PDFs are
written under the corresponding `figures/Figure*_panels/` or
`figures/FigureS*_panels/` directory.

## Data Availability

- Neutralization measurements were derived from CATNAP:
  <https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/>
- Figure-level source data are provided in `figures/source_data/`.
- Supplementary tables are provided in `tables/`.
- Large model weights and embedding caches are not stored in this GitHub
  repository; they are hosted separately as model/runtime artifacts.
- Processed datasets and archival release DOI: to be added.

## Citation

Citation details will be added after manuscript acceptance or public preprint
release.

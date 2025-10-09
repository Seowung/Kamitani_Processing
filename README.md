# Kamitani Processing

This repository contains scripts and notebooks for reproducing parts of the Kamitani et al. emotion video fMRI processing pipeline.  The code assumes your raw and derivative data live in a [BIDS](https://bids.neuroimaging.io/) hierarchy and focuses on three recurring tasks:

* first-level GLM estimation of block-wise responses with nilearn,
* whole-brain searchlight multivariate pattern analysis (MVPA), and
* exporting block-averaged response matrices for downstream visualization in Pycortex.

Although the examples were written against the University of Florida cluster paths used by the original author, the utilities are self-contained—adjust a few input paths and the scripts will work with any compatible BIDS dataset.

## Repository layout

```
.
├── first-level-analysis.py       # Batch first-level GLM estimation and report generation.
├── searchlight_mvpa_new.py       # Command line searchlight MVPA driver.
├── utils/
│   ├── export_processed_and_visualize.py  # Turn saved block averages into Pycortex flatmaps.
│   ├── process_and_cortex_helpers.py      # Shared helpers for saving/plotting processed data.
│   └── run_processing_and_save_template.py# Template for integrating your processing pipeline.
├── Kamitani_Analysis.ipynb       # Exploration notebook for GLM and decoding outputs.
├── GLM_tutorial.ipynb            # Step-by-step walkthrough of the GLM workflow.
├── Visualize_fMRI.ipynb          # Quick visual QC recipes.
└── Visualize_Seachlight.ipynb    # Inspect searchlight accuracy maps.
```

The notebooks mirror the scripts and are useful for interactively validating the pipeline before automating it on a cluster.

## Python environment

The scripts require Python 3.9+ and the following packages:

* `numpy`, `pandas`
* `nibabel`, `nilearn`, `pybids`
* `scikit-learn`, `scipy`
* `matplotlib`
* `pycortex` (only for the optional visualization utilities)

Install them with your preferred environment manager, for example:

```bash
conda create -n kamitani python=3.10
conda activate kamitani
pip install nilearn pybids pycortex scikit-learn matplotlib nibabel pandas scipy
```

## Data expectations

The batch scripts assume the following layout:

* `bids_root/` holds your raw dataset (`sub-*/ses-*/*`).
* `bids_root/derivatives/fmriprep/` contains preprocessed functional runs and anatomical derivatives (e.g., aparcaseg segmentations) produced by [fMRIPrep](https://fmriprep.org/).
* Task events live alongside the raw data as `sub-*_task-*_events.tsv` (see `layout.get(... suffix="events2")` in `first-level-analysis.py` if you use a different suffix).

The paths that start with `/orange/ruogu.fang/...` in the scripts are example cluster locations—replace them with the root of your dataset.

## Running the first-level GLM

`first-level-analysis.py` orchestrates the standard nilearn first-level workflow:

1. `collect_subject` queries BIDS for preprocessed BOLD runs, masks, confounds, and events.
2. `get_confounds` loads an aggressive nuisance regressor set (motion, WM/CSF, high-pass, scrubbing).
3. `fit_first_level` fits an SPM-style HRF GLM for each subject and generates a nilearn HTML report.
4. `save_contrast` stores effect, variance, and z-statistics for a list of predefined contrasts.

Update the `bids_root`, `subjects`, and `contrasts` definitions to match your dataset, then run:

```bash
python first-level-analysis.py
```

The script saves per-subject reports and NIfTI maps under `derivatives/nilearn/`.

## Searchlight MVPA

`searchlight_mvpa_new.py` implements a leave-run-out searchlight classifier comparing two emotion conditions.  The driver:

* builds trial-wise beta images using a least-squares-all (LSA) design per run,
* optionally constructs a visual cortex mask from fMRIPrep aparcaseg outputs,
* performs spherical searchlight decoding with either stratified K-fold or leave-one-run-out cross-validation, and
* writes balanced accuracy maps to disk.

Example invocation:

```bash
python searchlight_mvpa_new.py \
  --bids_root /path/to/Kamitani \
  --sub 01 \
  --task CowenKeltnerEmotionMovie \
  --cond_pos unpleasant --cond_neg neutral \
  --radius 5.0 \
  --use_aparcaseg_visual \
  --aparcaseg_dseg /path/to/sub-01_desc-aparcaseg_dseg.nii.gz \
  --aparcaseg_tsv /path/to/desc-aparcaseg_dseg.tsv \
  --continuous --close_mm 2 --bridge_mm 3 \
  --out_nii out/sub-01_unpleasant_vs_neutral.nii.gz
```

Omit the aparcaseg flags to run a whole-brain searchlight or provide a custom ROI mask with `--mask_img`.

## Exporting block-averaged responses

The utilities in `utils/` help you persist block-averaged response matrices and render them with Pycortex:

* `process_and_cortex_helpers.save_processed_outputs` saves the samples matrix, stimulus metadata, affine, and voxel indices used by a fitted `NiftiMasker`.
* `export_processed_and_visualize.py` recreates a Pycortex `Volume` from those files and optionally generates a flatmap PNG.
* `run_processing_and_save_template.py` shows how to hook your own `process_subject` function into the helper for batch export.

After running your own processing code to obtain `S_z`, `ids`, and a fitted masker:

```bash
python utils/run_processing_and_save_template.py
```

Then create a flatmap for a particular sample with:

```bash
python utils/export_processed_and_visualize.py \
  --samples_npy ./outputs/processed/Subject1/samples_z.npy \
  --ids_csv ./outputs/processed/Subject1/sample_ids.csv \
  --voxel_ijk_npy ./outputs/processed/Subject1/voxel_ijk.npy \
  --affine_npy ./outputs/processed/Subject1/affine.npy \
  --mask_shape_npy ./outputs/processed/Subject1/mask_shape.npy \
  --subject Subject1 --xfm voxel2mm_fmriprep \
  --sample_index 0 \
  --out_png ./outputs/processed/Subject1/sample0.png
```

Add `--overlay_svg` if you want to include an anatomical outline from your Pycortex database.

## Notebooks

The Jupyter notebooks replicate the scripted workflows with richer visualization:

* **GLM_tutorial.ipynb** – a guided first-level GLM tutorial (design matrices, confounds, contrasts).
* **Visualize_fMRI.ipynb** – quick plots for inspecting individual runs and GLM results.
* **Visualize_Seachlight.ipynb** – interactive review of searchlight accuracy volumes.
* **Kamitani_Analysis.ipynb** – extended analyses that combine GLM and MVPA outputs.

Open them in JupyterLab or VS Code after configuring your environment.

## Contributing

Issues and pull requests are welcome.  When adapting the scripts to new datasets, please keep function signatures stable so that the notebooks remain synchronized with the command-line entry points.


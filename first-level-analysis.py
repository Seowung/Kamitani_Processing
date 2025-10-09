import os
import pandas as pd
from pathlib import Path
from bids import BIDSLayout
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.masking import intersect_masks
from nilearn.image import load_img
from nilearn.glm.first_level import FirstLevelModel
import nibabel as nib

import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import make_second_level_design_matrix  # convenience
from nilearn import plotting
from nilearn.image import load_img, resample_to_img, math_img

from nilearn.plotting import plot_design_matrix, plot_design_matrix_correlation
import matplotlib
matplotlib.use("Agg")  # headless-safe on clusters


bids_root = Path("/orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/")  # dataset root (contains sub-XX/, sub-XX/ses-YY/, etc.)
deriv = os.path.join(bids_root, "derivatives")

# Allow derivatives so layout can see fMRIPrep outputs
layout = BIDSLayout(bids_root, validate=False, derivatives=True)


def collect_subject(subject):
    # All preproc bolds in standard space (adjust entities to match your outputs)
    imgs = layout.get(subject=subject, datatype="func", suffix="bold", space="MNI152NLin2009cAsym", desc="preproc",
                      extension=[".nii", ".nii.gz"], return_type="file")

    # One mask per run
    masks = layout.get(subject=subject, datatype="func", suffix="mask", space="MNI152NLin2009cAsym", desc="brain",
                       extension=[".nii", ".nii.gz"], return_type="file")

    # Confounds tsv per run
    confs = layout.get(subject=subject, suffix="timeseries",
                       desc="confounds", extension=".tsv",
                       return_type="file")

    # Events per run (from the raw BIDS side, not derivatives)
    events = layout.get(subject=subject, suffix="events2",
                        extension=".tsv", return_type="file")

    # TR from the first BOLD’s JSON
    meta = layout.get_metadata(imgs[0])
    tr = float(meta.get("RepetitionTime"))

    print("## Data Import Complete ##", flush=True)
    return imgs, masks, confs, events, tr


def get_confounds(img_files):
    # Strategy "simple" is a common starting point; "scrubbing" adds FD/DVARS censoring
    conf_list, sample_masks = load_confounds(
        img_files,
        motion='basic', 
        strategy=["motion", "high_pass", "wm_csf"],  # add "global_signal" if you intend GSR
        scrub=True, fd_threshold=0.5, std_dvars_threshold=1.5,
        demean=True
    )
    return conf_list, sample_masks


def subject_mask(mask_files):
    return intersect_masks(mask_files, threshold=1.0, connected=False)  # strict intersection


def load_events(event_files):
    return [pd.read_csv(ef, sep="\t") for ef in event_files]


def fit_first_level(imgs, events, confounds, sample_masks, mask_img, tr):
    fm = FirstLevelModel(
        t_r=tr,
        mask_img=mask_img,
        hrf_model="spm",  # SPM HRF (common default)
        drift_model="cosine", high_pass=0.008,  # ~128 s cutoff
        smoothing_fwhm=None,  # optional; fMRIPrep doesn’t smooth
        noise_model="ar1",  # AR(1) is typical for fMRI
        standardize='zscore',
        minimize_memory=True, verbose=1, n_jobs=2
    )
    fm.fit(imgs, events=events, confounds=confounds)
    return fm


def save_contrast(fm, contrast, out_dir, label):
    out_dir = Path(out_dir);
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ask for both effect and variance; this is the fixed-effects result across runs
    out = fm.compute_contrast(contrast, output_type="all")

    eff = out["effect_size"]
    var = out["effect_variance"]
    z = out["z_score"]  # handy for quick QA

    eff_p = out_dir / f"{label}_effect.nii.gz"
    var_p = out_dir / f"{label}_variance.nii.gz"
    z_p = out_dir / f"{label}_zmap.nii.gz"

    nib.save(eff, eff_p);
    nib.save(var, var_p);
    nib.save(z, z_p)
    return eff_p, var_p, z_p


subjects = ["01", "02", "03", "04", "05"]

contrasts = {
    "rest>neutral": "rest - neutral",
    "rest>pleasant": "rest - pleasant",
    "rest>unpleasant": "rest - unpleasant",

    "neutral>rest": "neutral - rest",
    "neutral>pleasant": "neutral - pleasant",
    "neutral>unpleasant": "neutral - unpleasant",

    "pleasant>rest": "pleasant - rest",
    "pleasant>neutral": "pleasant - neutral",
    "pleasant>unpleasant": "pleasant - unpleasant",

    "unpleasant>rest": "unpleasant - rest",
    "unpleasant>neutral": "unpleasant - neutral",
    "unpleasant>pleasant": "unpleasant - pleasant",
}


eff_map_per_sub = {}
effect_paths = {name: {} for name in contrasts}


for sub in subjects:
    imgs, masks, confs, events, tr = collect_subject(sub)
    conf_list, sample_masks = get_confounds(imgs)
    mask_img = subject_mask(masks)
    events_list = load_events(events)

    sample_masks = None

    print("## Fitting the first level analysis for subject-{}".format(sub), flush=True)
    fm = fit_first_level(imgs, events_list, conf_list, sample_masks, mask_img, tr)

    report = fm.generate_report(
        contrasts=contrasts,              # one or many
        title="sub-{} first-level GLM".format(sub),   # anything you like
        bg_img="MNI152TEMPLATE",          # or a NIfTI path
        alpha=0.05,                 # overall FWER level
        height_control="bonferroni",# voxelwise Bonferroni correction
        cluster_threshold=0,       # optional: remove clusters < 15 vox
        two_sided=False,
        plot_type="glass",                # "slice" or "glass"
        report_dims=(1600, 800),
)
    outroot = "/orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/derivatives/nilearn/no_smooth"
    sub_out = os.path.join(outroot, f"sub-{sub}")
    outdir = Path(sub_out)
    outdir.mkdir(parents=True, exist_ok=True)

    # In notebooks, just display(report)
    # To save a standalone HTML file:
    report.save_as_html(os.path.join(sub_out, "firstlevel_report.html"))
    # Or open directly in a browser:
    # report.open_in_browser()

    dm_dir = os.path.join(sub_out, "design_matrices")
    os.makedirs(dm_dir, exist_ok=True)

    # fm.design_matrices_ is a list: one pandas DataFrame per run
    for r_ix, dm in enumerate(fm.design_matrices_):
        dm_png = os.path.join(dm_dir, f"design_run-{r_ix + 1:02d}.png")
        corr_png = os.path.join(dm_dir, f"design_corr_run-{r_ix + 1:02d}.png")

        # Design matrix image
        plot_design_matrix(dm, output_file=dm_png)  # writes and closes the figure

        # Correlation heatmap between regressors (drifts/constant omitted)
        plot_design_matrix_correlation(dm, output_file=corr_png)  # writes and closes

    for label, expr in contrasts.items():
        # Ask for both effect and variance if you like; effect is sufficient for SecondLevelModel
        out = fm.compute_contrast(expr, output_type="all")  # effect_size, variance, z_score
        eff = out["effect_size"]  # fixed-effects effect across runs
        z = out["z_score"]  # optional QA

        eff_p = os.path.join(sub_out, f"{label}_effect.nii.gz")
        z_p = os.path.join(sub_out, f"{label}_zmap.nii.gz")
        nib.save(eff, eff_p)
        nib.save(z, z_p)

        effect_paths[label][sub] = str(
            eff_p)  # formatted like  {'rest>neutral': {'01': '/orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/derivatives/nilearn/sub-01/rest>neutral_effect.nii.gz'}}
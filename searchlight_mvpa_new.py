#!/usr/bin/env python
# coding: utf-8

'''
python searchlight_mvpa_new.py \
  --bids_root /orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/ \
  --sub 01 \
  --task CowenKeltnerEmotionMovie \
  --space MNI152NLin2009cAsym \
  --radius 10.0 \
  --cond_pos unpleasant \
  --cond_neg neutral \
  --use_aparcaseg_visual \
  --aparcaseg_dseg /orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/derivatives/sub-01/ses-anatomy/anat/sub-01_ses-anatomy_desc-aparcaseg_dseg.nii.gz \
  --aparcaseg_tsv /orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/derivatives/desc-aparcaseg_dseg.tsv \
  --visual_labels occipital,pericalcarine,cuneus,lingual,lateraloccipital,G_occipital_middle,G_occipital_superior,G_and_S_occipital_inferior,S_calcarine,S_occipital_anterior,S_occipital_middle_and_Lunatus,S_occipital_superior_and_transversalis,S_parieto_occipital,S_occipito-temporal_lateral,S_occipito-temporal_medial,G_occipit-temp_lat,G_occipit-temp_med,Pole_occipital,WhiteMatter_occipital,WhiteMatter_pericalcarine,WhiteMatter_cuneus,WhiteMatter_lingual,WhiteMatter_lateraloccipital,WhiteMatter_occipitotemporal,ctx-lh-occipital,ctx-rh-occipital \
  --continuous --close_mm 2 --bridge_mm 3 \
  --out_nii out_5mm/sub-01_visual_searchlight_unpleasant_vs_neutral.nii.gz \
  --drop_unbalanced_runs
'''


#!/usr/bin/env python
# coding: utf-8
"""Utility script for running a searchlight MVPA with leave-run-out cross-validation."""

import os
import argparse
import numpy as np
import pandas as pd

from nilearn.glm.first_level import first_level_from_bids, FirstLevelModel
from nilearn.image import concat_imgs
from nilearn.decoding import SearchLight
from nilearn.masking import compute_brain_mask
from nilearn.image import load_img, resample_to_img, math_img, new_img_like

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score
from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation, label as cc_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Searchlight MVPA (positive vs negative) using trialwise betas (LSA)."
    )
    parser.add_argument("--bids_root", type=str, required=True,
                        help="Root path of BIDS + derivatives")
    parser.add_argument("--sub", type=str, required=True,
                        help="Subject ID (without sub- prefix)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task label (or leave None for all tasks)")
    parser.add_argument("--space", type=str, default="MNI152NLin2009cAsym",
                        help="Spatial normalization / output space label")
    parser.add_argument("--radius", type=float, default=5.0,
                        help="Searchlight radius (in mm)")
    parser.add_argument("--cond_pos", type=str, required=True,
                        help="Name of the positive condition (e.g. unpleasant)")
    parser.add_argument("--cond_neg", type=str, required=True,
                        help="Name of the negative condition (e.g. neutral)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of folds for CV (if using KFold/StratifiedKFold)")
    parser.add_argument("--use_group_cv", action="store_true",
                        help="If set, use LeaveOneGroupOut CV by run (needs run labels)")
    parser.add_argument("--out_nii", type=str, default=None,
                        help="Output path for searchlight accuracy NIfTI (.nii.gz)")
    parser.add_argument("--mask_img", type=str, default=None,
                        help="Path to a binary ROI mask NIfTI in same space as betas (preferred).")
    parser.add_argument("--use_aparcaseg_visual", action="store_true",
                        help="If set, build a visual-cortex mask from fMRIPrep aparcaseg.")
    parser.add_argument("--aparcaseg_dseg", type=str, default=None,
                        help="Path to sub-*_desc-aparcaseg_dseg.nii.gz (T1w space).")
    parser.add_argument("--aparcaseg_tsv", type=str, default=None,
                        help="Path to sub-*_aparcaseg.tsv (label table).")
    parser.add_argument("--visual_labels", type=str,
                        default="pericalcarine,cuneus,lingual,lateraloccipital",
                        help="Comma-separated label substrings to include from aparcaseg TSV.")
    parser.add_argument("--continuous", action="store_true",
                        help="Make the visual ROI a single continuous parcel by closing/bridging.")
    parser.add_argument("--close_mm", type=float, default=2.0,
                        help="Structuring element radius (mm) for binary closing (seal sulci).")
    parser.add_argument("--bridge_mm", type=float, default=3.0,
                        help="Structuring element radius (mm) for iterative dilation to connect islands.")
    parser.add_argument("--drop_unbalanced_runs", action="store_true",
                        help=(
                            "Skip runs that do not contain at least one trial of both requested "
                            "conditions. Useful when debugging severe run-to-run imbalance."
                        ))
    return parser.parse_args()


def compute_lsa_run_betas(run_imgs, run_events, glm_params):
    """Fit trialwise GLM for a single run; return dict trial_label → beta Niimg."""
    ev2 = run_events.copy()
    counts = {}
    for ix, row in ev2.iterrows():
        cond = row["trial_type"]
        counts.setdefault(cond, 0)
        counts[cond] += 1
        new_label = f"{cond}__{counts[cond]:03d}"
        ev2.at[ix, "trial_type"] = new_label

    flm = FirstLevelModel(**glm_params)
    flm = flm.fit(run_imgs, ev2)

    betas = {}
    dm = flm.design_matrices_[0]
    for col in dm.columns:
        if "__" not in col:
            continue
        try:
            beta_img = flm.compute_contrast(col, output_type="effect_size")
            betas[col] = beta_img
        except Exception as exc:  # pragma: no cover - nilearn exceptions vary
            print(f"[WARN] Could not compute contrast for {col}: {exc}")
    return betas


def bal_acc_scorer(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def _sphere_struct(radius_vox):
    """Create a spherical structuring element as a boolean ndarray."""
    r = max(1, int(round(radius_vox)))
    grid = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    mask = sum(g ** 2 for g in grid) <= (r ** 2)
    return mask


def make_continuous_mask(mask_img, close_mm=2.0, bridge_mm=3.0, max_iter=20):
    """Make a binary mask continuous using closing, hole-fill, and iterative dilation bridges."""
    data = mask_img.get_fdata() > 0
    vx = np.sqrt((mask_img.affine[:3, :3] ** 2).sum(0))
    close_r = np.maximum(1, np.round(close_mm / vx).astype(int)).max()
    bridge_r = np.maximum(1, np.round(bridge_mm / vx).astype(int)).max()

    data = binary_closing(data, structure=_sphere_struct(close_r), iterations=1)
    data = binary_fill_holes(data)

    struct = _sphere_struct(bridge_r)
    iterations = 0
    while True:
        labeled, ncc = cc_label(data)
        if ncc <= 1 or iterations >= max_iter:
            break
        data = binary_dilation(data, structure=struct)
        iterations += 1

    return new_img_like(mask_img, data.astype("uint8"))


def build_visual_mask_from_aparcaseg(X4d_img, dseg_path, tsv_path, visual_substrings):
    """Build a binary visual cortex mask from aparcaseg outputs, resampled to functional space."""
    if dseg_path is None or tsv_path is None:
        raise ValueError("Need both aparcaseg paths to build aparcaseg-based mask.")

    dseg = load_img(dseg_path)
    table = pd.read_csv(tsv_path, sep='\t')

    if not {"index", "name"}.issubset(set(table.columns)):
        raise ValueError("Aparcaseg TSV must contain 'index' and 'name' columns.")

    mask_indices = set()
    for _, row in table.iterrows():
        name = str(row["name"]).lower()
        if any(sub in name for sub in visual_substrings):
            mask_indices.add(int(row["index"]))

    if not mask_indices:
        raise ValueError("No labels matched the requested visual substrings.")

    expr = " + ".join([f"(img == {idx})" for idx in sorted(mask_indices)]) or "img*0"
    aparc_vis_mask_native = math_img(expr, img=dseg)

    aparc_vis_mask_in_func = resample_to_img(
        aparc_vis_mask_native, X4d_img, interpolation="nearest"
    )
    aparc_vis_mask_in_func = math_img("img > 0", img=aparc_vis_mask_in_func)
    aparc_vis_mask_in_func = new_img_like(
        aparc_vis_mask_in_func,
        (aparc_vis_mask_in_func.get_fdata() > 0).astype("uint8")
    )

    return aparc_vis_mask_in_func


def run_searchlight_binary(X4d, y, groups, radius_mm, mask_img, cv, out_nii_path):
    if mask_img is None:
        mask_img = compute_brain_mask(X4d)

    sl = SearchLight(
        mask_img=mask_img,
        radius=radius_mm,
        estimator=SVC(class_weight='balanced'),
        n_jobs=-1,
        scoring=bal_acc_scorer,
        cv=cv,
        verbose=1,
    )

    if isinstance(cv, LeaveOneGroupOut):
        sl.fit(X4d, y=y, groups=groups)
    else:
        sl.fit(X4d, y=y)

    scores_img = sl.scores_img_
    if out_nii_path:
        scores_img.to_filename(out_nii_path)
    return scores_img


def main():
    args = parse_args()

    if args.out_nii is None:
        args.out_nii = (
            f"sub-{args.sub}_task-{args.task or 'all'}_searchlight_"
            f"{args.cond_pos}_vs_{args.cond_neg}.nii.gz"
        )
    out_dir = os.path.dirname(args.out_nii)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    models, models_run_imgs, events_dfs, _ = first_level_from_bids(
        dataset_path=args.bids_root,
        task_label=args.task,
        space_label=args.space,
        sub_labels=[args.sub],
        t_r=2,
        img_filters=[("desc", "preproc")],
        confounds_strategy=["motion", "wm_csf"],
        confounds_motion="basic",
        confounds_wm_csf="basic",
        hrf_model="spm",
        drift_model="cosine",
        signal_scaling=False,
    )

    run_imgs_list = models_run_imgs[0]
    run_events_list = events_dfs[0]

    glm_params = models[0].get_params()
    glm_params["signal_scaling"] = models[0].signal_scaling

    betas_imgs = []
    labels = []
    group_labels = []
    per_run_counts = []

    for run_i, (rimgs, revts) in enumerate(zip(run_imgs_list, run_events_list)):
        print(f"LSA run {run_i}")
        betas_run = compute_lsa_run_betas(rimgs, revts, glm_params)

        run_beta_imgs = []
        run_label_vals = []
        run_counts = {args.cond_pos: 0, args.cond_neg: 0}

        for label, img in betas_run.items():
            base = label.split("__")[0]
            if base == args.cond_pos:
                run_beta_imgs.append(img)
                run_label_vals.append(1)
                run_counts[args.cond_pos] += 1
            elif base == args.cond_neg:
                run_beta_imgs.append(img)
                run_label_vals.append(0)
                run_counts[args.cond_neg] += 1

        per_run_counts.append((run_i, run_counts))

        if not run_beta_imgs:
            print(f"  [WARN] No beta maps for requested conditions in run {run_i}; skipping.")
            continue

        if args.drop_unbalanced_runs and (run_counts[args.cond_pos] == 0 or run_counts[args.cond_neg] == 0):
            print(
                "  [INFO] Dropping run %d (pos=%d, neg=%d) due to missing condition."
                % (run_i, run_counts[args.cond_pos], run_counts[args.cond_neg])
            )
            continue

        if run_counts[args.cond_pos] == 0 or run_counts[args.cond_neg] == 0:
            print(
                "  [WARN] Run %d contains only one condition (pos=%d, neg=%d)."
                % (run_i, run_counts[args.cond_pos], run_counts[args.cond_neg])
            )

        betas_imgs.extend(run_beta_imgs)
        labels.extend(run_label_vals)
        group_labels.extend([run_i] * len(run_label_vals))

    if not betas_imgs:
        raise RuntimeError("No beta maps found for the requested conditions.")

    print("Per-run trial counts (positive, negative):")
    for run_i, counts in per_run_counts:
        print(
            "  Run %d → %s=%d, %s=%d"
            % (
                run_i,
                args.cond_pos,
                counts[args.cond_pos],
                args.cond_neg,
                counts[args.cond_neg],
            )
        )

    X4d = concat_imgs(betas_imgs)
    y = np.array(labels)
    groups = np.array(group_labels)

    if args.use_group_cv:
        cv = LeaveOneGroupOut()
    else:
        cv = StratifiedKFold(n_splits=args.n_splits)

    mask_img = None
    if args.mask_img:
        mask_img = load_img(args.mask_img)
    elif args.use_aparcaseg_visual:
        visual_substrings = [s.strip().lower() for s in args.visual_labels.split(",") if s.strip()]
        mask_img = build_visual_mask_from_aparcaseg(
            X4d_img=X4d,
            dseg_path=args.aparcaseg_dseg,
            tsv_path=args.aparcaseg_tsv,
            visual_substrings=visual_substrings,
        )

    if mask_img is not None and args.continuous:
        mask_img = make_continuous_mask(
            mask_img,
            close_mm=args.close_mm,
            bridge_mm=args.bridge_mm,
            max_iter=20,
        )

    print("Running searchlight …")
    run_searchlight_binary(
        X4d, y, groups,
        radius_mm=args.radius,
        mask_img=mask_img,
        cv=cv,
        out_nii_path=args.out_nii,
    )
    print("Searchlight Done")


if __name__ == "__main__":
    main()
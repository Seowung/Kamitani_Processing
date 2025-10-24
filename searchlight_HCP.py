#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-constrained searchlight using preprocessed trials_by_voxel + HCP-MMP1 (Figshare) atlas.
(Argparse-ified from the working Jupyter pipeline, without changing logic.)

HCP_HEMI_PARCELS "VC" "TPJ" "IPL" "PC" "STS" "TE" "MTC" "Insula" "DLPFC" "DMPFC" "VMPFC" "ACC" "OFC"


python searchlight_HCP.py \
  --bids-dir /orange/ruogu.fang/leem.s/EmotionVideo/Kamitani \
  --deriv-dir /orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/derivatives/preprocessed \
  --hcp-nifti /orange/ruogu.fang/leem.s/EmotionVideo/HCP-MMP1/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz \
  --hcp-labels /orange/ruogu.fang/leem.s/EmotionVideo/HCP-MMP1/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt \
  --sub-id 01 \
  --parcels "L_V1_ROI,R_V1_ROI,L_V2_ROI,R_V2_ROI,L_V3_ROI,R_V3_ROI,L_V3A_ROI,R_V3A_ROI" \
  --class-pos unpleasant --class-neg neutral \
  --subset-per-class 10 \
  --radius-mm 5 --cv-splits 5 --scoring accuracy --n-jobs 8 \
  --save-figs
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.maskers import NiftiMasker
from nilearn import image as nimg, plotting
from nilearn.decoding import SearchLight

from bids import BIDSLayout
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils import check_random_state

def resolve_parcels_arg(parcels_arg: str, labels_df: pd.DataFrame) -> list[str]:
    """Allow macros (VC, TPJ, etc.) or 'all' keyword."""
    arg = parcels_arg.strip()
    if arg.lower() == "all":
        return labels_df["name"].tolist()
    if arg in HCP_HEMI_PARCELS:
        return HCP_HEMI_PARCELS[arg]
    # otherwise treat as comma-separated list
    return [p.strip() for p in arg.split(",") if p.strip()]


def _list_subjects(deriv_dir: Path):
    subs = set()
    for p in deriv_dir.glob("sub-*/*_trials_by_voxel.npy"):
        subs.add(p.name.split("_")[0].replace("sub-",""))
    for p in deriv_dir.glob("sub-*_trials_by_voxel.npy"):
        subs.add(p.name.split("_")[0].replace("sub-",""))
    return sorted(subs)

def _paths_for(deriv_dir: Path, sub_id: str):
    sub = f"sub-{sub_id}"
    candidates = [
        {"X": deriv_dir / sub / f"{sub}_trials_by_voxel.npy",
         "meta": deriv_dir / sub / f"{sub}_trials_meta.csv",
         "mask": deriv_dir / sub / f"{sub}_mask_info.npz"},
        {"X": deriv_dir / f"{sub}_trials_by_voxel.npy",
         "meta": deriv_dir / f"{sub}_trials_meta.csv",
         "mask": deriv_dir / f"{sub}_mask_info.npz"},
    ]
    for c in candidates:
        if all(p.exists() for p in c.values()):
            return c
    raise FileNotFoundError(f"Could not find X/meta/mask for {sub} in {deriv_dir}")

def load_preprocessed(deriv_dir: Path, sub_id: str|None=None):
    if sub_id is None:
        subs = _list_subjects(deriv_dir)
        if not subs: raise RuntimeError("No *_trials_by_voxel.npy found.")
        sub_id = subs[0]
        print(f"[info] using subject sub-{sub_id}")
    paths = _paths_for(deriv_dir, sub_id)
    X = np.load(paths["X"])
    meta = pd.read_csv(paths["meta"])
    mi = np.load(paths["mask"], allow_pickle=True)
    mask_img = nib.Nifti1Image(mi["mask"].astype(np.uint8), mi["affine"])
    return sub_id, X, meta, mask_img

def reconstruct_4d_from_trials(X: np.ndarray, mask_img: nib.Nifti1Image):
    masker = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False, t_r=None)
    masker.fit()
    return masker.inverse_transform(X)

def load_figshare_hcp(hcp_nifti: Path, labels_txt: Path):
    atlas_img = nib.load(str(hcp_nifti))
    if atlas_img.ndim != 3:
        raise ValueError("Expected 3D HCP-MMP integer label map (Figshare); got non-3D image.")
    ids, names = [], []
    for ln in labels_txt.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln: continue
        parts = ln.split(None, 1)
        if len(parts) != 2: continue
        pid = int(parts[0]); pname = parts[1].strip()
        ids.append(pid); names.append(pname)
    labels_df = pd.DataFrame({"id": ids, "name": names})
    return atlas_img, labels_df

def ids_for_base_parcels(labels_df: pd.DataFrame, base_list: list[str]) -> list[int]:
    base_set = {b.strip().upper() for b in base_list}
    out_ids = []
    for _, row in labels_df.iterrows():
        base = row["name"].upper()
        if base in base_set:
            out_ids.append(int(row["id"]))
    return sorted(set(out_ids))

def attach_trial_names_from_stim_id(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'stim_id' of form '{trial_name}_t{onset}' into two new columns:
      - trial_name (lowercase string)
      - onset_from_id (float, optional)
    If 'stim_id' is missing or malformed, assigns NaN.
    Returns a copy of meta with added columns.
    """
    trial_names, onsets = [], []
    for val in meta.get("stim_id", []):
        if isinstance(val, str):
            m = re.match(r"^(.+?)_t([0-9]*\.?[0-9]+)$", val)
            if m:
                trial_names.append(m.group(1).lower())
                onsets.append(float(m.group(2)))
            else:
                # fallback if doesn't match _t pattern
                trial_names.append(val.lower())
                onsets.append(float("nan"))
        else:
            trial_names.append(None)
            onsets.append(float("nan"))
    meta = meta.copy()
    meta["trial_name"] = trial_names
    meta["onset_from_id"] = onsets
    return meta

def make_binary_dataset(X_full: np.ndarray,
                        meta: pd.DataFrame,
                        class_pos: str, class_neg: str,
                        subset_per_class: int|None,
                        random_seed: int):
    m = meta.copy()
    if "trial_name" not in m.columns or m["trial_name"].isna().all():
        raise RuntimeError("meta lacks 'trial_name'. Run attach_trial_names_from_bids first.")
    wanted = {class_pos.lower(), class_neg.lower()}
    keep = m["trial_name"].str.lower().isin(wanted)
    if not keep.any():
        raise RuntimeError(f"No rows match {wanted}. Found: {sorted(m['trial_name'].dropna().unique())}")
    X = X_full[keep.values]
    m_sel = m.loc[keep].reset_index(drop=True)
    y = (m_sel["trial_name"].str.lower().values == class_pos.lower()).astype(int)

    if subset_per_class is not None and subset_per_class > 0:
        rng = check_random_state(random_seed)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        n_pos = min(subset_per_class, len(idx_pos))
        n_neg = min(subset_per_class, len(idx_neg))
        sel = np.r_[rng.choice(idx_pos, n_pos, replace=False),
                    rng.choice(idx_neg, n_neg, replace=False)]
        sel.sort()
        X = X[sel]; y = y[sel]; m_sel = m_sel.iloc[sel].reset_index(drop=True)

    return X, y, m_sel

def run(args):
    sub_id, X_all, meta, mask_img = load_preprocessed(args.deriv_dir, args.sub_id)
    img4d = reconstruct_4d_from_trials(X_all, mask_img)
    n_total = int(mask_img.get_fdata().sum())
    print(f"[ok] sub-{sub_id}: X_all={X_all.shape} | grid vox={n_total}")

    hcp_img, labels_df = load_figshare_hcp(args.hcp_nifti, args.hcp_labels)

    parcels = resolve_parcels_arg(args.parcels, labels_df)
    parcel_ids = ids_for_base_parcels(labels_df, parcels)
    print(f"[info] requested parcels: {parcels[:8]}{' …' if len(parcels)>8 else ''}")
    print(f"[info] matched parcel IDs: {parcel_ids[:10]}{' …' if len(parcel_ids)>10 else ''} (n={len(parcel_ids)})")
    if not parcel_ids:
        raise RuntimeError("No parcel IDs matched the requested names. Check --parcels against your labels file.")

    hcp_data = np.round(hcp_img.get_fdata())
    roi_mask_in_atlas = nib.Nifti1Image(np.isin(hcp_data, parcel_ids).astype(np.uint8), hcp_img.affine)

    roi_mask_on_grid = nimg.resample_to_img(roi_mask_in_atlas, img4d, interpolation="nearest")
    roi_mask_on_grid = nib.Nifti1Image((roi_mask_on_grid.get_fdata()>0).astype(np.uint8), roi_mask_on_grid.affine)

    combo_mask = nimg.math_img("(a>0) & (b>0)", a=roi_mask_on_grid, b=mask_img)
    combo_mask = nib.Nifti1Image((combo_mask.get_fdata()>0).astype(np.uint8), combo_mask.affine)

    n_roi = int(combo_mask.get_fdata().sum())
    print(f"[counts] ROI voxels: {n_roi} ({100.0*n_roi/max(n_total,1):.1f}% of grid)")

    roi_masker = NiftiMasker(mask_img=combo_mask, standardize=False, detrend=False, t_r=None)
    roi_masker.fit()
    img4d_full = reconstruct_4d_from_trials(X_all, mask_img)
    X_roi = roi_masker.transform(img4d_full)
    print(f"[ok] X_roi: {X_roi.shape} (trials × roi_voxels)")

    if "trial_name" not in meta.columns or meta["trial_name"].isna().all():
        meta = attach_trial_names_from_stim_id(meta)

    X_bin, y_bin, meta_bin = make_binary_dataset(
        X_roi, meta, args.class_pos, args.class_neg, args.subset_per_class, args.seed
    )
    print(f"[info] {args.class_pos} vs {args.class_neg} → n={len(y_bin)} trials | roi_vox={X_bin.shape[1]}")

    roi_masker = NiftiMasker(mask_img=combo_mask, standardize=False, detrend=False, t_r=None)
    roi_masker.fit()
    img4d_roi = roi_masker.inverse_transform(X_bin)

    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
    estimator = SVC(kernel="linear", probability=False)

    sl = SearchLight(
        mask_img=combo_mask,
        radius=args.radius_mm,
        scoring=args.scoring,
        cv=cv,
        n_jobs=args.n_jobs,
        verbose=1
    )
    print(f"[info] Fitting SearchLight (radius={args.radius_mm}mm, scoring={args.scoring}, CV={args.cv_splits}) …")
    sl.fit(img4d_roi, y_bin)

    sl_scores = sl.scores_.astype(np.float32)
    sl_img = nib.Nifti1Image(sl_scores, img4d_roi.affine, img4d_roi.header)
    print("[ok] SearchLight finished:", sl_scores.shape)

    out_dir = args.deriv_dir / f"sub-{sub_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_ROI_r{}".format(int(args.radius_mm))
    out_nii = out_dir / f"sub-{sub_id}_sl_{args.class_pos}_vs_{args.class_neg}{tag}.nii.gz"
    nib.save(sl_img, str(out_nii))
    print("[save]", out_nii)

    if args.save_figs:
        try:
            bg = plotting.load_mni152_template()
        except Exception:
            bg = None
        sl_finite = np.nan_to_num(sl_scores, nan=0.0)
        plotting.plot_glass_brain(
            nib.Nifti1Image(sl_finite, sl_img.affine, sl_img.header),
            display_mode="lyrz", colorbar=True,
            title=f"SearchLight {args.class_pos} vs {args.class_neg} (ROI)"
        ).savefig(str(out_dir / f"sub-{sub_id}_sl_glass{tag}.png"), dpi=150, bbox_inches="tight")
        finite = sl_finite[sl_finite > 0]
        thr = float(np.percentile(finite, 75)) if finite.size else None
        plotting.plot_stat_map(
            sl_img, bg_img=bg, threshold=thr, display_mode="ortho", colorbar=True,
            title="Top quartile"
        ).savefig(str(out_dir / f"sub-{sub_id}_sl_stat{tag}.png"), dpi=150, bbox_inches="tight")
        print("[save] figures")

    return out_nii

def parse_args():
    p = argparse.ArgumentParser(description="ROI-constrained searchlight (CLI from working notebook)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bids-dir", type=Path, required=True, help="BIDS root (for events.tsv to get trial_name).")
    p.add_argument("--deriv-dir", type=Path, required=True, help="Derivatives/preprocessed dir with npy/csv/npz.")
    p.add_argument("--hcp-nifti", type=Path, required=True, help="Figshare HCP-MMP1 3D integer label map (NIfTI).")
    p.add_argument("--hcp-labels", type=Path, required=True, help="Figshare labels TXT (id name per line).")
    p.add_argument("--sub-id", type=str, default=None, help="Subject id like '01'. If omitted, pick first found.")
    p.add_argument("--parcels", type=str, required=True,
                   help="Comma-separated hemisphere names, e.g., 'L_V1_ROI,R_V1_ROI,L_V2_ROI,R_V2_ROI'.")
    p.add_argument("--class-pos", type=str, default="unpleasant", help="Positive class (trial_name).")
    p.add_argument("--class-neg", type=str, default="neutral", help="Negative class (trial_name).")
    p.add_argument("--subset-per-class", type=int, default=30, help="Small subset per class for quick test; 0 disables.")
    p.add_argument("--radius-mm", type=float, default=6.0, help="Searchlight sphere radius in mm.")
    p.add_argument("--cv-splits", type=int, default=5, help="Stratified K-fold splits.")
    p.add_argument("--scoring", type=str, default="roc_auc", choices=["roc_auc","accuracy"], help="Scoring metric.")
    p.add_argument("--n-jobs", type=int, default=8, help="Parallel jobs for SearchLight.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--onset-tol", type=float, default=0.1, help="Onset tolerance (sec) for matching events.")
    p.add_argument("--save-figs", action="store_true", help="Save PNG figures next to the NIfTI.")
    args = p.parse_args()
    if args.subset_per_class is not None and args.subset_per_class <= 0:
        args.subset_per_class = None
    return args

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# train_regression.py
# Notebook-style script: inline parcel loop, mirrors Jupyter conventions.

#!/usr/bin/env python3
# train_regression.py
# Strict notebook-aligned pipeline with your embedded helpers.
# - Loads preprocessed trials & mask
# - Builds trial × total_voxels (concatenated ROI voxels) using your build_parcel_feature_matrix
# - Aligns labels (numeric key + 1..11 → 2186..2196 patch)
# - Runs 6-fold RidgeCV *per ROI* (slice its voxel block) and saves per-ROI results

'''
python train_regression.py \
  --deriv-dir /orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/derivatives/preprocessed \
  --subjects 01 \
  --labels-csv ../ckvideo_data/CowenKeltnerEmotionalVideos.csv \
  --hcp-nifti /orange/ruogu.fang/leem.s/EmotionVideo/HCP-MMP1/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz \
  --hcp-labels /orange/ruogu.fang/leem.s/EmotionVideo/HCP-MMP1/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt \
  --out-dir /orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/derivatives/roi_cv_outputs \
  --hemi L \
  --n-folds 6

'''

from __future__ import annotations
import argparse
from pathlib import Path
import re
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.maskers import NiftiMasker
from nilearn import image as nimg

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


# =========================
# Your embedded helpers
# =========================

def _paths_for(deriv_dir: Path, sub_id: str):
    sub = f"sub-{sub_id}"
    cands = [
        {"X": deriv_dir / sub / f"{sub}_trials_by_voxel.npy",
         "meta": deriv_dir / sub / f"{sub}_trials_meta.csv",
         "mask": deriv_dir / sub / f"{sub}_mask_info.npz"},
        {"X": deriv_dir / f"{sub}_trials_by_voxel.npy",
         "meta": deriv_dir / f"{sub}_trials_meta.csv",
         "mask": deriv_dir / f"{sub}_mask_info.npz"},
    ]
    for c in cands:
        if all(p.exists() for p in c.values()):
            return c
    raise FileNotFoundError(f"Could not find preprocessed npy/csv/npz triple for {sub}")

def load_preprocessed(deriv_dir: Path, sub_id: str):
    paths = _paths_for(deriv_dir, sub_id)
    X = np.load(paths["X"])
    meta = pd.read_csv(paths["meta"])
    mi = np.load(paths["mask"], allow_pickle=True)
    mask_img = nib.Nifti1Image(mi["mask"].astype(np.uint8), mi["affine"])
    return X, meta, mask_img

def reconstruct_4d_from_trials(X: np.ndarray, mask_img: nib.Nifti1Image):
    m = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False, t_r=None).fit()
    return m.inverse_transform(X)

def load_figshare_hcp(hcp_nifti: Path, labels_txt: Path):
    img = nib.load(str(hcp_nifti))
    if img.ndim != 3:
        raise ValueError("HCP-MMP1 volumetric atlas must be a 3D integer label map.")
    ids, names = [], []
    for ln in labels_txt.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln: continue
        k, v = ln.split(None, 1)
        ids.append(int(k)); names.append(v.strip())
    labels_df = pd.DataFrame({"id": ids, "name": names})
    return img, labels_df

def ids_for_names(labels_df: pd.DataFrame, names: list[str]) -> list[int]:
    lookup = {n.upper(): i for n, i in zip(labels_df["name"].astype(str), labels_df["id"].astype(int))}
    out = []
    for nm in names:
        key = nm.strip().upper()
        if key in lookup: out.append(lookup[key])
    return sorted(set(out))

def build_single_parcel_mask(parcel_name: str,
                             atlas_img: nib.Nifti1Image,
                             labels_df,
                             like_img: nib.Nifti1Image | None = None) -> nib.Nifti1Image:
    """
    Build a binary mask for one HCP parcel (e.g., 'L_V1_ROI'), resampled to like_img if provided.
    """
    row = labels_df.loc[labels_df["name"].str.fullmatch(parcel_name, case=False)]
    if row.empty:
        raise ValueError(f"Parcel name {parcel_name} not found in atlas labels.")
    pid = int(row["id"].iloc[0])

    atlas_data = np.round(atlas_img.get_fdata()).astype(int)
    mask_data = (atlas_data == pid).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, atlas_img.affine)

    if like_img is not None:
        mask_img = nimg.resample_to_img(mask_img, like_img, interpolation="nearest")

    return mask_img

def build_parcel_feature_matrix(img4d_trials, parcel_names: list[str], atlas_img, labels_df):
    """
    For each parcel in 'parcel_names', extract *all voxel activations* and flatten.
    Returns:
      X_parcels: (n_trials, n_voxels_total)
      kept_names: list[str] for each voxel column in order (e.g., 'V1_00001', 'V1_00002', ...)
      voxel_counts: list[int] number of voxels per parcel (same order as parcel_names)
    """
    X_voxels, names_out, counts = [], [], []
    total_vox = 0

    for nm in parcel_names:
        pmask = build_single_parcel_mask(nm, atlas_img, labels_df, like_img=img4d_trials)
        mask_data = pmask.get_fdata() > 0
        nvox = int(mask_data.sum())

        if nvox == 0:
            print(f"[warn] {nm}: 0 voxels found, skipping.")
            counts.append(0)
            continue

        masker = NiftiMasker(mask_img=pmask, standardize=False, detrend=False, t_r=None)
        X_roi = masker.fit_transform(img4d_trials)  # (n_trials, n_vox_in_parcel)
        X_voxels.append(X_roi)
        counts.append(nvox)

        voxel_names = [f"{nm}_{i:05d}" for i in range(1, nvox + 1)]
        names_out.extend(voxel_names)
        total_vox += nvox

    if X_voxels:
        X_parcels = np.concatenate(X_voxels, axis=1)  # stack all voxel features side-by-side
    else:
        X_parcels = np.empty((img4d_trials.shape[-1], 0))

    print(f"[ok] Flattened features: {X_parcels.shape} (n_trials × total_voxels={total_vox})")
    return X_parcels, names_out, counts

def load_label_table(csv_path: Path):
    """
    Expects: col0 = filename.mp4, cols1..34 = label values.
    Returns df with:
      stimulus_name (stem), and 34 label columns as float
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 35:
        raise ValueError("Expected at least 35 columns (filename + 34 labels).")
    df = df.copy()
    df["stimulus_name"] = df.iloc[:,0].astype(str).apply(lambda s: Path(s).stem)
    label_cols = df.columns[1:35]
    df_labels = df[["stimulus_name"] + list(label_cols)].copy()
    for c in label_cols:
        df_labels[c] = pd.to_numeric(df_labels[c], errors="coerce")
    return df_labels, list(label_cols)

def _extract_numeric_id(s: str) -> int | None:
    """
    Grab the last run of digits from s and return it as int.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    stem = Path(s).stem
    m = re.search(r'(\d+)$', stem)
    if m:
        return int(m.group(1))
    return None

def _numeric_key_series(series: pd.Series) -> pd.Series:
    return series.apply(_extract_numeric_id)


# =========================
# Per-subject notebook-style flow
# =========================

def run_subject(
    DERIV_DIR: Path,
    SUB_ID: str,
    LABELS_CSV: Path,
    HCP_NIFTI: Path,
    HCP_LABELS: Path,
    OUT_DIR: Path,
    HEMI: str = "L",
    USE_ALL_ATLAS_PARCELS: bool = False,
    N_FOLDS: int = 6,
    SEED: int = 42,
):
    # ---- Load preprocessed trials and grid ----
    X_all, meta, mask_img = load_preprocessed(DERIV_DIR, SUB_ID)
    img4d = reconstruct_4d_from_trials(X_all, mask_img)
    print(f"[ok] trials x vox: {X_all.shape} | grid vox={int(mask_img.get_fdata().sum())}")

    # meta merge key (normalized string)
    if "stimulus_name" not in meta.columns:
        raise RuntimeError("meta must contain 'stimulus_name' for label alignment.")
    meta_key = meta["stimulus_name"].astype(str).str.strip().str.lower()
    meta_aligned = meta.copy()
    meta_aligned["stimulus_name_norm"] = meta_key

    # ---- Load HCP atlas + labels ----
    atlas_img, labels_df = load_figshare_hcp(HCP_NIFTI, HCP_LABELS)

    # ---- Choose parcel list ----
    if USE_ALL_ATLAS_PARCELS:
        PARCEL_LIST = labels_df["name"].astype(str).tolist()
    else:
        hemi_prefix = f"{HEMI}_"
        hemi_df = labels_df[labels_df["name"].astype(str).str.startswith(hemi_prefix)].copy()
        if len(hemi_df) < 180:
            raise RuntimeError(f"Found only {len(hemi_df)} parcels for hemi={HEMI}")
        PARCEL_LIST = hemi_df.iloc[:180]["name"].astype(str).tolist()
    print(f"[info] n parcels requested: {len(PARCEL_LIST)}")

    # ---- Build trial × voxel feature matrix (concatenated per-ROI voxels) ----
    X_parc, voxel_names, parcel_vox = build_parcel_feature_matrix(
        img4d_trials=img4d,
        parcel_names=PARCEL_LIST,
        atlas_img=atlas_img,
        labels_df=labels_df
    )
    print(f"[ok] Feature matrix: {X_parc.shape} (trials × voxels)")

    # ---- Load CK label table (strictly as in your block) ----
    df_labels_raw = pd.read_csv(LABELS_CSV)
    if df_labels_raw.shape[1] < 35:
        raise ValueError("Expected at least 35 columns (filename + 34 labels).")

    csv_name_col = df_labels_raw.iloc[:, 0].astype(str)
    csv_key_num = _numeric_key_series(csv_name_col)

    csv_numeric_strlens = csv_name_col.apply(
        lambda s: len(re.search(r'(\d+)$', Path(s).stem).group(1)) if re.search(r'(\d+)$', Path(s).stem) else np.nan
    )
    pad_width = int(pd.Series(csv_numeric_strlens).dropna().mode().iloc[0]) if csv_numeric_strlens.notna().any() else None

    label_cols = df_labels_raw.columns[1:35]
    df_labels = df_labels_raw.copy()
    for c in label_cols:
        df_labels[c] = pd.to_numeric(df_labels[c], errors="coerce")
    df_labels["stim_id_num"] = csv_key_num
    df_labels["stim_id_num"] = pd.to_numeric(csv_key_num, errors="coerce").astype("Int64")

    # ---- Duplicate rows 1..11 to 2186..2196 (CSV side) ----
    src_ids = list(range(1, 12))
    dst_ids = list(range(2186, 2197))

    rows_to_append = []
    for s, d in zip(src_ids, dst_ids):
        s_key = int(s)
        r = df_labels[df_labels["stim_id_num"] == s_key]
        if r.empty:
            print(f"[warn] source id {s_key} not found in df_labels_aligned; skipping")
            continue
        new = r.iloc[0].copy()
        # store as numeric Int64; pad only for display if needed
        new["stim_id_num"] = d
        rows_to_append.append(new)

    if rows_to_append:
        df_labels = pd.concat([df_labels, pd.DataFrame(rows_to_append)], ignore_index=True)
        df_labels = df_labels.drop_duplicates(subset=["stim_id_num"], keep="last").reset_index(drop=True)

    # ---- meta numeric key, merge on numeric to avoid zero-pad issues ----
    if "stimulus_name" not in meta.columns:
        raise RuntimeError("meta must contain 'stim_id_num' for label alignment.")
    meta_key_num = _numeric_key_series(meta["stimulus_name"])
    meta_aligned = meta.copy()
    meta_aligned["stim_id_num"] = pd.to_numeric(meta_key_num, errors="coerce").astype("Int64")

    joined = meta_aligned.merge(
        df_labels[["stim_id_num"] + list(label_cols)],
        on="stim_id_num",
        how="left"
    )

    # ---- Copy again 1..11 → 2186..2196 on the joined table (as in your block) ----
    for old_id, new_id in zip(range(1, 12), range(2186, 2197)):
        src = joined.loc[joined["stim_id_num"] == old_id, label_cols]
        if src.empty:
            print(f"[warn] no source for {old_id}")
            continue
        joined.loc[joined["stim_id_num"] == new_id, label_cols] = src.values
    print("[done] copied labels from 1–11 → 2186–2196")

    # ---- Filter rows that have all labels; subset X and build Y ----
    has_y = joined[label_cols].notna().all(axis=1)
    X = X_parc[has_y, :]
    Y = joined.loc[has_y, label_cols].to_numpy(dtype=float)

    print(f"[align] kept rows with labels: {has_y.sum()} / {len(has_y)}")
    print(f"[final] X={X.shape}, Y={Y.shape} (targets={len(label_cols)})")

    # ---- Prepare per-ROI column slices from voxel counts ----
    # parcel_vox[i] = number of voxels for PARCEL_LIST[i]
    col_slices = []
    start = 0
    for nvox in parcel_vox:
        end = start + int(nvox)
        col_slices.append((start, end))
        start = end

    # ---- Outputs ----
    out_dir = OUT_DIR / f"sub-{SUB_ID}" / f"roi_cv_{HEMI}_180" if not USE_ALL_ATLAS_PARCELS else OUT_DIR / f"sub-{SUB_ID}" / "roi_cv_all"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CV per ROI (multi-voxel within ROI) ----
    rows = []
    for k, nm in enumerate(PARCEL_LIST):
        nvox = int(parcel_vox[k]) if k < len(parcel_vox) else 0
        if nvox == 0:
            # Save NaNs if no voxels present
            r_vec = np.full(Y.shape[1], np.nan, dtype=float)
            pd.DataFrame({"label": label_cols, "pearson_r": r_vec}).to_csv(
                out_dir / f"roi_{k:03d}_{nm.replace('/', '-')}_label_r.csv", index=False
            )
            rows.append({"roi_rank": k, "parcel_name": nm, "n_voxels": 0, "mean_r": np.nan, "alpha_median": np.nan})
            print(f"  [{k+1:03d}/{len(PARCEL_LIST):03d}] {nm:>20s} | vox={0:5d} | mean r=  nan")
            continue

        s, e = col_slices[k]
        X_roi = X[:, s:e]  # (n_trials, nvoxels_in_parcel)

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros_like(Y, dtype=float)
        alphas = []

        for tr, te in kf.split(X_roi):
            sc = StandardScaler(with_mean=True, with_std=True).fit(X_roi[tr])
            Xtr, Xte = sc.transform(X_roi[tr]), sc.transform(X_roi[te])

            ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
            ridge.fit(Xtr, Y[tr])
            oof[te] = ridge.predict(Xte)
            alphas.append(float(ridge.alpha_))

        r_vec = [pearsonr(Y[:, j], oof[:, j])[0] if np.std(Y[:, j]) else np.nan for j in range(Y.shape[1])]
        r_vec = np.asarray(r_vec, float)

        pd.DataFrame({"label": label_cols, "pearson_r": r_vec}).to_csv(
            out_dir / f"roi_{k:03d}_{nm.replace('/', '-')}_label_r.csv", index=False
        )

        rows.append({
            "roi_rank": k,
            "parcel_name": nm,
            "n_voxels": nvox,
            "mean_r": float(np.nanmean(r_vec)),
            "alpha_median": float(np.median(alphas)) if len(alphas) else np.nan
        })
        print(f"  [{k+1:03d}/{len(PARCEL_LIST):03d}] {nm:>20s} | vox={nvox:5d} | mean r={np.nanmean(r_vec): .3f}")

    # ---- Summaries ----
    pd.DataFrame(rows).sort_values("mean_r", ascending=False).to_csv(out_dir / "roi_summary.csv", index=False)
    pd.DataFrame({"parcel": PARCEL_LIST, "n_vox": parcel_vox}).to_csv(out_dir / "parcel_voxel_counts.csv", index=False)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "subject": SUB_ID, "hemi": HEMI, "n_parcels": len(PARCEL_LIST),
            "n_folds": N_FOLDS, "n_trials": int(Y.shape[0]), "n_labels": int(Y.shape[1]),
            "labels": list(map(str, label_cols)), "use_all_parcels": bool(USE_ALL_ATLAS_PARCELS)
        }, f, indent=2)

    print(f"[save] {out_dir}")


# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="Per-subject ROI CV (notebook-aligned; embedded helpers).")
    ap.add_argument("--deriv-dir", type=Path, required=True)
    ap.add_argument("--subjects", type=str, required=True, help="Comma-separated IDs, e.g., 01,02")
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--hcp-nifti", type=Path, required=True)
    ap.add_argument("--hcp-labels", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hemi", type=str, default="L", choices=["L", "R"])
    ap.add_argument("--use-all-atlas-parcels", action="store_true", help="Use all parcels in labels file (ignores --hemi).")
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    DERIV_DIR = args.deriv_dir
    LABELS_CSV = args.labels_csv
    HCP_NIFTI = args.hcp_nifti
    HCP_LABELS = args.hcp_labels
    OUT_DIR = args.out_dir
    HEMI = args.hemi
    USE_ALL = args.use_all_atlas_parcels
    N_FOLDS = args.n_folds
    SEED = args.seed

    for SUB_ID in [s.strip() for s in args.subjects.split(",") if s.strip()]:
        run_subject(
            DERIV_DIR=DERIV_DIR,
            SUB_ID=SUB_ID,
            LABELS_CSV=LABELS_CSV,
            HCP_NIFTI=HCP_NIFTI,
            HCP_LABELS=HCP_LABELS,
            OUT_DIR=OUT_DIR,
            HEMI=HEMI,
            USE_ALL_ATLAS_PARCELS=USE_ALL,
            N_FOLDS=N_FOLDS,
            SEED=SEED,
        )

if __name__ == "__main__":
    main()

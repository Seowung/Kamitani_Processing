from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image
import nibabel as nib
import numpy as np

from bids import BIDSLayout
from nilearn import image
from nilearn.maskers import NiftiMasker

BIDS_DIR = "/orange/ruogu.fang/leem.s/EmotionVideo/Kamitani"
OUT_DIR  = "/orange/ruogu.fang/leem.s/EmotionVideo/Kamitani/derivatives/preprocessed"

# Parameters
SPACE         = "MNI152NLin2009cAsym"
MASK_STRATEGY = "intersection"   # 'intersection' or 'union'
HRF_SHIFT_S   = 4.0              # hemodynamic delay
POST_REST_S   = 0.0              # seconds after each video block
Z_THRESH      = 3.0              # despike threshold (± SD)
N_TS_VOX      = 12               # # voxel traces to overlay in time-series QC
SAVE_QC_PNGS  = True             # if True, also save PNGs alongside inline plots

def despike(X: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Winsorize each voxel time series at ±z_thresh SD."""
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, ddof=1, keepdims=True)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    Z = np.clip(Z, -z_thresh, z_thresh)
    return Z * sd + mu

def zscore_per_trial(X: np.ndarray) -> np.ndarray:
    """Row-wise z-scoring (per-trial): for each trial, z-score across voxels."""
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, ddof=1, keepdims=True)
    sd[sd == 0] = 1
    return (X - mu) / sd

def zscore_per_voxel_by_session(X: np.ndarray, sessions: list | np.ndarray) -> np.ndarray:
    """
    Column-wise z-scoring per session (per voxel across trials).
    X: (n_trials_total, n_voxels)
    sessions: length n_trials_total (values like 'ses-01', None, etc.)
    """
    Xz = X.copy()
    # normalize session labels to strings so None groups together
    sessions = np.array([str(s) if s is not None else "ses-NA" for s in sessions])
    for ses in np.unique(sessions):
        idx = np.where(sessions == ses)[0]
        if idx.size == 0:
            continue
        mu = X[idx, :].mean(axis=0, keepdims=True)
        sd = X[idx, :].std(axis=0, ddof=1, keepdims=True)
        sd[sd == 0] = 1
        Xz[idx, :] = (X[idx, :] - mu) / sd
    return Xz


def get_block_indices(onset: float, duration: float, tr: float,
                      shift: float, post_rest: float, nTR: int) -> np.ndarray:
    """Return TR indices for (onset..onset+duration+post_rest) shifted by +shift seconds."""
    start = onset + shift
    end = onset + duration + post_rest + shift
    i0 = max(int(np.floor(start / tr)), 0)
    i1 = min(int(np.floor(end / tr)), nTR)
    return np.arange(i0, i1, dtype=int) if i1 > i0 else np.array([], dtype=int)

def choose_stim_id(row: pd.Series) -> str:
    for k in ("stim_file", "stimulus", "trial_name", "file", "name"):
        if k in row and pd.notna(row[k]):
            return str(row[k])
    return f"{str(row.get('trial_type','stim'))}_t{row['onset']:.3f}"

def filter_video_events(df: pd.DataFrame) -> pd.DataFrame:
    if "trial_type" in df.columns:
        vids = df[df["trial_type"].astype(str).str.contains("video", case=False, na=False)]
        if len(vids) == 0:
            vids = df[~df["trial_type"].astype(str).str.contains("rest|null", case=False, na=False)]
        return vids.copy()
    return df.copy()



def build_subject_mask(layout, sub_id, space, strategy="intersection"):
    """
    Combine run-level masks into one per-subject mask.
    Returns a nib.Nifti1Image.
    """
    # Be explicit: masks live in derivatives
    run_masks = layout.get(
        subject=sub_id, suffix="mask", space=space,
        desc="brain", extension=[".nii", ".nii.gz"],
        scope="derivatives", return_type="filename"
    )

    # If 'desc=brain' doesn’t exist in your fMRIPrep, try without it
    if not run_masks:
        run_masks = layout.get(
            subject=sub_id, suffix="mask", space=space,
            extension=[".nii", ".nii.gz"],
            scope="derivatives", return_type="filename"
        )

    if not run_masks:
        # fall back to any subject-level mask we can find
        subj_masks = layout.get(
            subject=sub_id, suffix="mask", space=space,
            extension=[".nii", ".nii.gz"],
            scope="derivatives", return_type="filename"
        )
        if subj_masks:
            return nib.load(subj_masks[0])
        raise RuntimeError(f"No masks found for sub-{sub_id} in space {space}")

    # Load first as reference; resample others to it
    ref_img = nib.load(run_masks[0])
    imgs = [ref_img] + [image.resample_to_img(nib.load(p), ref_img, interpolation="nearest")
                        for p in run_masks[1:]]

    datas = [(np.asanyarray(img.get_fdata()) > 0.5) for img in imgs]
    if strategy == "intersection":
        mdat = np.logical_and.reduce(datas)
    elif strategy == "union":
        mdat = np.logical_or.reduce(datas)
    else:
        raise ValueError("mask_strategy must be 'intersection' or 'union'")

    mdat = mdat.astype(np.uint8)
    return nib.Nifti1Image(mdat, ref_img.affine, ref_img.header)


def maybe_save(figpath: Path):
    if SAVE_QC_PNGS:
        fig = plt.gcf()
        fig.savefig(figpath, dpi=150, bbox_inches="tight")

def plot_time_series(X: np.ndarray, events_df: pd.DataFrame, tr: float,
                     shift: float, post_rest: float, title: str,
                     n_voxels: int = 12, rng_seed: int = 0, save_path: Path | None = None):
    T, V = X.shape
    rng = np.random.default_rng(rng_seed)
    idx_vox = rng.choice(V, size=min(n_voxels, V), replace=False)
    t = np.arange(T)

    plt.figure(figsize=(14,4))
    plt.plot(t, X.mean(1), lw=2, label="Global mean")
    for vi in idx_vox:
        plt.plot(t, X[:, vi], lw=0.8, alpha=0.6)
    for _, r in events_df.iterrows():
        idxs = get_block_indices(float(r['onset']), float(r['duration']), tr, shift, post_rest, T)
        if idxs.size:
            plt.axvspan(idxs[0], idxs[-1], alpha=0.15)
    plt.xlabel("TR #"); plt.ylabel("Signal (a.u.)"); plt.title(title); plt.legend(loc="upper right")
    maybe_save(save_path if save_path is not None else Path("__no_save__"))
    plt.show()

def plot_despike_delta(X_raw: np.ndarray, X_ds: np.ndarray, save_path: Path | None = None):
    delta = (X_ds - X_raw).ravel()
    plt.figure(figsize=(6,4))
    plt.hist(delta, bins=100)
    plt.title("Despike delta histogram (post - pre)")
    plt.xlabel("Δ signal"); plt.ylabel("count")
    maybe_save(save_path if save_path is not None else Path("__no_save__"))
    plt.show()

def plot_block_hist(events_df: pd.DataFrame, tr: float, shift: float,
                    post_rest: float, T: int, save_path: Path | None = None):
    lengths = []
    for _, r in events_df.iterrows():
        idxs = get_block_indices(float(r['onset']), float(r['duration']), tr, shift, post_rest, T)
        if idxs.size:
            lengths.append(int(idxs.size))
    plt.figure(figsize=(6,4))
    plt.hist(lengths, bins=30)
    plt.title("Block nTR used (after HRF shift + post-rest)")
    plt.xlabel("nTR"); plt.ylabel("count")
    maybe_save(save_path if save_path is not None else Path("__no_save__"))
    plt.show()

def pca_reduce(X: np.ndarray, k: int = 10) -> np.ndarray:
    if X.size == 0: return X
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, U.shape[1])
    return U[:, :k] * S[:k]

def plot_pca_heatmap(Bz: np.ndarray, k: int = 10, save_path: Path | None = None):
    comp = pca_reduce(Bz, k=k)
    plt.figure(figsize=(8,6))
    im = plt.imshow(comp, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="PC score")
    plt.xlabel("PC"); plt.ylabel("Trial #")
    plt.title("Block means → PCA heatmap")
    maybe_save(save_path if save_path is not None else Path("__no_save__"))
    plt.show()

def plot_zscore_qc(Bz: np.ndarray, save_path: Path | None = None):
    mu = Bz.mean(1)
    sd = Bz.std(1, ddof=1)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].hist(mu, bins=60); ax[0].set_title("Per-trial means (row-wise z-score)")
    ax[1].hist(sd, bins=60); ax[1].set_title("Per-trial stds (row-wise z-score)")
    for a in ax: a.set_xlabel("value")
    maybe_save(save_path if save_path is not None else Path("__no_save__"))
    plt.show()

BIDS_DIR = Path(BIDS_DIR)
OUT_DIR  = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

layout = BIDSLayout(BIDS_DIR, derivatives=True, validate=False)
subjects = layout.get_subjects()
subjects[:5], len(subjects)

def process_subject(sub_id: str):
    sub = f"sub-{sub_id}"
    out_sub = OUT_DIR / sub
    qc_dir  = out_sub / "qc"
    out_sub.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # unified mask + masker
    unified_mask = build_subject_mask(layout, sub_id, space=SPACE, strategy=MASK_STRATEGY)
    masker = NiftiMasker(mask_img=unified_mask, standardize=False, detrend=False,
                         t_r=None, resampling_target="mask")

    bold_files = layout.get(subject=sub_id, suffix="bold", space=SPACE,
                            extension=[".nii",".nii.gz"], desc="preproc", return_type="file")
    all_blocks, all_meta = [], []

    for bf in bold_files:
        ent = layout.parse_file_entities(bf)
        task, run, session = ent.get("task"), ent.get("run"), ent.get("session")
        ev_files = layout.get(subject=sub_id, task=task, run=run, session=session, suffix="events",
                              extension=".tsv", return_type="file")
        if not ev_files:
            print(f"[WARN] No events for {Path(bf).name}")
            continue
        ev = pd.read_csv(ev_files[0], sep="\t")
        ev = filter_video_events(ev)
        if len(ev) == 0:
            print(f"[WARN] Events filtered to 0 rows for {Path(bf).name}")
            continue

        tr = float(nib.load(bf).header.get_zooms()[3])
        X_raw = masker.fit_transform(bf)
        X_ds  = despike(X_raw, z_thresh=Z_THRESH)

        # per-run QC (optional save)
        run_tag = f"{sub}_{task}_run-{run}"
        plot_time_series(X_raw, ev, tr, HRF_SHIFT_S, POST_REST_S,
                         f"{sub} {task} run-{run} — RAW",
                         n_voxels=N_TS_VOX, save_path=qc_dir / f"{run_tag}_ts_pre_despike.png")
        plot_time_series(X_ds, ev, tr, HRF_SHIFT_S, POST_REST_S,
                         f"{sub} {task} run-{run} — DESPIKED",
                         n_voxels=N_TS_VOX, save_path=qc_dir / f"{run_tag}_ts_post_despike.png")
        plot_despike_delta(X_raw, X_ds, save_path=qc_dir / f"{run_tag}_despike_effect_hist.png")
        plot_block_hist(ev, tr, HRF_SHIFT_S, POST_REST_S, X_raw.shape[0],
                        save_path=qc_dir / f"{run_tag}_block_length_distribution.png")

        # collect blocks
        T = X_ds.shape[0]
        for _, r in ev.iterrows():
            idxs = get_block_indices(float(r["onset"]), float(r["duration"]),
                                     tr, HRF_SHIFT_S, POST_REST_S, T)
            if idxs.size:
                all_blocks.append(X_ds[idxs, :].mean(0))
                all_meta.append({
                    "subject": sub, "task": task, "run": run, "session": session, 
                    "onset": float(r["onset"]), "duration": float(r["duration"]),
                    "stim_id": choose_stim_id(r), "stimulus_name":r["stimulus_name"] ,"nTR_used": int(idxs.size), "TR": tr
                })

    if not all_blocks:
        print(f"[WARN] No blocks extracted for {sub}")
        return None

    #X_all = np.vstack(all_blocks)
    #X_all = zscore_per_trial(X_all)
    
    X_all = np.vstack(all_blocks)

    # Build a sessions vector aligned to trials
    sessions_vec = [m.get("session") for m in all_meta]

    # Per-session, per-voxel z-score (column-wise within each session)
    X_all = zscore_per_voxel_by_session(X_all, sessions_vec)


    # save arrays + meta + mask + config
    np.save(out_sub / f"{sub}_trials_by_voxel.npy", X_all)
    pd.DataFrame(all_meta).to_csv(out_sub / f"{sub}_trials_meta.csv", index=False)
    np.savez_compressed(out_sub / f"{sub}_mask_info.npz",
                        mask=np.asanyarray(unified_mask.get_fdata()).astype(np.uint8),
                        affine=unified_mask.affine)
    with open(out_sub / f"{sub}_config.json", "w") as f:
        json.dump({
            "HRF_shift_s": HRF_SHIFT_S,
            "Post_rest_s": POST_REST_S,
            "Despike_z": Z_THRESH,
            #"zscore_mode": "row-wise (per trial)",
            "zscore_mode": "per-session (per-voxel, across trials)",
            "mask_strategy": MASK_STRATEGY,
            "space": SPACE
        }, f, indent=2)

    # cross-trial QC
    plot_pca_heatmap(X_all, k=10, save_path=qc_dir / f"{sub}_block_means_pca_heatmap.png")
    plot_zscore_qc(X_all, save_path=qc_dir / f"{sub}_zscore_qc.png")

    print(f"[OK] {sub}: {X_all.shape[0]} trials × {X_all.shape[1]} voxels → {out_sub}")
    return X_all


for sub_id in subjects:
    process_subject(sub_id)
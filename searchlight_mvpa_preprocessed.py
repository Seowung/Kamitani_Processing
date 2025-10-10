"""
run_searchlight.py — Voxel-wise searchlight decoding with nilearn.

Features:
- Load ROI data from a bdpy BData file and build a 4D NIfTI (X,Y,Z,T).
- Labels from:
    • BData (categorical)
    • BData valence key (mapped to unpleasant/neutral/pleasant)
    • A directory of 0001.mat … NNNN.mat valence files (mapped likewise)
- Optional averaging across repeated labels before decoding.
- Optional binary filtering to specific classes (e.g., unpleasant vs neutral).
- Linear SVC pipeline with StandardScaler inside nilearn SearchLight.
- Saves accuracy map and a glass-brain PNG.

Requires: bdpy, nibabel, nilearn, scikit-learn, pyyaml, scipy
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import re
import numpy as np
import nibabel as nib
import yaml
from scipy.io import loadmat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from nilearn.decoding import SearchLight
from nilearn import plotting
import bdpy


# ───────────────────────────── Helpers ─────────────────────────────

def _load_bdata_array(bdata: bdpy.BData, key: str, *, prefer_label: bool = True) -> np.ndarray:
    """Return a 1-D array stored in a ``BData`` container."""
    attempts = []
    if prefer_label:
        attempts.append(lambda: bdata.get_label(key))
    attempts.append(lambda: bdata.get(key))
    attempts.append(lambda: bdata.get_metadata(key))

    for attempt in attempts:
        try:
            values = attempt()
        except Exception:
            continue
        if values is None:
            continue

        array = np.asarray(values)
        if array.size == 0:
            continue
        if array.ndim > 1 and 1 in array.shape:
            array = array.reshape(-1)
        elif array.ndim > 1:
            raise ValueError(
                f"Expected 1-D data for key '{key}', but received shape {array.shape}."
            )
        return array.astype(array.dtype, copy=False)

    raise KeyError(f"Unable to locate data for key '{key}' in the BData file.")


def _load_roi_data(
    bdata: bdpy.BData,
    roi_expression: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get (samples × voxels) data and voxel selector for an ROI using bdpy.select.

    Examples of roi_expression:
        'VC'                # a named ROI
        'V1 | V2'           # union
        'Cortex = 1 & ~V1'  # logical expression (depends on your metadata)
    """
    # bdpy.select returns (X_roi, selector) when return_index=True
    roi_data, voxel_selector = bdata.select(roi_expression, return_index=True)

    # Ensure expected shapes/types
    roi_data = np.asarray(roi_data)
    voxel_selector = np.asarray(voxel_selector)

    # voxel_selector can be boolean or integer indices; both are fine downstream.
    # roi_data has shape (n_samples, n_selected_voxels).
    if roi_data.ndim != 2:
        raise ValueError(f"Expected 2-D ROI matrix, got shape {roi_data.shape}.")

    return roi_data, voxel_selector



def _aggregate_samples(
    matrix: np.ndarray,
    labels: Sequence,
    average_samples: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally average samples that share the same label."""
    labels = np.asarray(labels)
    if not average_samples:
        return matrix, labels

    aggregated = []
    aggregated_labels = []
    for label in np.unique(labels):
        idx = labels == label
        aggregated.append(np.nanmean(matrix[idx], axis=0))
        aggregated_labels.append(label)

    return np.vstack(aggregated), np.asarray(aggregated_labels)


def _labels_from_valence(valence: Sequence) -> np.ndarray:
    """Convert continuous valence ratings into categorical labels."""
    ratings = np.asarray(valence, dtype=float).reshape(-1)
    categories = np.full(ratings.shape, "neutral", dtype=object)
    categories[ratings > 7] = "pleasant"
    categories[ratings < 3] = "unpleasant"
    return categories


def _create_nifti(
    matrix: np.ndarray,
    selector: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Convert a matrix of voxel responses into a 4-D NIfTI image.

    matrix: (n_samples, n_voxels)
    selector: Boolean or index array selecting the voxels used for the matrix.
    metadata: dict containing voxel_i, voxel_j, voxel_k arrays (for full dataset).
    """
    voxel_i = metadata["voxel_i"][selector].astype(int)
    voxel_j = metadata["voxel_j"][selector].astype(int)
    voxel_k = metadata["voxel_k"][selector].astype(int)

    nx = int(voxel_i.max() + 1)
    ny = int(voxel_j.max() + 1)
    nz = int(voxel_k.max() + 1)
    n_samples = matrix.shape[0]

    volume = np.zeros((nx, ny, nz, n_samples), dtype=np.float32)
    mask = np.zeros((nx, ny, nz), dtype=np.uint8)

    volume[voxel_i, voxel_j, voxel_k] = matrix.T
    mask[voxel_i, voxel_j, voxel_k] = 1

    affine = np.eye(4)
    return nib.Nifti1Image(volume, affine), nib.Nifti1Image(mask, affine)


def _encode_labels(labels: Sequence) -> tuple[np.ndarray, list[str]]:
    """Encode arbitrary labels as integer class indices."""
    labels = np.asarray(labels)
    unique_labels, encoded = np.unique(labels, return_inverse=True)
    return encoded.astype(int), unique_labels.tolist()


def _load_roi_definition(path: Path | None) -> dict[str, str]:
    """Load ROI definitions from a YAML/JSON mapping."""
    if path is None:
        return {}

    with path.open() as f:
        if path.suffix in {".yaml", ".yml"}:
            rois = yaml.safe_load(f)
            if isinstance(rois, dict) and "rois" in rois:
                return {roi["name"]: roi["select"] for roi in rois["rois"]}
            raise ValueError("ROI YAML must contain a 'rois' list of name/select pairs.")
        rois = yaml.safe_load(f)

    if isinstance(rois, dict):
        return {name: definition for name, definition in rois.items()}

    raise ValueError("Unsupported ROI definition file format.")


def _infer_sample_ids_from_bdata(bdata: bdpy.BData) -> np.ndarray | None:
    """
    Try to recover a per-sample integer ID from BData to match 0001.mat ... NNNN.mat.
    Returns None if nothing usable is found.
    """
    candidate_keys = [
        "stimulus_id", "stimulus_index", "clip_id", "clip_index",
        "movie_id", "movie_index", "trial_id", "trial_index",
        "stimulus", "clip", "movie", "trial"
    ]
    for key in candidate_keys:
        arr = None
        for getter in (bdata.get_label, bdata.get, bdata.get_metadata):
            try:
                arr = getter(key)
                if arr is not None:
                    break
            except Exception:
                continue
        if arr is None:
            continue

        arr = np.asarray(arr)
        if arr.ndim > 1 and 1 in arr.shape:
            arr = arr.reshape(-1)

        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(int)

        if arr.dtype.kind in {"U", "S", "O"}:
            parsed = []
            ok = True
            for x in arr:
                s = str(x)
                m = re.search(r"(\d+)", s)
                if m:
                    parsed.append(int(m.group(1)))
                else:
                    ok = False
                    break
            if ok:
                return np.asarray(parsed, dtype=int)
    return None


def _read_valence_from_mat(file_path: Path, *, key_candidates: list[str]) -> float:
    """Load a single .mat file and return a scalar valence score."""
    data = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
    keys = [k for k in data.keys() if not k.startswith("__")]

    for k in key_candidates:
        if k in data:
            val = np.asarray(data[k]).reshape(-1)
            return float(val[0])

    if len(keys) == 1:
        val = np.asarray(data[keys[0]]).reshape(-1)
        return float(val[0])

    for k in keys:
        if any(tag in k.lower() for tag in ["valence", "score", "rating", "mean"]):
            val = np.asarray(data[k]).reshape(-1)
            return float(val[0])

    for k in keys:
        arr = np.asarray(data[k])
        if np.issubdtype(arr.dtype, np.number):
            return float(np.ravel(arr)[0])

    raise ValueError(f"Could not find a numeric valence in {file_path.name}")


def _load_valence_from_mat_dir(
    valence_dir: Path,
    *,
    n_samples: int,
    sample_ids: np.ndarray | None,
    mat_key_candidates: list[str] | None = None,
) -> np.ndarray:
    """Build a valence vector from a directory of 0001.mat … NNNN.mat files."""
    if mat_key_candidates is None:
        mat_key_candidates = ["mean_score", "valence", "score", "rating", "amt_score", "mean"]

    files = sorted([p for p in Path(valence_dir).glob("*.mat")])
    id_to_path: dict[int, Path] = {}
    for p in files:
        m = re.match(r"^(\d+)\.mat$", p.name) or re.match(r"^(\d+).*\.mat$", p.name)
        if not m:
            continue
        fid = int(m.group(1))
        id_to_path[fid] = p

    if not id_to_path:
        raise FileNotFoundError(f"No .mat files found in {valence_dir}")

    vals = np.empty(n_samples, dtype=float)

    if sample_ids is not None:
        for i, sid in enumerate(sample_ids):
            if sid not in id_to_path:
                raise KeyError(
                    f"Sample ID {sid} has no matching .mat file in {valence_dir} "
                    f"(expected {sid:04d}.mat)."
                )
            vals[i] = _read_valence_from_mat(id_to_path[sid], key_candidates=mat_key_candidates)
    else:
        needed = min(n_samples, len(files))
        sorted_paths = [id_to_path[k] for k in sorted(id_to_path.keys())]
        for i in range(needed):
            vals[i] = _read_valence_from_mat(sorted_paths[i], key_candidates=mat_key_candidates)
        if needed < n_samples:
            raise ValueError(
                f"Only {needed} .mat files found but dataset has {n_samples} samples."
            )

    return vals


def _keep_only_classes(matrix: np.ndarray, labels: np.ndarray, classes_to_keep: tuple[str, ...]):
    """Subset samples so only the requested classes remain."""
    labels = np.asarray(labels)
    keep_mask = np.isin(labels, classes_to_keep)
    if keep_mask.sum() < 2:
        raise ValueError(f"Not enough samples after filtering to {classes_to_keep}.")
    return matrix[keep_mask], labels[keep_mask], keep_mask


def _labels_to_binary(labels: np.ndarray, positive: tuple[str, ...], negative: tuple[str, ...]) -> np.ndarray:
    """Map labels to 0/1 after subsetting. 1=positive, 0=negative."""
    labels = np.asarray(labels)
    y = np.zeros(labels.shape[0], dtype=int)
    y[np.isin(labels, list(positive))] = 1
    return y


# ─────────────────────────── Main Function ───────────────────────────

def run_searchlight(
    fmri_file: Path,
    roi_expression: str,
    label_key: str,
    output_dir: Path,
    *,
    average_samples: bool = True,
    radius: float = 3.0,
    n_jobs: int = 1,
    cv: int = 3,
    estimator: str = "svc",
    roi_definitions: Path | None = None,
    # Label sourcing:
    valence_key: str | None = None,             # read valence from BData key -> (unpleasant/neutral/pleasant)
    valence_dir: Path | None = None,            # read valence from 0001.mat ... files in this directory
    valence_mat_keys: list[str] | None = None,  # candidate variable names inside .mat files
    # Binary filtering (subset to exactly two classes, in labels form):
    positive_classes: tuple[str, ...] = ("unpleasant",),
    negative_classes: tuple[str, ...] = ("neutral",),
    enable_binary_filter: bool = True,
) -> Path:
    """
    Execute a searchlight analysis and return the path to the NIfTI map.

    Parameters
    ----------
    fmri_file : Path
        BData HDF5 path (e.g., fmri_Subject1_for_tutorial.h5).
    roi_expression : str
        bdpy ROI expression, or a name present in `roi_definitions`.
    label_key : str
        Label key in BData (used if valence_key/valence_dir are None).
    output_dir : Path
        Directory where outputs will be saved.
    average_samples : bool
        Average repeats sharing the same label (after filtering).
    radius : float
        Searchlight radius (in voxels).
    n_jobs : int
        Parallel workers for SearchLight.
    cv : int
        Cross-validation folds.
    estimator : str
        Currently only 'svc' is supported (linear SVC).
    roi_definitions : Path | None
        YAML/JSON mapping of {name: bdpy_select_expression}.
    valence_key : str | None
        If provided, pull valence vector from BData and map to three categories.
    valence_dir : Path | None
        If provided, read valence from .mat files (0001.mat…).
    valence_mat_keys : list[str] | None
        Candidate variable names to read inside .mat files (first match wins).
    positive_classes : tuple[str, ...]
        Which labels are considered positive (map to y=1).
    negative_classes : tuple[str, ...]
        Which labels are considered negative (map to y=0).
    enable_binary_filter : bool
        If True, subset to only (positive ∪ negative) and run binary classification.

    Returns
    -------
    Path
        Path to the saved NIfTI scores image (searchlight_scores.nii.gz).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    bdata = bdpy.BData(str(fmri_file))

    if roi_definitions is not None:
        roi_table = _load_roi_definition(roi_definitions)
        roi_expression = roi_table.get(roi_expression, roi_expression)

    matrix, selector = _load_roi_data(bdata, roi_expression)

    # ----- Load labels (strings: unpleasant/neutral/pleasant or your own) -----
    if valence_dir is not None:
        sample_ids = _infer_sample_ids_from_bdata(bdata)
        n_samples = matrix.shape[0]
        valence = _load_valence_from_mat_dir(
            valence_dir=Path(valence_dir),
            n_samples=n_samples,
            sample_ids=sample_ids,
            mat_key_candidates=valence_mat_keys,
        )
        labels = _labels_from_valence(valence)
    elif valence_key is not None:
        valence = _load_bdata_array(bdata, valence_key)
        labels = _labels_from_valence(valence)
    else:
        raw_labels = _load_bdata_array(bdata, label_key)
        labels = _labels_from_valence(raw_labels) if label_key.lower() == "valence" else raw_labels
    # -------------------------------------------------------------------------

    # ----- Optional: keep ONLY two classes (e.g., unpleasant vs neutral) -----
    if enable_binary_filter:
        matrix, labels, kept_mask = _keep_only_classes(
            matrix, labels, classes_to_keep=positive_classes + negative_classes
        )
    # -------------------------------------------------------------------------

    # Average repeats AFTER filtering so only kept classes contribute
    matrix, labels = _aggregate_samples(matrix, labels, average_samples)

    # Final targets
    if enable_binary_filter:
        y = _labels_to_binary(labels, positive=positive_classes, negative=negative_classes)
        class_names = ["+".join(negative_classes), "+".join(positive_classes)]
    else:
        y, class_names = _encode_labels(labels)

    if len(np.unique(y)) < 2:
        raise ValueError("Searchlight requires at least two classes after filtering/averaging.")

    # Build 4D NIfTI from the (possibly filtered/averaged) samples
    metadata = {k: np.asarray(bdata.get_metadata(k)) for k in ["voxel_i", "voxel_j", "voxel_k"]}
    fmri_img, mask_img = _create_nifti(matrix, selector, metadata)

    # Estimator
    if estimator == "svc":
        base_estimator = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    else:
        raise ValueError(f"Unsupported estimator '{estimator}'.")

    # Searchlight
    searchlight = SearchLight(
        mask_img=mask_img,
        radius=radius,
        estimator=base_estimator,
        n_jobs=n_jobs,
        verbose=1,
        cv=cv,
        scoring="accuracy",
    )
    searchlight.fit(fmri_img, y)

    # Save outputs
    scores_img = nib.Nifti1Image(searchlight.scores_, fmri_img.affine)
    nii_path = output_dir / "searchlight_scores.nii.gz"
    scores_img.to_filename(str(nii_path))

    # Quick-look visualization
    threshold = np.nanpercentile(searchlight.scores_, 95)
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = None
    display = plotting.plot_glass_brain(
        scores_img,
        colorbar=True,
        threshold=threshold,
        plot_abs=False,
        title=f"Searchlight accuracy ({class_names[0]} vs {class_names[1]})" if enable_binary_filter
              else "Searchlight accuracy",
    )
    figure_path = output_dir / "searchlight_glass_brain.png"
    display.savefig(str(figure_path))
    display.close()

    print("Saved searchlight scores to", nii_path)
    print("Saved glass brain visualisation to", figure_path)
    if enable_binary_filter:
        print("Classes:", class_names, "| y: 0 =", class_names[0], ", 1 =", class_names[1])
    else:
        print("Classes:", class_names)

    return nii_path


nii_path = run_searchlight(
    fmri_file=Path("/red/ruogu.fang/leem.s/EmotionVideo/EmotionVideoNeuralRepresentationPython/data/fmri/paper2020/fmri_Subject1.h5"),
    roi_expression="VC",
    label_key="stimulus_name",
    output_dir=Path("./outputs/searchlight"),
    average_samples=False,
    radius=5.0,
    n_jobs=-1,
    cv=5,
    estimator="svc",
    # If your valence comes from .mat files:
    valence_dir=Path("/red/ruogu.fang/leem.s/EmotionVideo/EmotionVideoNeuralRepresentationPython/data/features/amt/mean_score_concat/dimension/"),
    valence_mat_keys=["mean_score", "valence", "score", "rating", "amt_score"],
    # Binary filtering to unpleasant vs neutral:
    positive_classes=("unpleasant",),
    negative_classes=("neutral",),
    enable_binary_filter=True,
)

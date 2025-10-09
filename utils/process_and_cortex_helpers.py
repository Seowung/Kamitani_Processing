
"""
Utilities to save block-averaged fMRI outputs and visualize them with pycortex.
Works with data processed by a NiftiMasker (nilearn) in voxel-mask space.
"""

from pathlib import Path
import numpy as np
import nibabel as nb
import json

def save_processed_outputs(S_z, ids_df, masker, outdir, run_metadata=None):
    """
    Save processed samples and metadata in a portable format.

    Parameters
    ----------
    S_z : np.ndarray, shape (n_samples, n_voxels)
        Block-averaged, z-scored samples across videos (rows) by voxels (columns).
    ids_df : pandas.DataFrame
        Must contain at least a column 'stim_id' and 'onset' (float). Will be saved to CSV.
    masker : nilearn.maskers.NiftiMasker
        Fitted masker that provides mask_img, affine, shape.
    outdir : str or Path
        Directory to write outputs into.
    run_metadata : dict or None
        Optional dictionary with extra info (TR, subject, task, etc.), saved as JSON.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mask_img = masker.mask_img_
    if mask_img is None:
        # nilearn >= 0.10 stores mask_img in attribute mask_img
        mask_img = masker.mask_img

    mask_img = nb.load(mask_img) if isinstance(mask_img, (str, Path)) else mask_img
    mask_data = mask_img.get_fdata().astype(bool)
    affine = mask_img.affine
    shape3d = mask_img.shape

    # Map masker voxel ordering back to 3D ijk indices
    # nilearn NiftiMasker uses mask_data.ravel(order="F")? In practice, it uses mask_data.ravel(order="C")
    # We'll reconstruct by querying the masker._masker_.mask_img_ if available; otherwise assume C-order.
    # Robust approach: compute indices of nonzero voxels in C-order:
    ijk = np.vstack(np.nonzero(mask_data)).astype(np.int32)  # (3, n_vox)
    # But ensure the column order matches masker's transform order. For NiftiMasker default, order is C-major flattening.
    # We'll verify with a synthetic vector roundtrip if needed. For portability we store ijk and leave plotting to use it consistently.

    np.save(outdir / "voxel_ijk.npy", ijk)                       # (3, n_vox)
    np.save(outdir / "mask_shape.npy", np.array(shape3d, dtype=np.int32))
    np.save(outdir / "affine.npy", affine)

    # Save the samples matrix
    np.save(outdir / "samples_z.npy", S_z.astype(np.float32))

    # Save ids
    ids_path = outdir / "sample_ids.csv"
    ids_df.to_csv(ids_path, index=False)

    # Save a small JSON sidecar
    sidecar = {
        "n_samples": int(S_z.shape[0]),
        "n_voxels": int(S_z.shape[1]),
        "mask_shape": [int(x) for x in shape3d],
        "affine": affine.tolist(),
        "voxel_ijk_file": "voxel_ijk.npy",
        "samples_file": "samples_z.npy",
        "ids_file": ids_path.name,
    }
    if run_metadata:
        sidecar["meta"] = run_metadata
    with open(outdir / "metadata.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    return {
        "samples_file": str(outdir / "samples_z.npy"),
        "ids_file": str(ids_path),
        "voxel_ijk_file": str(outdir / "voxel_ijk.npy"),
        "affine_file": str(outdir / "affine.npy"),
        "mask_shape_file": str(outdir / "mask_shape.npy"),
        "sidecar": str(outdir / "metadata.json"),
    }


def vector_to_volume(sample_vec, mask_shape, voxel_ijk, fill=np.nan):
    """
    Convert a voxel vector (length n_vox) into a 3D volume (mask_shape).

    Parameters
    ----------
    sample_vec : (n_vox,) array
    mask_shape : (3,) tuple/list
    voxel_ijk : (3, n_vox) int array of i,j,k indices (x=i, y=j, z=k)
    fill : value to prefill the 3D grid (default NaN)

    Returns
    -------
    vol : 3D numpy array with shape mask_shape, filled at ijk locations.
    """
    vol = np.full(mask_shape, fill, dtype=np.float32)
    i, j, k = voxel_ijk  # ijk indexing
    # Tutorial used [k, j, i] when assigning to voxel_vol.data
    vol[k, j, i] = sample_vec.astype(np.float32)
    return vol


def make_cortex_volume(sample_vec, subject, xfmname, voxel_ijk, affine, mask_shape, cmap="RdGy_r"):
    """
    Build a pycortex Volume from a voxel vector using ijk mapping.

    Returns
    -------
    cortex.Volume
    """
    import cortex  # local import to avoid hard dep

    vol = vector_to_volume(sample_vec, tuple(mask_shape), voxel_ijk, fill=np.nan)
    # Create an empty pycortex volume and assign into .data like in the tutorial
    voxel_vol = cortex.Volume.empty(subject, xfmname, value=np.nan, cmap=cmap)
    # Ensure shape compatibility
    if voxel_vol.data.shape != vol.shape:
        raise ValueError(f"Shape mismatch: pycortex volume grid {voxel_vol.data.shape} vs mask {vol.shape}")
    voxel_vol.data[...] = vol  # assign full 3D grid
    return voxel_vol


def save_cortex_png(voxel_vol, out_png, overlay_svg=None, with_colorbar=True, with_curvature=True, labelsize=24, linewidth=2):
    """
    Save a flatmap PNG using cortex.quickshow.
    """
    import cortex
    import matplotlib.pyplot as plt

    fig = cortex.quickshow(
        voxel_vol,
        with_colorbar=with_colorbar,
        with_curvature=with_curvature,
        overlay_file=overlay_svg,
        labelsize=labelsize,
        linewidth=linewidth,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)

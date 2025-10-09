
#!/usr/bin/env python3
"""
Export processed block-averaged samples to disk and optionally make a pycortex flatmap PNG.

Usage:
  python export_processed_and_visualize.py \
      --samples_npy /path/to/samples_z.npy \
      --ids_csv /path/to/sample_ids.csv \
      --voxel_ijk_npy /path/to/voxel_ijk.npy \
      --affine_npy /path/to/affine.npy \
      --mask_shape_npy /path/to/mask_shape.npy \
      --subject Subject1 \
      --xfm voxel2mm_fmriprep \
      --sample_index 0 \
      --out_png /tmp/sample0.png \
      [--overlay_svg ./data/pycortex/Subject1/overlays.svg]

You can also call the `save_processed_outputs` function directly if you have S_z, ids_df, and a fitted masker.
"""
import argparse
import numpy as np
from process_and_cortex_helpers import make_cortex_volume, save_cortex_png

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_npy", required=True)
    ap.add_argument("--ids_csv", required=True)
    ap.add_argument("--voxel_ijk_npy", required=True)
    ap.add_argument("--affine_npy", required=True)
    ap.add_argument("--mask_shape_npy", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--xfm", default="voxel2mm_fmriprep")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--overlay_svg", default=None)
    args = ap.parse_args()

    samples = np.load(args.samples_npy)
    voxel_ijk = np.load(args.voxel_ijk_npy)
    affine = np.load(args.affine_npy)
    mask_shape = np.load(args.mask_shape_npy)

    vec = samples[args.sample_index]
    vol = make_cortex_volume(vec, args.subject, args.xfm, voxel_ijk, affine, mask_shape)
    png = save_cortex_png(vol, args.out_png, overlay_svg=args.overlay_svg)
    print("Saved:", png)

if __name__ == "__main__":
    main()

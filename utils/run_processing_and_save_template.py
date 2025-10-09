
"""
Template: run your block-averaging pipeline and save outputs for pycortex visualization.
- Import your existing `process_subject` function (from the code you already have).
- Update BIDS paths and subject IDs.
"""
from pathlib import Path
import pandas as pd
# from your_module_where_process_subject_lives import process_subject  # <-- replace with your import
from process_and_cortex_helpers import save_processed_outputs

# Example stub showing expected usage
def main():
    # Example: result from your existing code
    # S_z, ids, masker = process_subject(sub="01", ses=None)
    raise SystemExit("Replace this stub: import your `process_subject` and call save_processed_outputs(S_z, ids, masker, outdir).")

    outdir = Path("./outputs/processed/Subject1")
    outdir.mkdir(parents=True, exist_ok=True)

    # Suppose you have S_z, ids (pandas DataFrame), and masker
    files = save_processed_outputs(S_z, ids, masker, outdir, run_metadata={"subject": "Subject1", "xfm": "voxel2mm_fmriprep"})
    print("Wrote: ", files)

if __name__ == "__main__":
    main()

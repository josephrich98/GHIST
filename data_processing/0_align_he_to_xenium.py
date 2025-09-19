import tifffile
import numpy as np
import pandas as pd
from skimage.transform import estimate_transform, warp
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp_he_img",
        default="Xenium_V1_Human_Colon_Cancer_P5_CRC_Add_on_FFPE_he_image.ome.tif",
        type=str,
        help="corresponding full resolution H&E image file path",
    )
    parser.add_argument(
        "--fp_alignment_csv",
        default="Xenium_V1_Human_Colon_Cancer_P5_CRC_Add_on_FFPE_he_imagealignment.csv",
        type=str,
        help="alignment CSV file path (control points between H&E and Xenium)",
    )
    parser.add_argument(
        "--n_processes",
        default=16,
        type=int,
        help="num cpus to use, or None for all cpus - 1",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)
    parser.add_argument(
        "--del_intm_files",
        default=True,
        type=bool,
        help="delete intermediate saved files",
    )

    config = parser.parse_args()

    dir_output = config.dir_output
    os.makedirs(dir_output, exist_ok=True)

    # Output path
    out_path = os.path.join(dir_output, "he_image_aligned.tif")
    out_path_lossy = os.path.join(dir_output, "he_image_aligned_jpg.tif")

    # --- Load image ---
    print(f"Loading H&E image from {config.fp_he_img}...")
    he_img = tifffile.imread(config.fp_he_img)

    # --- Load alignment CSV (landmark correspondences) ---
    df = pd.read_csv(config.fp_alignment_csv)

    # Moving = alignment coords (Xenium), Fixed = H&E coords
    src = df[["alignmentX", "alignmentY"]].values  # from Xenium
    dst = df[["fixedX", "fixedY"]].values          # to H&E

    # --- Estimate affine transform ---
    print("Estimating affine transform...")
    tform = estimate_transform("affine", src, dst)

    # --- Apply transform ---
    print("Applying transform to H&E image...")
    aligned = warp(
        he_img,
        tform.inverse,
        preserve_range=True,
        order=1,
        output_dtype=he_img.dtype,
    )

    # --- Save aligned TIFF ---
    print(f"Saving aligned H&E image to {out_path}...")
    tifffile.imwrite(out_path, aligned, compression="zlib")

    print(f"Also saving a lossy compressed version to {out_path_lossy}...")
    # tifffile.imwrite(out_path_lossy, aligned, compression="jpeg", jpeg_quality=90)

    print(f"Aligned H&E saved to {out_path}")

    # Optional: quick check
    plt.figure(figsize=(8, 8))
    plt.imshow(aligned)
    plt.axis("off")
    plt.show()

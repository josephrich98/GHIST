import tifffile
import numpy as np
import pandas as pd
from skimage.transform import AffineTransform, warp
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
        help="corresponding full resolution H&E image file path",
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

    # Paths
    out_path = os.path.join(dir_output, "he_image_aligned.tif")

    # --- Load image ---
    he_img = tifffile.imread(config.fp_he_img)

    # --- Load transform from CSV ---
    df = pd.read_csv(config.fp_alignment_csv)
    # Usually 2 rows (x and y), with "scale", "shear", "rotate", "translate"
    # 10x convention: affine transform in 3x3 homogeneous matrix
    M = df.values
    if M.shape == (3, 3):
        affine_matrix = M
    else:
        raise ValueError("Unexpected alignment CSV format")

    # --- Apply transform ---
    tform = AffineTransform(matrix=affine_matrix)
    aligned = warp(he_img, tform.inverse, preserve_range=True).astype(he_img.dtype)

    # --- Save aligned TIFF ---
    tifffile.imwrite(out_path, aligned)

    print(f"Aligned H&E saved to {out_path}")

    # Optional: quick check
    plt.figure(figsize=(8, 8))
    plt.imshow(aligned)
    plt.axis("off")
    plt.show()

import tifffile
import numpy as np
import pandas as pd
from skimage.transform import estimate_transform, warp, rescale
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
        "--downsample_factor",
        default=0.1,
        type=float,
        help="factor to downsample before warp (0.1 = 10% of original size)",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)

    config = parser.parse_args()

    dir_output = config.dir_output
    os.makedirs(dir_output, exist_ok=True)

    out_path = os.path.join(dir_output, "he_image_aligned_zlib.tif")

    # --- Load image ---
    print(f"Loading H&E image from {config.fp_he_img}...")
    he_img = tifffile.imread(config.fp_he_img)

    # --- Downsample image first ---
    if config.downsample_factor < 1.0:
        print(f"Downsampling H&E before warp by factor {config.downsample_factor}...")
        he_lowres = rescale(
            he_img,
            config.downsample_factor,
            preserve_range=True,
            anti_aliasing=True,
            channel_axis=None
        ).astype(he_img.dtype)
    else:
        he_lowres = he_img

    # --- Load alignment CSV ---
    df = pd.read_csv(config.fp_alignment_csv)

    # Scale coordinates if downsampling
    src = df[["alignmentX", "alignmentY"]].values
    dst = df[["fixedX", "fixedY"]].values
    if config.downsample_factor < 1.0:
        dst = dst * config.downsample_factor

    # --- Estimate affine transform ---
    print("Estimating affine transform...")
    tform = estimate_transform("affine", src, dst)

    # --- Apply transform ---
    print("Applying transform to H&E image...")
    aligned = warp(
        he_lowres,
        tform.inverse,
        preserve_range=True,
        order=1,
        output_dtype=he_img.dtype,
    )

    # Cast back to the original dtype
    aligned = aligned.astype(he_img.dtype)

    # --- Save compressed TIFF ---
    print(f"Saving aligned H&E to {out_path}...")
    tifffile.imwrite(out_path, aligned, compression="zlib")

    print("Done!")

    # Optional: quick check
    plt.figure(figsize=(8, 8))
    plt.imshow(aligned, cmap="gray" if aligned.ndim == 2 else None)
    plt.axis("off")
    plt.show()

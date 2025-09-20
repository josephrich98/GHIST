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
        "--fp_he_aligned_img",
        default="Xenium_V1_Human_Colon_Cancer_P5_CRC_Add_on_FFPE_he_image_aligned.ome.tif",
        type=str,
        help="output aligned H&E image file path",
    )
    parser.add_argument(
        "--xenium_img",
        default="Xenium_V1_Human_Colon_Cancer_P5_CRC_Add_on_FFPE_xenium_image.ome.tif",
        type=str,
        help="corresponding full resolution Xenium image file path",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)

    config = parser.parse_args()
    fp_dapi = config.xenium_img
    fp_he_orig = config.fp_he_img
    fp_he_aligned = config.fp_he_aligned_img

    # Define crop coordinates (y1:y2, x1:x2) -- adjust as needed
    y1, y2 = 10000, 12000
    x1, x2 = 15000, 17000

    def load_crop(fp, y1, y2, x1, x2):
        with tifffile.TiffFile(fp) as tif:
            store = tif.series[0].aszarr()
            return np.asarray(store[y1:y2, x1:x2])

    # Load crops
    dapi_crop = load_crop(fp_dapi, y1, y2, x1, x2)
    he_orig_crop = load_crop(fp_he_orig, y1, y2, x1, x2)
    he_aligned_crop = load_crop(fp_he_aligned, y1, y2, x1, x2)

    # Overlay helper
    def overlay(base, overlay, alpha_base=0.5, alpha_overlay=0.5):
        plt.imshow(base, cmap="gray", alpha=alpha_base)
        plt.imshow(overlay, alpha=alpha_overlay)
        plt.axis("off")

    # Plot comparison
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    overlay(dapi_crop, he_orig_crop)
    plt.title("Xenium morphology + Original H&E")

    plt.subplot(1,2,2)
    overlay(dapi_crop, he_aligned_crop)
    plt.title("Xenium morphology + Registered H&E")

    plt.tight_layout()
    plt.show()
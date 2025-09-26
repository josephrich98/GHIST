import os
import tifffile
import numpy as np
from skimage.transform import resize
import argparse

def get_file_size_mb(fp):
    return os.path.getsize(fp) / (1024**2)  # bytes to MB

def downsample_to_size(input_fp, output_fp, target_size_mb, compression="zlib"):
    # Load image
    img = tifffile.imread(input_fp)
    orig_shape = img.shape
    orig_dtype = img.dtype

    # Estimate current size (raw on disk)
    current_size_mb = get_file_size_mb(input_fp)
    print(f"Current file size: {current_size_mb:.2f} MB, shape: {orig_shape}, dtype: {orig_dtype}")

    if target_size_mb > current_size_mb:
        raise ValueError("Target size must be smaller than current size.")

    # Compute scale factor
    scale_factor = (target_size_mb / current_size_mb) ** (1/2)  # sqrt for 2D scaling
    print(f"Target size: {target_size_mb:.2f} MB, estimated scale factor: {scale_factor:.3f}")

    # New dimensions
    new_y = max(1, int(orig_shape[0] * scale_factor))
    new_x = max(1, int(orig_shape[1] * scale_factor))
    if img.ndim == 3:   # RGB
        new_shape = (new_y, new_x, orig_shape[2])
    else:               # grayscale
        new_shape = (new_y, new_x)

    # Downsample
    print(f"Resizing to {new_shape} ...")
    downsampled = resize(
        img,
        new_shape,
        preserve_range=True,
        anti_aliasing=True
    )

    # Convert to uint8 if original was uint16 (cuts size in half again)
    if img.dtype == np.uint16:
        downsampled = (downsampled / 256).astype(np.uint8)
        print("Converted to uint8 for smaller size.")
    else:
        downsampled = downsampled.astype(img.dtype)

    # Save
    tifffile.imwrite(output_fp, downsampled, compression=compression)
    final_size_mb = get_file_size_mb(output_fp)
    print(f"Saved {output_fp}, final size: {final_size_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", required=True, help="Input TIFF path")
    parser.add_argument("--output_fp", required=True, help="Output TIFF path")
    parser.add_argument("--target_mb", type=float, required=True, help="Desired size in MB")
    args = parser.parse_args()

    downsample_to_size(args.input_fp, args.output_fp, args.target_mb)

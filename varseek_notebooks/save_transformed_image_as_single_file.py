import argparse
import tifffile
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(
        description="Extract a representative RGB slice from a multi-slice multi-channel TIFF."
    )
    parser.add_argument(
        "--input_fp", type=str, required=True,
        help="Path to the input TIFF file."
    )
    parser.add_argument(
        "--output_fp", type=str, default=None,
        help="Path to the output TIFF file. Default: input filename with '_sliceN.tif'."
    )
    parser.add_argument(
        "--slice_number", type=int, default=None,
        help="1-indexed slice number to extract. If not provided, picks the slice with the highest total intensity."
    )
    parser.add_argument(
        "--num_channels", type=int, default=None,
        help="Number of channels to extract. Default: use the second dimension of the TIFF array."
    )

    args = parser.parse_args()

    # Load image
    multi = tifffile.imread(args.input_fp)

    # Infer number of channels
    if args.num_channels is None:
        args.num_channels = multi.shape[1]

    # Determine slice if not given
    slice_number = args.slice_number
    if slice_number is None:
        highest_total = 0
        best_idx = None
        for i in range(multi.shape[0]):  # loop over z slices
            total = 0
            for j in range(multi.shape[1]):  # loop over channels
                total += np.sum(multi[i][j])
            if total > highest_total:
                highest_total = total
                best_idx = i
        if highest_total == 0:
            raise ValueError("All slices are empty; cannot determine slice_number")
        slice_number = best_idx + 1  # convert to 1-indexed
        print(f"[INFO] Selected slice {slice_number} (highest total intensity = {highest_total})")

    # Extract channels for this slice
    slices = [multi[i] for i in range(args.num_channels)]
    rgb = np.stack(slices, axis=-1)

    from pdb import set_trace; set_trace()

    # Determine output path
    if args.output_fp is None:
        base, ext = os.path.splitext(args.input_fp)
        args.output_fp = f"{base}_slice{slice_number}.{ext}"

    # Save RGB image
    tifffile.imwrite(args.output_fp, rgb)
    print(f"[INFO] Saved RGB slice to {args.output_fp}")


if __name__ == "__main__":
    main()
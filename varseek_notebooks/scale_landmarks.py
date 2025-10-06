import pandas as pd
import argparse

def scale_landmarks(
    fp_in, fp_out,
    original_x_mvg, original_y_mvg, final_x_mvg, final_y_mvg,
    original_x_fix, original_y_fix, final_x_fix, final_y_fix
):
    # Load the landmark file (no header, since BigWarp exports without one)
    df = pd.read_csv(fp_in, header=None)

    # Assign column names for clarity
    df.columns = [
        "point_id", "enabled",
        "x_moving", "y_moving",
        "x_fixed", "y_fixed"
    ]

    # Compute scaling factors
    scale_x_mvg = final_x_mvg / original_x_mvg
    scale_y_mvg = final_y_mvg / original_y_mvg
    scale_x_fix = final_x_fix / original_x_fix
    scale_y_fix = final_y_fix / original_y_fix

    print(f"Scaling moving (x,y): {scale_x_mvg:.4f}, {scale_y_mvg:.4f}")
    print(f"Scaling fixed  (x,y): {scale_x_fix:.4f}, {scale_y_fix:.4f}")

    # Apply scaling
    df["x_moving"] = df["x_moving"].astype(float) * scale_x_mvg
    df["y_moving"] = df["y_moving"].astype(float) * scale_y_mvg
    df["x_fixed"]  = df["x_fixed"].astype(float) * scale_x_fix
    df["y_fixed"]  = df["y_fixed"].astype(float) * scale_y_fix

    # Save back to CSV in the same format
    df.to_csv(fp_out, header=False, index=False, quoting=1)  # quoting=1 forces double quotes

    print(f"Saved scaled landmarks → {fp_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_in", required=True, help="Input BigWarp landmark CSV")
    parser.add_argument("--fp_out", required=True, help="Output scaled landmark CSV")

    parser.add_argument("--original_x_mvg", type=float, required=True)
    parser.add_argument("--original_y_mvg", type=float, required=True)
    parser.add_argument("--final_x_mvg", type=float, required=True)
    parser.add_argument("--final_y_mvg", type=float, required=True)

    parser.add_argument("--original_x_fix", type=float, required=True)
    parser.add_argument("--original_y_fix", type=float, required=True)
    parser.add_argument("--final_x_fix", type=float, required=True)
    parser.add_argument("--final_y_fix", type=float, required=True)

    args = parser.parse_args()

    scale_landmarks(
        args.fp_in, args.fp_out,
        args.original_x_mvg, args.original_y_mvg, args.final_x_mvg, args.final_y_mvg,
        args.original_x_fix, args.original_y_fix, args.final_x_fix, args.final_y_fix
    )
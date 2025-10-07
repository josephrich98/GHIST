import pandas as pd
import csv
import argparse

parser = argparse.ArgumentParser(
    description="Convert 10x Genomics landmarks to BigWarp format."
)
parser.add_argument(
    "-i", "--input_csv", type=str, required=True,
    help="Path to the input CSV file containing 10x landmarks."
)
parser.add_argument(
    "-o", "--output_csv", type=str, required=True,
    help="Path to the output CSV file in BigWarp format."
)

args = parser.parse_args()

# Read the input CSV
df = pd.read_csv(args.input_csv)

# Build the rows
out_rows = []
for i, row in enumerate(df.itertuples(index=False), start=1):
    out_rows.append([
        f"Pt-{i}",
        "True",
        f"{row.alignmentX}",
        f"{row.alignmentY}",
        f"{row.fixedX}",
        f"{row.fixedY}"
    ])

# Write output without headers, fully quoted
with open(args.output_csv, "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerows(out_rows)

print(f"Converted {len(out_rows)} landmarks from {args.input_csv} to {args.output_csv}")
import pandas as pd
import pathlib
import natsort
import argparse
import os


def main(config):

    dir_output = config.dir_output
    os.makedirs(dir_output, exist_ok=True)

    fp_transcripts = config.fp_transcripts
    min_qv = config.min_qv

    gene_col = config.gene_col

    transcripts_to_filter = [
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword_",
        "BLANK_",
        "Blank-",
        "NegPrb",
        "Unassigned",
    ]

    print("Loading transcripts file")
    if pathlib.Path(fp_transcripts).suffixes[-1] == ".gz":
        if ".tsv" in fp_transcripts:
            df = pd.read_csv(fp_transcripts, sep="\t", compression="gzip")
        else:
            df = pd.read_csv(fp_transcripts, compression="gzip")
    else:
        if ".tsv" in fp_transcripts:
            df = pd.read_csv(fp_transcripts, sep="\t")
        else:
            df = pd.read_csv(fp_transcripts)
    print(df.head())

    print("Filtering transcripts")
    if "qv" in df.columns:
        df = df[
            (df["qv"] >= min_qv)
            & (~df[gene_col].str.startswith(tuple(transcripts_to_filter)))
        ]
    else:
        df = df[(~df[gene_col].str.startswith(tuple(transcripts_to_filter)))]

    gene_names = df[gene_col].unique()
    print("%d unique genes" % len(gene_names))
    gene_names = natsort.natsorted(gene_names)

    fp_out = os.path.join(dir_output, config.fp_out)
    with open(fp_out, "w") as f:
        for line in gene_names:
            f.write(f"{line}\n")

    print(f"Saved to {fp_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp_transcripts",
        default="transcripts.csv.gz",
        type=str,
        help="transcripts file path",
    )
    parser.add_argument(
        "--min_qv",
        default=20,
        type=int,
        help="min qv of transcripts to keep",
    )
    parser.add_argument(
        "--gene_col",
        default="feature_name",
        type=str,
        help="col name of genes in the transcripts file",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)
    parser.add_argument(
        "--fp_out",
        default="genes.txt",
        type=str,
        help="output file name",
    )

    config = parser.parse_args()
    main(config)

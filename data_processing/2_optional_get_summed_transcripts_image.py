import tifffile
import pandas as pd
import numpy as np
import os
import natsort
import argparse
from tqdm import tqdm
import multiprocessing as mp
import shutil


def process_gene_chunk(gene_chunk, df, map_height, map_width, dir_output):
    for i_fe, fe in enumerate(gene_chunk):
        df_fe = df.loc[df["feature_name"] == fe]
        map_fe = np.zeros((map_height, map_width))

        for idx in df_fe.index:
            idx_x = np.round(df.iloc[idx]["x_location"]).astype(int)
            idx_y = np.round(df.iloc[idx]["y_location"]).astype(int)

            map_fe[idx_y, idx_x] += 1

        tifffile.imwrite(
            dir_output + "/" + fe + ".tif",
            map_fe.astype(np.uint8),
            photometric="minisblack",
        )


def get_n_processes(config_n_processes):
    if config_n_processes is None:
        n_processes = mp.cpu_count() - 2
    else:
        n_processes = (
            config_n_processes
            if config_n_processes <= mp.cpu_count()
            else mp.cpu_count()
        )
    return n_processes


def main(config):
    """
    Generates summed transcript expression image from transcripts.csv.gz, which contains transcript data with locations:

        "transcript_id","cell_id","overlaps_nucleus","feature_name","x_location","y_location","z_location","qv"
        281474976710656,565,0,"SEC11C",4.395842,328.66647,12.019493,18.66248
        281474976710657,540,0,"NegControlCodeword_0502",5.074415,236.96484,7.6085105,18.634956
        281474976710658,562,0,"SEC11C",4.702023,322.79715,12.289083,18.66248
        281474976710659,271,0,"DAPK3",4.9066014,581.42865,11.222615,20.821745
        281474976710660,291,0,"TCIM",5.6606994,720.85175,9.265523,18.017488
        281474976710661,297,0,"TCIM",5.899098,748.5928,9.818688,18.017488

    For verifying alignment of transcripts with H&E image

    """

    dir_output = config.dir_output
    os.makedirs(dir_output, exist_ok=True)

    dir_output_imgs = dir_output + "/transcript_imgs"
    os.makedirs(dir_output_imgs, exist_ok=True)

    fp_out_filtered = os.path.join(dir_output, config.fp_out_filtered)

    if not os.path.exists(fp_out_filtered):
        print("Loading transcripts file")
        df = pd.read_csv(config.fp_transcripts, compression="gzip")
        print(df.head())

        print("Filtering transcripts")
        df = df[
            (df["qv"] >= config.min_qv)
            & (~df["feature_name"].str.startswith("NegControlProbe_"))
            & (~df["feature_name"].str.startswith("antisense_"))
            & (~df["feature_name"].str.startswith("NegControlCodeword_"))
            & (~df["feature_name"].str.startswith("BLANK_"))
        ]

        df.reset_index(inplace=True, drop=True)
        print("Finished filtering")
        print("...")

        df.to_csv(fp_out_filtered)
    else:
        print("Loading filtered transcripts")
        df = pd.read_csv(fp_out_filtered, index_col=0)

    print(df.head())

    gene_names = df["feature_name"].unique()
    print("%d unique genes" % len(gene_names))

    gene_names = natsort.natsorted(gene_names)
    gene_names = gene_names

    # with open(dir_output+'/'+config.fp_out_gene_names, 'w') as f:
    #     for line in gene_names:
    #         f.write(f"{line}\n")

    map_width = int(np.ceil(df["x_location"].max())) + 1
    map_height = int(np.ceil(df["y_location"].max())) + 1
    print("W: %d, H: %d" % (map_width, map_height))

    print("Converting to maps")

    n_processes = get_n_processes(config.n_processes)

    gene_names_chunks = np.array_split(gene_names, n_processes)
    processes = []

    for gene_chunk in gene_names_chunks:
        p = mp.Process(
            target=process_gene_chunk,
            args=(gene_chunk, df, map_height, map_width, dir_output_imgs),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Combine channel-wise
    summed_all_genes = np.zeros((map_height, map_width), dtype=np.uint16)

    for i_fe, fe in enumerate(tqdm(gene_names)):
        gene_img = tifffile.imread(dir_output_imgs + "/" + fe + ".tif") 
        summed_all_genes += gene_img

    # Sum across all markers
    tifffile.imwrite(
        dir_output + "/" + config.fp_out_image,
        summed_all_genes.astype(np.uint16),
        photometric="minisblack",
    )

    print("Saved maps")

    if config.del_intm_files:
        print("Deleting intermediate files")
        # input("Press Enter to continue or CTRL+C to quit...")

        shutil.rmtree(dir_output_imgs)
        if os.path.exists(fp_out_filtered):
            os.remove(fp_out_filtered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp_transcripts",
        default="transcripts.csv.gz",
        type=str,
        help="transcripts file path",
    )
    parser.add_argument(
        "--min_qv", default=20, type=int, help="min qv of transcripts to keep"
    )
    parser.add_argument(
        "--n_processes",
        default=24,
        type=int,
        help="num cpus to use, or None for all cpus - 1",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)
    parser.add_argument(
        "--fp_out_filtered", default="transcripts_filtered.csv", type=str
    )
    parser.add_argument("--fp_out_image", default="summed_transcripts.tif", type=str)
    parser.add_argument(
        "--del_intm_files",
        default=True,
        type=bool,
        help="delete intermediate saved files",
    )

    config = parser.parse_args()
    main(config)

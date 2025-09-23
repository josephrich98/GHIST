import os
import shutil
from tqdm import tqdm
import pandas as pd
import argparse
import sys
import varseek as vk
import anndata as ad
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fastqs", required=True, nargs='+', help="path to directory of fastq files or list of fastq files")
    parser.add_argument("--technology", required=True, help="technology for varseek")
    parser.add_argument("--dir_output", default="data_processing", type=str)
    parser.add_argument("--vk_ref_dir", default="vk_ref_out", type=str, help="directory to store varseek reference files, default: vk_ref_out")
    parser.add_argument("--vk_count_dir", default="vk_count_dir", type=str, help="directory to store variant data, default: vk_count_dir")
    parser.add_argument("--index", help="path to varseek index, default: <vk_ref_dir>/cosmic_cmc_index.idx")
    parser.add_argument("--t2g", help="path to varseek t2g file, default: <vk_ref_dir>/cosmic_cmc_t2g.txt")
    parser.add_argument("-k", "--k", default=51, type=int, help="k for varseek count")
    parser.add_argument("--min_counts", default=3, type=int, help="min counts for varseek count")
    parser.add_argument("--disable_use_binary_matrix", action="store_false", help="whether to use binary matrix for varseek count (default: use binary matrix)")
    parser.add_argument("--disable_drop_empty_columns", action="store_false", help="whether to drop empty columns for varseek count (default: drop empty columns)")
    parser.add_argument("--n_processes", default=24, type=int, help="max number of cpus to use")

    config = parser.parse_args()

    os.makedirs(config.dir_output, exist_ok=True)
    
    if config.index is None:
        config.index = os.path.join(config.vk_ref_dir, "cosmic_cmc_index.idx")
    if config.t2g is None:
        config.t2g = os.path.join(config.vk_ref_dir, "cosmic_cmc_t2g.txt")

    if not os.path.exists(config.index) or not os.path.exists(config.t2g):
        raise ValueError(f"Please download the varseek index/t2g from Box, or make it with `vk ref --index {config.index} --t2g {config.t2g} -v cosmic_cmc -s cdna --dlist_reference_source t2t`")
    
    if os.path.isdir(config.fastqs[0]):
        if len(config.fastqs) > 1:
            raise ValueError("If --fastqs is a directory, only provide one directory")
        config.fastqs = config.fastqs[0]
        
    if isinstance(config.fastqs, str):
        if not os.path.exists(config.fastqs) or len(os.listdir(config.fastqs)) == 0:
            raise ValueError(f"Please make sure the fastq files are in {config.fastqs}")
    elif isinstance(config.fastqs, list):
        for fastq in config.fastqs:
            if not os.path.exists(fastq):
                raise ValueError(f"Please make sure the fastq file {fastq} exists")
    
    if os.path.exists(config.vk_count_dir) and len(os.listdir(config.vk_count_dir)) > 0:
        print(f"vk count output directory {config.vk_count_dir} already exists and is not empty, skipping vk count")
    else:
        print("Running vk count")
        vk_count_output_dict = vk.count(
            fastqs=config.fastqs,
            index=config.index,
            t2g=config.t2g,
            technology=config.technology,
            k=config.k,
            out=config.vk_count_dir,
            threads=config.n_processes,
            min_counts=config.min_counts,
            use_binary_matrix=config.disable_use_binary_matrix,
            drop_empty_columns=config.disable_drop_empty_columns,
        )
    
    # save to CSV
    # Load AnnData
    adata = ad.read_h5ad(vk_count_output_dict["adata_path"])

    # Convert to DataFrame
    df = pd.DataFrame(
        adata.X.toarray() if not isinstance(adata.X, pd.DataFrame) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    # Write to CSV
    final_csv_path = os.path.join(config.dir_output, "variant_matrix.csv")
    df.to_csv(final_csv_path, index=True, header=True)

    # Now remove the index name and column name in the CSV
    with open(final_csv_path, "r") as f:
        lines = f.readlines()

    # Replace the first line so the first cell is blank
    lines[0] = "," + lines[0]

    with open(final_csv_path, "w") as f:
        f.writelines(lines)

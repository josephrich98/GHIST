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

    parser.add_argument("--dir_output", required=True, type=str)
    parser.add_argument("--technology", required=True, help="technology for varseek")
    parser.add_argument("--variant_data_dir", default=None, type=str, help="directory to store variant data, default: <dir_output>/variant_data")
    parser.add_argument("--fastqs_dir", default=None, type=str, help="directory containing fastq files, default: <variant_data_dir>/fastqs")
    parser.add_argument("--vk_ref_dir", default=None, type=str, help="directory to store varseek reference files, default: <variant_data_dir>/vk_ref_out")
    parser.add_argument("--index", help="path to varseek index, default: <vk_ref_dir>/cosmic_cmc_index.idx")
    parser.add_argument("--t2g", help="path to varseek t2g file, default: <vk_ref_dir>/cosmic_cmc_t2g.txt")
    parser.add_argument("-k", "--k", default=51, type=int, help="k for varseek count")
    parser.add_argument("--min_counts", default=3, type=int, help="min counts for varseek count")
    parser.add_argument("--disable_use_binary_matrix", action="store_false", help="whether to use binary matrix for varseek count (default: use binary matrix)")
    parser.add_argument("--disable_drop_empty_columns", action="store_false", help="whether to drop empty columns for varseek count (default: drop empty columns)")
    parser.add_argument("--n_processes", default=24, type=int, help="max number of cpus to use")

    config = parser.parse_args()

    os.makedirs(config.dir_output, exist_ok=True)

    if config.variant_data_dir is None:
        variant_data_dir = os.path.join(config.dir_output, "variant_data")
    os.makedirs(variant_data_dir, exist_ok=True)
    
    if config.vk_ref_dir is None:
        vk_ref_dir = os.path.join(variant_data_dir, "vk_ref_out")
    if config.index is None:
        index = os.path.join(vk_ref_dir, "cosmic_cmc_index.idx")
    if config.t2g is None:
        t2g = os.path.join(vk_ref_dir, "cosmic_cmc_t2g.txt")

    if not os.path.exists(index) or not os.path.exists(t2g):
        raise ValueError(f"Please download the varseek index/t2g from Box, or make it with `vk ref --index {index} --t2g {t2g} -v cosmic_cmc -s cdna --dlist_reference_source t2t`")
    
    if config.fastqs_dir is None:
        fastqs_dir = os.path.join(variant_data_dir, "")  #!!!!!! ensure that I am using the correct fastqs here
    if not os.path.exists(fastqs_dir) or len(os.listdir(fastqs_dir)) == 0:
        raise ValueError(f"Please make sure the fastq files are in {fastqs_dir}")
    
    vk_count_out = os.path.join(variant_data_dir, "vk_count_out")
    if os.path.exists(vk_count_out) and len(os.listdir(vk_count_out)) > 0:
        print(f"vk count output directory {vk_count_out} already exists and is not empty, skipping vk count")
    else:
        print("Running vk count")
        vk_count_output_dict = vk.count(
            fastqs_dir,
            index=index,
            t2g=t2g,
            technology=config.technology,
            k=config.k,
            out=vk_count_out,
            threads=config.n_processes,
            min_counts=config.min_counts,
            use_binary_matrix=config.use_binary_matrix,
            drop_empty_columns=config.drop_empty_columns,
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
    with open("matrix.csv", "r") as f:
        lines = f.readlines()

    # Replace the first line so the first cell is blank
    lines[0] = "," + lines[0]

    with open("matrix.csv", "w") as f:
        f.writelines(lines)

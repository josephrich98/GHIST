import os
import shutil
from tqdm import tqdm
import pandas as pd
import argparse
import sys
import varseek as vk
import anndata as ad
import pandas as pd
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fastqs", required=True, nargs='+', help="path to directory of fastq files or list of fastq files")
    parser.add_argument("--technology", required=True, help="technology for varseek")
    parser.add_argument("--cell_barcode_to_id", default=None, help="path to csv file with cell barcode to cell id mapping, with columns 'cell_barcode' and 'cell_id'")
    parser.add_argument("--cell_gene_matrix_filtered_path", default=None, help="path to filtered cell by gene matrix csv file")
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

    cell_barcode_to_id_path = os.path.join(config.dir_output, config.cell_barcode_to_id)
    if not os.path.exists(cell_barcode_to_id_path):
        print("Warning: barcode to cell id mapping file not found, will keep original barcodes")
    
    cell_gene_matrix_filtered_path = os.path.join(config.dir_output, config.cell_gene_matrix_filtered_path)
    if not os.path.exists(cell_gene_matrix_filtered_path):
        print("Warning: filtered cell by gene matrix file not found, will not filter cells based on this file")
    
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
        )
    
    # save to CSV
    adata = ad.read_h5ad(vk_count_output_dict["adata_path"])

    print(f"Initial shape of variant matrix: {adata.shape}")

    if not config.disable_use_binary_matrix:
        adata.X = (adata.X > 0).astype(int)
    
    if not config.drop_empty_columns:
        adata = adata[:, np.array((adata.X != 0).sum(axis=0)).flatten() > 0]

    if os.path.exists(cell_barcode_to_id_path):
        mapping_df = pd.read_csv(cell_barcode_to_id_path, index_col=0)
        id_map = mapping_df["cell_id_num"].to_dict()
        adata.obs.index = adata.obs.index.map(id_map)
        adata_shape_before = adata.shape
        adata = adata[adata.obs.index.notna()].copy()
        adata.obs.index = adata.obs.index.astype(int)
        adata_shape_after = adata.shape
        if adata_shape_after == 0:
            raise ValueError("No cells left after mapping barcodes to cell ids, please check the barcode to cell id mapping file")
        if adata_shape_before[0] != adata_shape_after[0]:
            print(f"Warning: {adata_shape_before[0] - adata_shape_after[0]} cells were removed because they were not found in the barcode to cell id mapping file")
    
    if os.path.exists(cell_gene_matrix_filtered_path):
        df = pd.read_csv(cell_gene_matrix_filtered_path, usecols=[0])
        df.columns = ["id"]

        # keep only rows in adata that are in df
        adata_shape_before = adata.shape
        adata = adata[adata.obs.index.isin(df["id"])].copy()
        adata_shape_after = adata.shape
        if adata_shape_after == 0:
            raise ValueError("No cells left after filtering based on filtered cell by gene matrix, please check the filtered cell by gene matrix file")
        if adata_shape_before[0] != adata_shape_after[0]:
            print(f"Warning: {adata_shape_before[0] - adata_shape_after[0]} cells were removed because they were not found in the filtered cell by gene matrix file")
    
    print(f"Final shape of variant matrix: {adata.shape}")

    for percentage in [1, 5, 10, 25, 50]:
        gene_mask = (np.count_nonzero(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, axis=0) >= percentage/100 * adata.n_obs)
        print(gene_mask.sum(), f"genes have nonzero counts in at least {percentage}% of cells")

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

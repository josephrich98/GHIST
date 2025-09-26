# %%
import os
import subprocess
import pandas as pd
import varseek as vk

# %%
w = 37
k = 41
threads = 8
filters=(
    "alignment_to_reference:is_not_true",
    "num_distinct_triplets:greater_than=5",  # filters out VCRSs in with 5 or fewer distinct triplets
    "longest_homopolymer_length:less_or_equal=10",  # filters out VCRSs with a homopolymer length greater than 10 bp
)
dlist_reference_source = "t2t"

data_dir = "/mnt/gpussd2/jrich/Desktop/GHIST/data_varseek"    # os.path.join(os.path.dirname(os.getcwd()), "data_varseek")
vk_ref_dir = os.path.join(data_dir, "vk_ref_out")
reference_dir = os.path.join(data_dir, "reference")

reference_genomes_dir = os.path.join(reference_dir, "reference_genomes")

tcga_dir = os.path.join(reference_dir, "tcga")
tcga_maf_path = os.path.join(tcga_dir, "mc3.v0.2.8.PUBLIC.maf")
tcga_csv_path = os.path.join(tcga_dir, "tcga_mc3.csv")
vk_ref_out_tcga = os.path.join(vk_ref_dir, "tcga")
tcga_index = os.path.join(vk_ref_dir, "tcga_index.idx")
tcga_t2g = os.path.join(vk_ref_dir, "tcga_t2g.txt")
tcga_reference_genome_cdna_path = os.path.join(reference_genomes_dir, "ensembl_grch37_release70_dir", "Homo_sapiens.GRCh37.70.cdna.all.fa")

cosmic_dir = os.path.join(reference_dir, "cosmic")
vk_ref_out_cosmic = os.path.join(vk_ref_dir, "cosmic_cmc")
cosmic_index = os.path.join(vk_ref_dir, f"cosmic_cmc_k{k}_index.idx")
cosmic_t2g = os.path.join(vk_ref_dir, f"cosmic_cmc_k{k}_t2g.txt")

dbsnp_dir = os.path.join(reference_dir, "dbsnp")
dbsnp_vcf_path = os.path.join(dbsnp_dir, "GCF_000001405.40.gz")
vk_ref_out_dbsnp = os.path.join(vk_ref_dir, "dbsnp")
dbsnp_index = os.path.join(vk_ref_dir, "dbsnp_index.idx")
dbsnp_t2g = os.path.join(vk_ref_dir, "dbsnp_t2g.txt")
dbsnp_reference_genome_path = os.path.join(reference_genomes_dir, "ncbi_grch38", "GCF_000001405.40_GRCh38.p14_genomic.fna")

# %% [markdown]
# ## COSMIC CMC

# %%
# vk.ref(
#     variants="cosmic_cmc",
#     sequences="cdna",
#     w=w,
#     k=k,
#     dlist_reference_source=dlist_reference_source,
#     filters=filters,
#     out=vk_ref_out_cosmic,
#     threads=threads,
#     index_out=cosmic_index,
#     t2g_out=cosmic_t2g,
# )

# %% [markdown]
# ## TCGA

# %%
# TCGA
ensembl_grch37_release70_cdna_url = "https://ftp.ensembl.org/pub/release-70/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh37.70.cdna.all.fa.gz"
tcga_url = "https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc"

if not os.path.exists(tcga_reference_genome_cdna_path):
    os.makedirs(os.path.dirname(tcga_reference_genome_cdna_path), exist_ok=True)
    subprocess.run(f"wget -O {tcga_reference_genome_cdna_path}.gz {ensembl_grch37_release70_cdna_url}", shell=True, check=True)
    subprocess.run(f"gunzip {tcga_reference_genome_cdna_path}.gz", shell=True, check=True)

if not os.path.exists(tcga_maf_path):
    os.makedirs(os.path.dirname(tcga_maf_path), exist_ok=True)
    subprocess.run(f"wget -O {tcga_maf_path}.gz {tcga_url}", shell=True, check=True)
    subprocess.run(f"gunzip {tcga_maf_path}.gz", shell=True, check=True)

# mutation_pattern = r"(?:c|g)\.([0-9_\-\+\*\(\)\?]+)([a-zA-Z>]+)"

def cds_to_cdna(tcga_df, var_column="HGVSc"):    
    tcga_df = tcga_df.copy()
    length_prefilter = len(tcga_df)

    if "variant_type" not in tcga_df.columns:
        vk.utils.add_variant_type(tcga_df, var_column=var_column)  # Add variant_type column based on var_column

    tcga_df[["nucleotide_positions", "actual_variant"]] = tcga_df[var_column].str.extract(vk.constants.mutation_pattern)  # Extract nucleotide positions and mutation info from Mutation CDS
    tcga_df = tcga_df.dropna(subset=["nucleotide_positions", "actual_variant"])  # Filter out tcga_df that did not match the re
    tcga_df["nucleotide_positions_cdna"] = tcga_df["cDNA_position"].astype(str).str.replace("-", "_")

    tcga_df["HGVSc_cdna"] = "c." + tcga_df["nucleotide_positions_cdna"] + tcga_df["actual_variant"]

    length_postfilter = len(tcga_df)
    print(f"Filtered out {length_prefilter - length_postfilter} rows that did not match the regex pattern.")
    print(f"Remaining rows: {length_postfilter}")

    return tcga_df

if not os.path.exists(tcga_csv_path):
    tcga_df = pd.read_csv(tcga_maf_path, sep="\t", comment="#", low_memory=False)
    tcga_df = cds_to_cdna(tcga_df, var_column="HGVSc")
    tcga_df[["Transcript_ID", "HGVSc_cdna", "HGVSc", "ENSP", "HGVSp", "Gene", "Hugo_Symbol", "Chromosome", "Start_Position", "End_Position", "Strand", "Reference_Allele", "Tumor_Seq_Allele1", "Tumor_Seq_Allele2", "dbSNP_RS", "all_effects"]].to_csv(tcga_csv_path, index=False)

# %%
vk.ref(
    variants=tcga_csv_path,
    sequences=tcga_reference_genome_cdna_path,
    w=w,
    k=k,
    filters=filters,
    dlist_reference_source=dlist_reference_source,
    seq_id_column="Transcript_ID",
    var_column="HGVSc_cdna",
    out=vk_ref_out_tcga,
    threads=threads,
    index_out=tcga_index,
    t2g_out=tcga_t2g,
)

# %% [markdown]
# ## dbSNP

# # %%
# dbsnp_reference_genome_url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/annotation_releases/9606/110/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
# dbsnp_vcf_url = "https://ftp.ncbi.nih.gov/snp/latest_release/VCF/GCF_000001405.40.gz"  # .40 = GRCh38; .25 = GRCh37
# dbsnp_tbi_url = "https://ftp.ncbi.nih.gov/snp/latest_release/VCF/GCF_000001405.40.gz.tbi"

# if not os.path.exists(dbsnp_reference_genome_path):
#     !wget -O {dbsnp_reference_genome_path}.gz {dbsnp_reference_genome_url}
#     !gunzip {dbsnp_reference_genome_path}.gz

# if not os.path.exists(dbsnp_vcf_path):
#     !wget -O {dbsnp_vcf_path} {dbsnp_vcf_url}

# # %%
# vk.ref(
#     variants=dbsnp_vcf_path,
#     sequences=dbsnp_reference_genome_path,
#     w=w,
#     k=k,
#     filters=filters,
#     dlist_reference_source=dlist_reference_source,
#     out=vk_ref_out_dbsnp,
#     threads=threads,
#     index_out=dbsnp_index,
#     t2g_out=dbsnp_t2g,
# )



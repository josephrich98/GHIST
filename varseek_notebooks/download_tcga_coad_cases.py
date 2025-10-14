import os
import shutil
import requests
import numpy as np
import pandas as pd
import subprocess
import logging
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_CASES = "https://api.gdc.cancer.gov/cases"
GDC_MANIFEST = "https://api.gdc.cancer.gov/manifest"

def get_file_map(project, experimental_strategy, data_format=None, data_type=None):
    filters = [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "=", "content": {"field": "experimental_strategy", "value": [experimental_strategy]}}
    ]
    if data_format:
        filters.append({"op": "=", "content": {"field": "data_format", "value": [data_format]}})
    if data_type:
        filters.append({"op": "=", "content": {"field": "data_type", "value": [data_type]}})

    payload = {
        "filters": {"op": "and", "content": filters},
        "fields": "cases.submitter_id,id,file_name,data_type,data_format,experimental_strategy",
        "format": "JSON",
        "size": 10000
    }

    r = requests.post(GDC_FILES, json=payload)
    hits = r.json()["data"]["hits"]

    result = {}
    for h in hits:
        for case in h.get("cases", []):
            result.setdefault(case["submitter_id"], []).append(h["id"])
    return result

def make_metadata_df(tcga_project, metadata_csv_out, number_cases=None):
    if os.path.isfile(metadata_csv_out):
        logger.info(f"Loading existing metadata CSV from {metadata_csv_out}")
        df = pd.read_csv(metadata_csv_out)
        return df

    # --- Run each ---
    logger.info(f"Querying GDC for project {tcga_project}...")
    svs_map = get_file_map(tcga_project, experimental_strategy="Tissue Slide", data_format="SVS")
    wxs_bam_map = get_file_map(tcga_project, experimental_strategy="WXS", data_format="BAM")
    wxs_vcf_map = get_file_map(tcga_project, experimental_strategy="WXS", data_format="VCF")
    rnaseq_bam_map = get_file_map(tcga_project, experimental_strategy="RNA-Seq", data_format="BAM")
    rnaseq_tsv_map = get_file_map(tcga_project, experimental_strategy="RNA-Seq", data_format="TSV")

    # Get union of all patient IDs
    all_cases = set(svs_map) | set(wxs_bam_map) | set(wxs_vcf_map) | set(rnaseq_bam_map) | set(rnaseq_tsv_map)

    records = []
    for cid in tqdm(sorted(all_cases), desc="Building metadata DataFrame"):
        records.append({
            "patient_id": cid,
            "tissue_slide_id": ";".join(svs_map.get(cid, [])),
            "wxs_bam_id": ";".join(wxs_bam_map.get(cid, [])),
            "wxs_vcf_id": ";".join(wxs_vcf_map.get(cid, [])),
            "rnaseq_bam_id": ";".join(rnaseq_bam_map.get(cid, [])),
            "rnaseq_vcf_id": ";".join(rnaseq_tsv_map.get(cid, []))
        })

    df = pd.DataFrame.from_records(records)
    df = df.fillna("")
    logger.info(f"Found {len(df)} total patients in {tcga_project} with any data type.")

    if number_cases:
        df = df.head(number_cases)
        logger.info(f"Limiting to first {number_cases} cases for testing.")
    
    logger.info("Querying GDC for clinical and demographic data...")
    for case_id in tqdm(df["patient_id"], desc="Fetching clinical data"):
        payload = {
            "filters": {
                "op": "=",
                "content": {"field": "cases.submitter_id", "value": [case_id]}
            },
            "fields": "submitter_id,demographic.gender,demographic.race,demographic.ethnicity,diagnoses.age_at_diagnosis,diagnoses.tissue_or_organ_of_origin,diagnoses.tumor_grade,diagnoses.prior_malignancy,project.project_id",
            "format": "JSON",
            "size": 1
        }

        r = requests.post(GDC_CASES, json=payload)
        data = r.json()["data"]["hits"]

        if not data:
            print(f"No data found for {case_id}")
            continue

        case = data[0]
        diag = case.get("diagnoses", [{}])[0]
        demo = case.get("demographic", {})
        patient_dict = {
            "case_uuid": case.get("id"),
            "project_id": case.get("project", {}).get("project_id"),
            "tissue_or_organ_of_origin": diag.get("tissue_or_organ_of_origin"),
            "gender": demo.get("gender"),
            "race": demo.get("race"),
            "ethnicity": demo.get("ethnicity"),
            "age_at_diagnosis": diag.get("age_at_diagnosis"),
            "vital_status": diag.get("vital_status"),
            "tumor_grade": diag.get("tumor_grade"),
            "prior_malignancy": diag.get("prior_malignancy")
        }
        df.loc[df["patient_id"] == case_id, list(patient_dict.keys())] = list(patient_dict.values())

    if metadata_csv_out:
        os.makedirs(os.path.dirname(metadata_csv_out) or ".", exist_ok=True)
        df.to_csv(metadata_csv_out, index=False)
        logger.info(f"Saved metadata CSV as {metadata_csv_out}")

    return df

def filter_df(df, columns_to_enforce, manifest_out, number_cases=None):
    has_all_columns = df[columns_to_enforce].replace("", np.nan).notna().all(axis=1)
    for col in columns_to_enforce:
        n_col = (df[col] != "").sum()
        logger.info(f"{col}: {n_col} patients")

    df_all = df[has_all_columns].copy()
    logger.info(f"All required columns: {len(df_all)} / {len(df)}")

    if number_cases:
        df_all = df_all.head(number_cases)
        logger.info(f"Limiting to first {number_cases} cases for testing.")

    # Collect all UUIDs from the 3 relevant columns
    uuids = set()

    for col in columns_to_enforce:
        for ids in df_all[col]:
            uuids.update([x for x in ids.split(";") if x])

    logger.info(f"{len(uuids)} total unique files.")

    manifest_payload = {"ids": list(uuids)}
    r = requests.post(GDC_MANIFEST, json=manifest_payload)
    os.makedirs(os.path.dirname(manifest_out) or ".", exist_ok=True)
    with open(manifest_out, "w") as f:
        f.write(r.text)
    logger.info(f"Saved manifest as {manifest_out}")

    return df_all

def download_with_gdc_client(manifest_out, gdc_client, gdc_token, download_dir, threads=2, dry_run=True, organized_dir=None):
    if os.path.isdir(download_dir) and os.listdir(download_dir):
        logger.warning(f"Download directory {download_dir} already exists and is not empty. Skipping download.")
        return
    if organized_dir and os.path.isdir(organized_dir) and os.listdir(organized_dir):
        logger.warning(f"Download directory {organized_dir} already exists and is not empty. Skipping download.")
        return

    gdc_command = f"{gdc_client} download -m {manifest_out} -d {download_dir} -n {threads}"
    if gdc_token:
        gdc_command += f" --token-file {gdc_token}"
    logger.info(f"Download command: {gdc_command}")
    if not dry_run:
        logger.info("Downloading files from GDC...")
        os.makedirs(download_dir, exist_ok=True)
        subprocess.run(gdc_command, shell=True, check=True)
        logger.info("✅ Download complete.")
    else:
        logger.info("Dry run specified; skipping download.")

def move_file_by_uuid(uuid, patient_id, download_dir, subdir):
    # Find the folder created by gdc-client for that UUID
    src_dir = os.path.join(download_dir, uuid)
    if not os.path.isdir(src_dir):
        return False

    # Find the actual file (should be exactly one per folder)
    files = os.listdir(src_dir)
    if not files:
        return False

    # Destination folder
    dest_dir = os.path.join(organized_dir, patient_id, subdir, uuid)
    os.makedirs(dest_dir, exist_ok=True)

    for fname in files:
        shutil.move(os.path.join(src_dir, fname),
                    os.path.join(dest_dir, fname))
    return True

def organize_files(df_all, download_dir, organized_dir, columns_to_enforce):
    if os.path.isdir(organized_dir) and os.listdir(organized_dir):
        logger.warning(f"Organized directory {organized_dir} already exists and is not empty. Skipping reorganization.")
        return
    
    os.makedirs(organized_dir, exist_ok=True)

    # Move files
    for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Organizing files by patient"):
        pid = row["patient_id"]

        for col in columns_to_enforce:
            for uuid in row[col].split(";"):
                if uuid:
                    file_moved = move_file_by_uuid(uuid, pid, download_dir, col[:-3])  # remove the "_id"

    logger.info("✅ Files reorganized into folders by patient ID.")

def parse_args():
    # --- Parse args ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcga_project", type=str, default="TCGA-COAD", help="GDC project ID to query")
    parser.add_argument("--columns_to_enforce", type=str, nargs="+", default=["tissue_slide_id", "wxs_bam_id", "rnaseq_bam_id"], help="Columns that must be non-empty to include a patient for download")
    parser.add_argument("-t", "--threads", type=int, default=2, help="threads for GDC download")
    parser.add_argument("-n", "--number_cases", type=int, default=None, help="Limit to first N cases (for testing)")
    parser.add_argument("--metadata_csv_out", type=str, default="tcga_coad_multimodal.csv", help="Output CSV file listing TCGA-COAD cases with multiple data types")
    parser.add_argument("--manifest_out", type=str, default="gdc_manifest.txt", help="Path to GDC manifest file")
    parser.add_argument("--download_dir", type=str, default="gdc_download_2025", help="Directory where gdc-client downloads files")
    parser.add_argument("--organized_dir", type=str, default="organized_by_patient", help="Directory to organize files by patient ID")
    parser.add_argument("--gdc_client", type=str, default="gdc-client", help="Path to gdc-client executable")
    parser.add_argument("--gdc_token", type=str, default=None, help="Path to GDC token file (if needed)")
    parser.add_argument("--dry_run", action="store_true", help="If set, only create manifest and do not download or reorganize files")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    manifest_out = args.manifest_out
    gdc_client = args.gdc_client
    gdc_token = args.gdc_token
    metadata_csv_out = args.metadata_csv_out
    number_cases = args.number_cases
    dry_run = args.dry_run
    download_dir = args.download_dir
    organized_dir = args.organized_dir
    columns_to_enforce = args.columns_to_enforce
    tcga_project = args.tcga_project.upper()
    threads = args.threads

    df = make_metadata_df(tcga_project, metadata_csv_out, number_cases)
    df_all = filter_df(df, columns_to_enforce, manifest_out, number_cases)
    download_with_gdc_client(manifest_out, gdc_client, gdc_token, download_dir, threads, dry_run, organized_dir)    
    if not dry_run:
        organize_files(df_all, download_dir, organized_dir, columns_to_enforce)

    logger.info("All done!")

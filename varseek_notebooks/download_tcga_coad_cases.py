import os
import shutil
import requests
import pandas as pd
import subprocess
import argparse

# --- Parse args ---
parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=str, default="gdc_manifest_all3.txt", help="Path to GDC manifest file")
parser.add_argument("--gdc_client", type=str, default="gdc-client", help="Path to gdc-client executable")
parser.add_argument("--gdc_token", type=str, default=None, help="Path to GDC token file (if needed)")
parser.add_argument("--output_csv", type=str, default="tcga_coad_multimodal.csv", help="Output CSV file listing TCGA-COAD cases with multiple data types")
parser.add_argument("-n", "--number_cases", type=int, default=None, help="Limit to first N cases (for testing)")
parser.add_argument("--dry_run", action="store_true", help="If set, only create manifest and do not download or reorganize files")
parser.add_argument("--download_dir", type=str, default="gdc_download_2025", help="Directory where gdc-client downloads files")
parser.add_argument("--organized_dir", type=str, default="organized_by_patient", help="Directory to organize files by patient ID")
args = parser.parse_args()

manifest = args.manifest
gdc_client = args.gdc_client
gdc_token = args.gdc_token
output_csv = args.output_csv
number_cases = args.number_cases
dry_run = args.dry_run
download_dir = args.download_dir
organized_dir = args.organized_dir

GDC_FILES = "https://api.gdc.cancer.gov/files"

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

# --- Run each ---
tcga_project = "TCGA-COAD"
svs_map = get_file_map(tcga_project, experimental_strategy="Tissue Slide", data_format="SVS")
wxs_bam_map = get_file_map(tcga_project, experimental_strategy="WXS", data_format="BAM")
wxs_vcf_map = get_file_map(tcga_project, experimental_strategy="WXS", data_format="VCF")
rnaseq_bam_map = get_file_map(tcga_project, experimental_strategy="RNA-Seq", data_format="BAM")
rnaseq_tsv_map = get_file_map(tcga_project, experimental_strategy="RNA-Seq", data_format="TSV")


# Get union of all patient IDs
all_cases = set(svs_map) | set(wxs_bam_map) | set(wxs_vcf_map) | set(rnaseq_bam_map) | set(rnaseq_tsv_map)

records = []
for cid in sorted(all_cases):
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

# Count patients with each modality present
n_tissue = (df["tissue_slide_id"] != "").sum()
n_wxs_bam = (df["wxs_bam_id"] != "").sum()
n_rnaseq_bam = (df["rnaseq_bam_id"] != "").sum()

# Count patients with all three non-empty
has_all_three = ((df["tissue_slide_id"] != "") &
               (df["wxs_bam_id"] != "") &
               (df["rnaseq_bam_id"] != ""))

df_all = df[has_all_three].copy()
print(f"Tissue Slide: {n_tissue}")
print(f"WXS BAM: {n_wxs_bam}")
print(f"RNA-Seq BAM: {n_rnaseq_bam}")
print(f"All three modalities: {len(df_all)} / {len(df)}")

if number_cases:
    df_all = df_all.head(number_cases)
    print(f"Limiting to first {number_cases} cases for testing.")

# Collect all UUIDs from the 3 relevant columns
uuids = set()

for col in ["tissue_slide_id", "wxs_bam_id", "rnaseq_bam_id"]:
    for ids in df_all[col]:
        uuids.update([x for x in ids.split(";") if x])

print(f"{len(uuids)} total unique files.")

manifest_payload = {"ids": list(uuids)}
r = requests.post("https://api.gdc.cancer.gov/manifest", json=manifest_payload)
open(manifest, "w").write(r.text)
print(f"Saved manifest as {manifest}")

gdc_command = f"{gdc_client} download -m {manifest} -d gdc_download_2025"
if gdc_token:
    gdc_command += f" --token-file {gdc_token}"
print(f"Download command: {gdc_command}")
if not dry_run:
    print("Downloading files from GDC...")
    subprocess.run(gdc_command, shell=True, check=True)
    print("✅ Download complete.")
else:
    print("Dry run specified; skipping download.")

# # Load your filtered DataFrame (patients with all 3)
# df_all = pd.read_csv(output_csv)
# df_all = df_all.fillna("")

os.makedirs(organized_dir, exist_ok=True)

def move_file_by_uuid(uuid, patient_id, subdir):
    # Find the folder created by gdc-client for that UUID
    src_dir = os.path.join(download_dir, uuid)
    if not os.path.isdir(src_dir):
        return False

    # Find the actual file (should be exactly one per folder)
    files = os.listdir(src_dir)
    if not files:
        return False

    # Destination folder
    dest_dir = os.path.join(organized_dir, patient_id, subdir)
    os.makedirs(dest_dir, exist_ok=True)

    for fname in files:
        shutil.move(os.path.join(src_dir, fname),
                    os.path.join(dest_dir, fname))
    return True


# Move files
for _, row in df_all.iterrows():
    pid = row["patient_id"]

    for uuid in row["tissue_slide_id"].split(";"):
        if uuid:
            move_file_by_uuid(uuid, pid, "tissue_slide")

    for uuid in row["wxs_bam_id"].split(";"):
        if uuid:
            move_file_by_uuid(uuid, pid, "wxs")

    for uuid in row["rnaseq_bam_id"].split(";"):
        if uuid:
            move_file_by_uuid(uuid, pid, "rnaseq")

print("✅ Files reorganized into folders by patient ID.")

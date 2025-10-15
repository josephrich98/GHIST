#!/usr/bin/env bash
set -euo pipefail

# Default values
ENSEMBL_RELEASE=111
REFERENCE_SOURCE="cdna"
THREADS=2
OUTPUT="out.vcf.gz"
READ2VCF="read2vcf.sh"
PATTERN=""

# Parse command-line flags
while getopts ":r:b:f:e:s:t:o:v:p:" opt; do
  case $opt in
    r) REFERENCE_FASTA="$OPTARG" ;;
    b) BOWTIE_INDEX_PREFIX="$OPTARG" ;;
    f) FASTQ_DIR="$OPTARG" ;;
    e) ENSEMBL_RELEASE="$OPTARG" ;;
    s) REFERENCE_SOURCE="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    v) READ2VCF="$OPTARG" ;;
    p) PATTERN="$OPTARG" ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "${REFERENCE_FASTA:-}" ] || [ -z "${BOWTIE_INDEX_PREFIX:-}" ] || [ -z "${FASTQ_DIR:-}" ]; then
  echo "Usage: $0 -r <REFERENCE_FASTA> -b <BOWTIE_INDEX_PREFIX> -f <FASTQ_DIR> [-e ENSEMBL_RELEASE] [-s REFERENCE_SOURCE] [-t THREADS] [-o OUTPUT] [-v READ2VCF] [-p PATTERN]"
  echo
  echo "Required arguments:"
  echo "  -r   Path to Ensembl cDNA FASTA file (e.g., Homo_sapiens.GRCh38.cdna.all.fa). Will be downloaded with gget if absent."
  echo "  -b   Prefix for Bowtie2 index output files. Will be created with Bowtie2 if absent."
  echo "  -f   Directory containing FASTQ files"
  echo
  echo "Optional arguments:"
  echo "  -e   Ensembl release number (default: 111)"
  echo "  -s   Reference source (default: cdna)"
  echo "  -t   Number of threads (default: 2)"
  echo "  -o   Output VCF file (default: out.vcf.gz)"
  echo "  -v   Path to read2vcf script (default: read2vcf.sh)"
  echo "  -p   Pattern for FASTQ files (default: None)"
  exit 1
fi

# ===============================
# Ensure reference FASTA exists
# ===============================
if [ ! -f "$REFERENCE_FASTA" ]; then
    echo "Reference FASTA not found. Downloading Ensembl release $ENSEMBL_RELEASE..."
    REFERENCE_DIR=$(dirname "$REFERENCE_FASTA")
    mkdir -p "$REFERENCE_DIR"
    gget ref -r "$ENSEMBL_RELEASE" -d -od "$REFERENCE_DIR" -w "$REFERENCE_SOURCE" human
    echo "Unzipping reference..."
    gunzip -f "${REFERENCE_FASTA}.gz"
else
    echo "Reference FASTA already exists: $REFERENCE_FASTA"
fi

# ===============================
# Build Bowtie2 index
# ===============================
if [ ! -f "$BOWTIE_INDEX_PREFIX.1.bt2" ]; then
    echo "Building Bowtie2 index..."
    mkdir -p "$(dirname "$BOWTIE_INDEX_PREFIX")"
    bowtie2-build --threads "$THREADS" "$REFERENCE_FASTA" "$BOWTIE_INDEX_PREFIX"
    echo "Bowtie2 index built successfully."
else
    echo "Reference Bowtie2 index already exists: $BOWTIE_INDEX_PREFIX"
fi

# ===============================
# Filter FASTQ files if pattern is provided
# ===============================
echo "Searching for FASTQ files in directory: $FASTQ_DIR"
if [ -n "$PATTERN" ]; then
    echo "Filtering FASTQ files in $FASTQ_DIR matching regex: $PATTERN"
    FASTQ_FILES=$(find "$FASTQ_DIR" -type f | grep -E "$PATTERN" | tr '\n' ' ')
else
    echo "Using all FASTQ files in $FASTQ_DIR"
    FASTQ_FILES=$(find "$FASTQ_DIR" -type f \( \
        -name '*.fq' -o -name '*.fastq' -o -name '*.fq.gz' -o -name '*.fastq.gz' \
    \) | tr '\n' ' ')
fi
echo "FASTQ files to be processed: $FASTQ_FILES"

# ===============================
# Run read2vcf
# ===============================
if [ ! -f "$OUTPUT" ]; then
    echo "Running read2vcf..."
    "$READ2VCF" --threads "$THREADS" -f "$REFERENCE_FASTA" -x "$BOWTIE_INDEX_PREFIX" -o "$OUTPUT" $FASTQ_FILES
    echo "read2vcf complete."
else
    echo "Output VCF already exists: $OUTPUT"
fi

echo "All steps completed successfully."

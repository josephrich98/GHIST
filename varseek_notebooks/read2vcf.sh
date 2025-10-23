#!/usr/bin/env bash
set -euo pipefail

check_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required tool '$1' is not installed or not in PATH."
    exit 1
  fi
}

# Defaults
THREADS=1
FASTA_REF=""
STAR_GENOME_DIR="star_genome_index"
OUTPUT="out.vcf.gz"
MIN_COUNTS=3
INCLUDE_EXPR=""
SKIP_INDELS=""
DISABLE_BAQ=""
SPLIT_BAM_BY_N=false
REGIONS_FILE=""
GTF_FILE=""
READ_LENGTH=90  # for STAR
LIMIT_SJDB_INSERT_NSJ=1000000
LIMIT_BAM_SORT_RAM=0
STAR_ALIGNMENT_PREFIX="star_"
ENSEMBL_RELEASE=111
DISABLE_BCFTOOLS_CALL=false
BCFTOOLS_CALL_PRIOR=""
TMP_DIR="/tmp"

# Helper
usage() {
  cat <<EOF
Usage: $0 [options] [input1 [input2 ...]]

Options:
  --threads INT          Number of threads (default: 1)
  -f, --fasta-ref FILE   Reference fasta file (required)
  -x, --star-genome-dir DIR  STAR index directory (required if FASTQ inputs)
  -o, --output FILE      Output VCF (default: out.vcf.gz)
  --min-counts INT       Minimum count threshold for filtering (default: 3)
  -i, --include EXPR     bcftools filter expression (default: 'INFO/AD[1] >= 3')
  -I, --skip-indels      Skip indels in the output
  --disable-baq          Disable BAQ computation in mpileup
  --split-bam-by-n       Split BAM by N in CIGAR (spliced reads)
  --regions FILE         BED file of regions to restrict variant calling to (optional)
  --gtf FILE             genome annotation GTF file (for genomeGenerate)
  --read-length INT      read length
  --star-alignment-prefix PREFIX    prefix for STAR output BAM
  --ensembl-release INT  Ensembl release number (default: 111)
  --disable-bcftools-call  Disable running bcftools call (default: false)
  --bcftools-call-prior FLOAT   Prior for bcftools call (default: none)
  --tmp-dir DIR          Temporary directory (default: /tmp)
  -h, --help             Show this help

Positional arguments:
  Input FASTQ files. If multiple fastqs, separate with commas. If paired fastqs, first pass in R1 files separated by commas, then space, then R2 files separated by commas.

Example:
  $0 --threads 4 -f ref.fa -x star_genome -o out.vcf.gz -i 'FORMAT/AD[1]>=5' aln1.bam aln2.bam
  $0 --threads 4 -f ref.fa -x star_genome -o out.vcf.gz -1 reads_1.fq -2 reads_2.fq
  $0 --threads 4 -f ref.fa -x star_genome -o out.vcf unpaired_reads.fq
EOF
  exit 1
}

# Parse arguments
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS="$2"
      shift 2
      ;;
    -f|--fasta-ref)
      FASTA_REF="$2"
      shift 2
      ;;
    -x|--star-genome-dir)
      STAR_GENOME_DIR="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    --min-counts)
      MIN_COUNTS="$2"
      shift 2
      ;;
    -i|--include)
      INCLUDE_EXPR="$2"
      shift 2
      ;;
    -I|--skip-indels)
      SKIP_INDELS=1
      shift
      ;;
    --disable-baq)
      DISABLE_BAQ=1
      shift
      ;;
    --split-bam-by-n)
      SPLIT_BAM_BY_N=true
      shift
      ;;
    --regions)
      REGIONS_FILE="$2"
      shift 2
      ;;
    --gtf)
      GTF_FILE="$2"
      shift 2
      ;;
    --read-length)
      READ_LENGTH="$2"
      shift 2
      ;;
    --star-alignment-prefix)
      STAR_ALIGNMENT_PREFIX="$2"
      shift 2
      ;;
    --ensembl-release)
      ENSEMBL_RELEASE="$2"
      shift 2
      ;;
    --disable-bcftools-call)
      DISABLE_BCFTOOLS_CALL=true
      shift
      ;;
    --bcftools-call-prior)
      BCFTOOLS_CALL_PRIOR="$2"
      shift 2
      ;;
    --tmp-dir)
      TMP_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Append remaining positional arguments
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

check_tool bcftools
check_tool samtools
check_tool STAR

READ_LENGTH_MINUS_ONE=$((READ_LENGTH - 1))

# Validation
if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "Error: No input files provided."
  usage
fi

if [[ -z "$FASTA_REF" ]]; then
  echo "Error: --fasta-ref is required."
  usage
fi

if [[ "$OUTPUT" != *.vcf* ]]; then
  echo "Error: --output must be a .vcf file path."
  usage
fi

if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || [[ "$THREADS" -lt 1 ]]; then
  echo "Error: --threads must be an integer >= 1."
  usage
fi

# check if REGIONS_FILE exists if provided
if [[ -n "$REGIONS_FILE" && ! -f "$REGIONS_FILE" ]]; then
  echo "Error: Regions file '$REGIONS_FILE' does not exist."
  exit 1
fi

if (( MIN_COUNTS == 0 || MIN_COUNTS == 1 )); then
  echo "Warning: filtering by a minimum count threshold is highly recommended."
  echo "Additionally, indels observed once will not be output regardless of settings (bcftools mpileup behavior)."
fi

FIRST_LIST="${ARGS[0]}"
FIRST_FASTQ_FILE="${FIRST_LIST%%,*}"

# ===============================
# Ensure reference FASTA exists
# ===============================
if [ ! -f "$FASTA_REF" ]; then
    echo "Reference FASTA not found. Downloading Ensembl release $ENSEMBL_RELEASE..."
    REFERENCE_DIR=$(dirname "$FASTA_REF")
    mkdir -p "$REFERENCE_DIR"
    gget ref -r "$ENSEMBL_RELEASE" -d -od "$REFERENCE_DIR" -w dna human
    echo "Unzipping reference..."
    gunzip -f "${FASTA_REF}.gz"
else
    echo "Reference FASTA already exists: $FASTA_REF"
fi

# ===============================
# Ensure reference GTF exists
# ===============================
if [ ! -f "$GTF_FILE" ]; then
    echo "Reference GTF not found. Downloading Ensembl release $ENSEMBL_RELEASE..."
    REFERENCE_DIR=$(dirname "$GTF_FILE")
    mkdir -p "$REFERENCE_DIR"
    gget ref -r "$ENSEMBL_RELEASE" -d -od "$REFERENCE_DIR" -w gtf human
    echo "Unzipping reference..."
    gunzip -f "${GTF_FILE}.gz"
else
    echo "Reference GTF already exists: $GTF_FILE"
fi

# check if STAR genome directory exists
if [[ ! -d "$STAR_GENOME_DIR" || -z "$(ls -A "$STAR_GENOME_DIR" 2>/dev/null)" ]]; then
  echo "Building STAR genome index in '$STAR_GENOME_DIR'..."
  mkdir -p "$STAR_GENOME_DIR"
  STAR \
    --runThreadN "$THREADS" \
    --runMode genomeGenerate \
    --genomeDir "$STAR_GENOME_DIR" \
    --genomeFastaFiles "$FASTA_REF" \
    --sjdbGTFfile "$GTF_FILE" \
    --sjdbOverhang "$READ_LENGTH_MINUS_ONE" \
    $( [ -n "$LIMIT_SJDB_INSERT_NSJ" ] && echo "--limitSjdbInsertNsj $LIMIT_SJDB_INSERT_NSJ" ) \
    $( [ -n "$LIMIT_BAM_SORT_RAM" ] && echo "--limitBAMsortRAM $LIMIT_BAM_SORT_RAM" )
fi

# check if f"{STAR_ALIGNMENT_PREFIX}Aligned.sortedByCoord.out.bam" exists
OUT_BAM="${STAR_ALIGNMENT_PREFIX}Aligned.sortedByCoord.out.bam"
MAKE_BAM_INDEX=false
if [[ ! -f "$OUT_BAM" ]]; then
  echo "Aligning reads to genome..."
  MAKE_BAM_INDEX=true
  STAR \
      --runThreadN "$THREADS" \
      --genomeDir "$STAR_GENOME_DIR" \
      --readFilesIn "${ARGS[@]}" \
      --sjdbOverhang "$READ_LENGTH_MINUS_ONE" \
      --outFileNamePrefix "$STAR_ALIGNMENT_PREFIX" \
      --outSAMtype BAM SortedByCoordinate \
      --outSAMunmapped Within \
      --outSAMmapqUnique 60 \
      --twopassMode Basic \
      --limitSjdbInsertNsj "$LIMIT_SJDB_INSERT_NSJ" \
      --limitBAMsortRAM "$LIMIT_BAM_SORT_RAM" \
      $( [[ "$FIRST_FASTQ_FILE" == *.gz ]] && echo "--readFilesCommand zcat" )
fi

if [[ "$MAKE_BAM_INDEX" == true || ! -f "${OUT_BAM}.bai" ]]; then
  echo "Indexing BAM '$OUT_BAM'..."
  samtools index -@ "$THREADS" "$OUT_BAM"
fi

# Determine output format
if [[ "$OUTPUT" == *.gz ]]; then
  OUTPUT_TYPE="-Oz"
else
  OUTPUT_TYPE="-Ov"
fi


# ===============================
# Split spliced reads (N in CIGAR)
# ===============================
if [[ "${SPLIT_BAM_BY_N:-false}" == true ]]; then
  # SPLIT_BAM="${STAR_ALIGNMENT_PREFIX}split_exons.bam"
  SPLIT_SORTED_BAM="${STAR_ALIGNMENT_PREFIX}split_exons.sorted.bam"

  if [[ ! -f "$SPLIT_SORTED_BAM" ]]; then
    echo "Splitting spliced alignments at introns (CIGAR Ns)..."
    # samtools view -h "$OUT_BAM" \
    # | awk 'BEGIN{OFS="\t"}
    #        /^@/ {print; next}
    #        {
    #          cigar=$6;
    #          if (cigar ~ /N/) {
    #            split(cigar, segs, /[0-9]+N/);
    #            for (i=1; i<=length(segs); i++) {
    #              if (segs[i] != "") {
    #                $6 = segs[i];
    #                print;
    #              }
    #            }
    #          } else {
    #            print;
    #          }
    #        }' \
    # | samtools view -bS -o "$SPLIT_BAM" -
    # samtools sort -@ "$THREADS" -o "$SPLIT_SORTED_BAM" "$SPLIT_BAM"
    # samtools index -@ "$THREADS" "$SPLIT_SORTED_BAM"

    gatk SplitNCigarReads -R "$FASTA_REF" -I "$OUT_BAM" -O "$SPLIT_SORTED_BAM" --tmp-dir "$TMP_DIR" --create-output-bam-index

    echo "Split BAM created and indexed: $SPLIT_SORTED_BAM"
  else
    echo "Split exon BAM already exists: $SPLIT_SORTED_BAM"
  fi

  # Update downstream BAM path to point to the split file
  OUT_BAM="$SPLIT_SORTED_BAM"
fi



echo "Processing with bcftools mpileup + filter..."
# echo "BAMs: ${BAM_FILES[*]}"
# echo "Output: $OUTPUT ($OUTPUT_TYPE)"
# echo "Filter expression: ${INCLUDE_EXPR:-None}"

cmd="bcftools mpileup \
    --threads \"$THREADS\" \
    -A \
    -f \"$FASTA_REF\" \
    -a INFO/AD \
    -Q 0 \
    -d 10000 \
    ${REGIONS_FILE:+-R \"$REGIONS_FILE\"} \
    ${DISABLE_BAQ:+-B} \
    ${SKIP_INDELS:+-I} \
    -Ou \"$OUT_BAM\""

# Conditionally filter
if (( MIN_COUNTS > 1 )) || [[ -n "$INCLUDE_EXPR" ]]; then
    if [[ -n "$INCLUDE_EXPR" ]]; then
        # both INCLUDE_EXPR and MIN_COUNTS filters
        cmd+=, | bcftools filter -i \"(${INCLUDE_EXPR}) && (INFO/AD[1] >= $MIN_COUNTS)\" --threads \"$THREADS\" -Ou"
    else
        # only MIN_COUNTS filter
        cmd+=" | bcftools filter -i \"INFO/AD[1] >= $MIN_COUNTS\" --threads \"$THREADS\" -Ou"
    fi
fi

if [[ "$DISABLE_BCFTOOLS_CALL" == false ]]; then
    # Add bcftools call step
    cmd+=" | bcftools call -m -A -v --threads \"$THREADS\" -Ou"
    if [[ -n "$BCFTOOLS_CALL_PRIOR" ]]; then
        cmd+=" --prior \"$BCFTOOLS_CALL_PRIOR\""
    fi
fi

# Always normalize
cmd+=" | bcftools norm -f \"$FASTA_REF\" -c s -d all -m -any --threads \"$THREADS\""

# Conditionally filter again - I filter before normalization to speed up the process, but I need to filter again after normalization to be accurate (AD values may have changed)
if (( MIN_COUNTS > 1 )) || [[ -n "$INCLUDE_EXPR" ]]; then
    if [[ -n "$INCLUDE_EXPR" ]]; then
        # both INCLUDE_EXPR and MIN_COUNTS filters
        cmd+=" -Ou | bcftools filter -i \"(${INCLUDE_EXPR}) && (INFO/AD[1] >= $MIN_COUNTS)\" --threads \"$THREADS\""
    else
        # only MIN_COUNTS filter
        cmd+=" -Ou | bcftools filter -i \"INFO/AD[1] >= $MIN_COUNTS\" --threads \"$THREADS\""
    fi
fi

# Finally, view/output
if [[ "$DISABLE_BCFTOOLS_CALL" == true ]]; then
    cmd+=" -Ou | bcftools view --threads \"$THREADS\" -e 'ALT=\"<*>\"' -o \"$OUTPUT\" $OUTPUT_TYPE"
else
    cmd+=" -o \"$OUTPUT\" $OUTPUT_TYPE"
fi

# Run it
echo "$cmd"
eval "$cmd"
bcftools index -f --threads "$THREADS" "$OUTPUT"

echo "Program complete. VCF output written to $OUTPUT"

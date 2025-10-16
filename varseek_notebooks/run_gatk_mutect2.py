import argparse
import os
import sys
import shutil
import subprocess
import gzip
import pysam
import pandas as pd

parser = argparse.ArgumentParser(description="Run GATK Mutect2 on a set of reads and report compare with hap.py")

# Paths
parser.add_argument("--synthetic_read_fastq", help="Path to synthetic read FASTQ")
parser.add_argument("--synthetic_read_fastq2", help="Path to synthetic read FASTQ paired")
parser.add_argument("--reference_genome_fasta", help="Path to reference genome fasta")
parser.add_argument("--reference_genome_gtf", help="Path to reference genome GTF")
parser.add_argument("--genomes1000_vcf", default="1000GENOMES-phase_3.vcf", help="Path to 1000 genomes vcf file")
parser.add_argument("--star_genome_dir", default="", help="Path to star_genome_dir")
parser.add_argument("--aligned_and_unmapped_bam", default="", help="Path to aligned_and_unmapped_bam. If not provided, will be created")
parser.add_argument("--out", default="out", help="Path to out folder")
parser.add_argument("--tmp_dir", default=None, help="Path to tmp folder")

# Parameters
parser.add_argument("--threads", default=2, help="Number of threads")
parser.add_argument("--read_length", default=150, help="Read length")
parser.add_argument("--limitSjdbInsertNsj", default='1000000', help="Limit SjdbInsertNsj")
parser.add_argument("--limitBAMsortRAM", default='0', help="limitBAMsortRAM")
parser.add_argument("--apply_mutation_filters", action="store_true", help="Use filtered vcf for accuracy analysis (otherwise use unfiltered)")
parser.add_argument("--disable_tool_default_read_filters", action="store_true", help="Disable tool default read filters")
parser.add_argument("--skip_accuracy_analysis", action="store_true", help="Skip accuracy analysis (beyond simple time and memory benchmarking)")

# Executables
parser.add_argument("--STAR", default="STAR", help="Path to STAR executable")
parser.add_argument("--java", default="java", help="Path to java executable")
parser.add_argument("--picard_jar", default="picard.jar", help="Path to picard.jar executable")
parser.add_argument("--gatk", default="gatk", help="Path to gatk executable")

# Just for accuracy analysis
parser.add_argument("--varseek_denovo_vcf", help="Path to varseek denovo vcf")
parser.add_argument("--happy_env", default=None, help="If using conda, name of conda environment with hap.py installed. If using docker, set to None.")


args = parser.parse_args()

star_genome_dir = args.star_genome_dir if args.star_genome_dir else os.path.join(args.out, "star_genome")
gatk_parent = args.out
tmp_dir = args.tmp_dir
reference_genome_fasta = args.reference_genome_fasta
reference_genome_gtf = args.reference_genome_gtf
genomes1000_vcf = args.genomes1000_vcf
threads = args.threads
read_length_minus_one = int(args.read_length) - 1
apply_mutation_filters = args.apply_mutation_filters
disable_tool_default_read_filters = args.disable_tool_default_read_filters
skip_accuracy_analysis = args.skip_accuracy_analysis
synthetic_read_fastq = args.synthetic_read_fastq
synthetic_read_fastq2 = args.synthetic_read_fastq2
aligned_and_unmapped_bam = args.aligned_and_unmapped_bam
limitSjdbInsertNsj = str(args.limitSjdbInsertNsj)
limitBAMsortRAM = str(args.limitBAMsortRAM)
varseek_denovo_vcf = args.varseek_denovo_vcf
happy_env = args.happy_env

STAR = args.STAR
java = args.java
picard_jar = args.picard_jar
gatk = args.gatk


for name, path in {"STAR": STAR, "java": java, "picard_jar": picard_jar, "gatk": gatk}.items():
    if not os.path.exists(path) and not shutil.which(path):
        raise FileNotFoundError(f"{name} not found or installed properly.")

java_home = os.path.dirname(os.path.dirname(java))

os.environ['JAVA_HOME'] = java_home
os.environ['PATH'] = f"{os.environ['JAVA_HOME']}/bin:" + os.environ['PATH']

os.makedirs(star_genome_dir, exist_ok=True)

alignment_folder = f"{gatk_parent}/alignment"
os.makedirs(alignment_folder, exist_ok=True)

gatk_supporting_files = f"{gatk_parent}/supporting_files"
os.makedirs(gatk_supporting_files, exist_ok=True)

plot_output_folder = f"{gatk_parent}/plots"
os.makedirs(plot_output_folder, exist_ok=True)

out_file_name_prefix = f"{alignment_folder}/sample_"

vcf_folder = f"{gatk_parent}/vcfs"
mutect2_folder = f"{vcf_folder}/mutect2"
mutect2_folder = f"{vcf_folder}/mutect2"

os.makedirs(vcf_folder, exist_ok=True)
os.makedirs(mutect2_folder, exist_ok=True)

aligned_only_bam = f"{alignment_folder}/aligned_only.bam"
unmapped_bam = f"{alignment_folder}/unmapped.bam"
merged_bam = f"{alignment_folder}/merged.bam"

marked_duplicates_bam = f"{alignment_folder}/marked_duplicates.bam"
marked_dup_metrics_txt = f"{alignment_folder}/marked_dup_metrics.txt"

split_n_cigar_reads_bam = f"{alignment_folder}/split_n_cigar_reads.bam"
recal_data_table = f"{alignment_folder}/recal_data.table"
recalibrated_bam = f"{alignment_folder}/recalibrated.bam"
covariates_plot = f"{alignment_folder}/AnalyzeCovariates.pdf"
mutect2_unfiltered_vcf = f"{mutect2_folder}/mutect2_output_unfiltered.g.vcf.gz"

mutect2_filtered_vcf = f"{mutect2_folder}/mutect2_output_filtered.vcf.gz"
mutect2_filtered_applied_vcf = f"{mutect2_folder}/mutect2_output_filtered_applied.vcf.gz"

panel_of_normals_vcf = f"{gatk_supporting_files}/1000g_pon.hg38.vcf.gz"
panel_of_normals_vcf_filtered = f"{gatk_supporting_files}/1000g_pon.hg38_filtered.vcf.gz"
mutect2_unfiltered_vcf = f"{mutect2_folder}/mutect2_output_unfiltered.g.vcf.gz"
mutect2_filtered_vcf = f"{mutect2_folder}/mutect2_output_filtered.vcf.gz"
mutect2_filtered_applied_vcf = f"{mutect2_folder}/mutect2_output_filtered_applied.vcf.gz"

reference_genome_dict = reference_genome_fasta.replace(".fa", ".dict")

# commented out, as these should already be done prior to running this script
genomes1000_vcf_url = "https://ftp.ensembl.org/pub/release-111/variation/vcf/homo_sapiens/1000GENOMES-phase_3.vcf.gz"

download_reference_genome_fasta_command = ["gget", "ref", "-r", "111", "-d", "-od", os.path.dirname(reference_genome_fasta), "-w", "dna", "human"]
unzip_reference_genome_fasta_command = ["gunzip", f"{reference_genome_fasta}.gz"]

download_reference_genome_gtf_command = ["gget", "ref", "-r", "111", "-d", "-od", os.path.dirname(reference_genome_gtf), "-w", "gtf", "human"]
unzip_reference_genome_gtf_command = ["gunzip", f"{reference_genome_gtf}.gz"]

download_1000_genomes_command = ["wget", "-O", f"{genomes1000_vcf}.gz", genomes1000_vcf_url]
unzip_1000_genomes_command = ["gunzip", f"{genomes1000_vcf}.gz"]

if not os.path.exists(reference_genome_fasta):
    subprocess.run(download_reference_genome_fasta_command, check=True)
    subprocess.run(unzip_reference_genome_fasta_command, check=True)

if not os.path.exists(reference_genome_gtf):
    subprocess.run(download_reference_genome_gtf_command, check=True)
    subprocess.run(unzip_reference_genome_gtf_command, check=True)

if not os.path.exists(genomes1000_vcf):
    subprocess.run(download_1000_genomes_command, check=True)
    subprocess.run(unzip_1000_genomes_command, check=True)

#* STAR Build
star_build_command = [
    STAR,
    "--runThreadN", str(threads),
    "--runMode", "genomeGenerate",
    "--genomeDir", star_genome_dir,
    "--genomeFastaFiles", reference_genome_fasta,
    "--sjdbGTFfile", reference_genome_gtf,
    "--sjdbOverhang", str(read_length_minus_one),
]
if len(os.listdir(star_genome_dir)) == 0:
    subprocess.run(star_build_command, check=True)

#* Reference genome index file
if not os.path.exists(f"{reference_genome_fasta}.fai"):
    _ = pysam.faidx(reference_genome_fasta)
# commented out, as these should already be done prior to running this script

#* STAR Alignment
star_align_command = [
    STAR,
    "--runThreadN", str(threads),
    "--genomeDir", star_genome_dir,
    "--readFilesIn", synthetic_read_fastq,
    "--sjdbOverhang", str(read_length_minus_one),
    "--outFileNamePrefix", out_file_name_prefix,
    "--outSAMtype", "BAM", "SortedByCoordinate",
    "--outSAMunmapped", "Within",
    "--outSAMmapqUnique", "60",
    "--twopassMode", "Basic",
    "--limitSjdbInsertNsj", limitSjdbInsertNsj,
    "--limitBAMsortRAM", limitBAMsortRAM,
]
if synthetic_read_fastq.endswith(".gz"):
    star_align_command += ["--readFilesCommand", "zcat"]
if synthetic_read_fastq2 is not None:
    idx = star_align_command.index(synthetic_read_fastq)
    star_align_command.insert(idx + 1, synthetic_read_fastq2)
if not aligned_and_unmapped_bam:
    aligned_and_unmapped_bam = f"{out_file_name_prefix}Aligned.sortedByCoord.out.bam"
if not os.path.exists(aligned_and_unmapped_bam):
    subprocess.run(star_align_command, check=True)

#* FASTQ to SAM
fastq_to_sam_command = [
    java, "-jar", picard_jar, "FastqToSam",
    "-FASTQ", synthetic_read_fastq,
    "-OUTPUT", unmapped_bam,
    "-READ_GROUP_NAME", "rg1",
    "-SAMPLE_NAME", "sample1",
    "-LIBRARY_NAME", "lib1",
    "-PLATFORM_UNIT", "unit1",
    "-PLATFORM", "ILLUMINA",
    "-SEQUENCING_CENTER", "center1"
]
if synthetic_read_fastq2 is not None:
    fastq_to_sam_command += ["-FASTQ2", synthetic_read_fastq2]
if tmp_dir:
    fastq_to_sam_command += ["-TMP_DIR", tmp_dir]
if not os.path.exists(unmapped_bam):
    subprocess.run(fastq_to_sam_command, check=True)

#* CreateSequenceDictionary
create_sequence_dict_command = [
    java, "-jar", picard_jar, "CreateSequenceDictionary",
    "-R", reference_genome_fasta,
    "-O", reference_genome_dict
]
if not os.path.exists(reference_genome_dict):
    subprocess.run(create_sequence_dict_command, check=True)

#* MergeBamAlignment
merge_bam_alignment_command = [
    java, "-jar", picard_jar, "MergeBamAlignment",
    "--ALIGNED_BAM", aligned_and_unmapped_bam,
    "--UNMAPPED_BAM", unmapped_bam,
    "--OUTPUT", merged_bam,
    "--REFERENCE_SEQUENCE", reference_genome_fasta,
    "--SORT_ORDER", "coordinate",
    "--INCLUDE_SECONDARY_ALIGNMENTS", "false",
    "--VALIDATION_STRINGENCY", "SILENT"
]
if tmp_dir:
    merge_bam_alignment_command += ["--TMP_DIR", tmp_dir]
if not os.path.exists(merged_bam):
    subprocess.run(merge_bam_alignment_command, check=True)

#* MarkDuplicates
mark_duplicates_command = [
    java, "-jar", picard_jar, "MarkDuplicates",
    "--INPUT", merged_bam,
    "--OUTPUT", marked_duplicates_bam,
    "--METRICS_FILE", marked_dup_metrics_txt,
    "--CREATE_INDEX", "true",
    "--VALIDATION_STRINGENCY", "SILENT"
]
if tmp_dir:
    mark_duplicates_command += ["--TMP_DIR", tmp_dir]
if not os.path.exists(marked_duplicates_bam):
    subprocess.run(mark_duplicates_command, check=True)

#* SplitNCigarReads
split_n_cigar_reads_command = [
    gatk, "SplitNCigarReads",
    "-R", reference_genome_fasta,
    "-I", marked_duplicates_bam,
    "-O", split_n_cigar_reads_bam
]
if tmp_dir:
    split_n_cigar_reads_command += ["--tmp-dir", tmp_dir]
if not os.path.exists(split_n_cigar_reads_bam):
    subprocess.run(split_n_cigar_reads_command, check=True)

#* IndexFeatureFile
index_feature_file_command = [
    gatk, "IndexFeatureFile",
    "-I", genomes1000_vcf
]
if not os.path.exists(f"{genomes1000_vcf}.idx"):
    subprocess.run(index_feature_file_command, check=True)

#* BaseRecalibrator
base_recalibrator_command = [
    gatk, "BaseRecalibrator",
    "-I", split_n_cigar_reads_bam,
    "-R", reference_genome_fasta,
    "--use-original-qualities",
    "--known-sites", genomes1000_vcf,
    "-O", recal_data_table
]
if tmp_dir:
    base_recalibrator_command += ["--tmp-dir", tmp_dir]
if not os.path.exists(recal_data_table):
    subprocess.run(base_recalibrator_command, check=True)

#* ApplyBQSR
apply_bqsr_command = [
    gatk, "ApplyBQSR",
    "--add-output-sam-program-record",
    "-R", reference_genome_fasta,
    "-I", split_n_cigar_reads_bam,
    "--use-original-qualities",
    "--bqsr-recal-file", recal_data_table,
    "-O", recalibrated_bam
]
if not os.path.exists(recalibrated_bam):
    subprocess.run(apply_bqsr_command, check=True)

#* AnalyzeCovariates
analyze_covariates_command = [
    gatk, "AnalyzeCovariates",
    "-bqsr", recal_data_table,
    "-plots", covariates_plot
]
if not os.path.exists(covariates_plot):
    subprocess.run(analyze_covariates_command, check=True)

#* Mutect2
mutect2_command = [
    gatk, "Mutect2",
    "-R", reference_genome_fasta,
    "-I", recalibrated_bam,
    "-O", mutect2_unfiltered_vcf,
    "--dont-use-soft-clipped-bases",
    "--min-base-quality-score", "10",
    "--native-pair-hmm-threads", str(threads)
]
if disable_tool_default_read_filters:
    mutect2_command += ["--disable-tool-default-read-filters"]
if tmp_dir:
    mutect2_command += ["--tmp-dir", tmp_dir]
if not os.path.exists(mutect2_unfiltered_vcf):
    subprocess.run(mutect2_command, check=True)

#* FilterMutectCalls
filter_mutect_calls_command = [
    gatk, "FilterMutectCalls",
    "-R", reference_genome_fasta,
    "-V", mutect2_unfiltered_vcf,
    "-O", mutect2_filtered_vcf
]
if tmp_dir:
    filter_mutect_calls_command += ["--tmp-dir", tmp_dir]
if not os.path.exists(mutect2_filtered_vcf):
    subprocess.run(filter_mutect_calls_command, check=True)

#* SelectVariants
select_variants_command = [
    gatk, "SelectVariants",
    "-V", mutect2_filtered_vcf,
    "--exclude-filtered", "true",
    "-O", mutect2_filtered_applied_vcf
]
if tmp_dir:
    select_variants_command += ["--tmp-dir", tmp_dir]
if not os.path.exists(mutect2_filtered_applied_vcf):
    subprocess.run(select_variants_command, check=True)

if skip_accuracy_analysis:
    print("Skipping accuracy analysis")
    sys.exit()



mutect2_vcf_file = mutect2_filtered_applied_vcf if apply_mutation_filters else mutect2_unfiltered_vcf
mutect2_vcf_file = os.path.realpath(mutect2_vcf_file)

varseek_denovo_vcf = os.path.realpath(varseek_denovo_vcf)

happy_out = os.path.join(gatk_parent, "hap_py_out", "gatk_mutect2")
happy_out = os.path.realpath(happy_out)

reference_genome_fasta = os.path.realpath(reference_genome_fasta)




def is_vcf_normalized(vcf_path):
    """
    Returns True if the VCF has a header line indicating bcftools norm was run.
    """
    # Auto-detect if compressed
    open_func = gzip.open if vcf_path.endswith(".gz") else open

    with open_func(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##bcftools_normCommand="):
                return True
            # Headers always start with ##
            if not line.startswith("##"):
                # Stop at the column header line
                break
    return False

def add_normalized_before_first_dot(path):
    dirname, basename = os.path.split(path)
    if "." in basename:
        parts = basename.split(".", 1)
        new_basename = parts[0] + "_normalized." + parts[1]
    else:
        new_basename = basename + "_normalized"
    return os.path.join(dirname, new_basename)

def make_normalized_vcf(test_vcf, reference_fasta):
    test_vcf_unnormalized = test_vcf
    test_vcf = add_normalized_before_first_dot(test_vcf_unnormalized)
    
    if os.path.isfile(test_vcf):
        print(f"Normalized file {test_vcf} already exists. Skipping normalization.")
    else:
        output_type = "-Oz" if test_vcf.endswith(".vcf.gz") else "-Ov"
        bcftools_normalization_command = ["bcftools", "norm", "-c", "w", "-f", reference_fasta, "-m", "-both", output_type, "-o", test_vcf, test_vcf_unnormalized]
        subprocess.run(bcftools_normalization_command, check=True)
        subprocess.run(["bcftools", "index", "-f", "-t", test_vcf], check=True)
    
    if test_vcf.endswith(".gz") and not os.path.isfile(f"{test_vcf}.tbi"):
        subprocess.run(["bcftools", "index", "-t", test_vcf], check=True)

    return test_vcf

def compare_two_vcfs_with_hap_py(ground_truth_vcf, test_vcf, reference_fasta, output_dir = ".", dry_run = False, use_docker = True, output_prefix = "happy", happy_env = None):
    ground_truth_vcf_dir = os.path.dirname(ground_truth_vcf)
    test_vcf_dir = os.path.dirname(test_vcf)
    reference_fasta_dir = os.path.dirname(reference_fasta)
    
    reference_fasta_index = f"{reference_fasta}.fai"
    if not os.path.isfile(reference_fasta_index):
        subprocess.run(["samtools", "faidx", reference_fasta], check=True)

    if not is_vcf_normalized(test_vcf):
        test_vcf = make_normalized_vcf(test_vcf, reference_fasta)
    
    if not is_vcf_normalized(ground_truth_vcf):
        ground_truth_vcf = make_normalized_vcf(ground_truth_vcf, reference_fasta)

    summary_csv_path = os.path.join(output_dir, f"{output_prefix}.summary.csv")
    if os.path.isfile(summary_csv_path):
        print(f"Summary file {summary_csv_path} already exists. Skipping hap.py run.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_prefix_full = os.path.join(output_dir, output_prefix)
        if use_docker:
            command = f"docker run --rm -v {ground_truth_vcf_dir}:{ground_truth_vcf_dir} -v {test_vcf_dir}:{test_vcf_dir} -v {reference_fasta_dir}:{reference_fasta_dir} -v {output_dir}:{output_dir} mgibio/hap.py:v0.3.12 /opt/hap.py/bin/hap.py -r {reference_fasta} --engine=scmp-somatic -o {output_prefix_full} {ground_truth_vcf} {test_vcf}"
        else:
            command = f"hap.py -r {reference_fasta} --engine=scmp-somatic -o {output_prefix_full} {ground_truth_vcf} {test_vcf}"
            if happy_env is not None:
                command = f"conda run -n {happy_env} " + command
        if dry_run:
            print("Dry run is true. Run the following command in the terminal, or set dry_run = False:")
            print(command)
            return
        else:
            subprocess.run(command, shell=True, check=True)
                
    # Step 1: Load happy.vcf.gz (annotated VCF)
    happy_vcf = pysam.VariantFile(os.path.join(output_dir, f"{output_prefix}.vcf.gz"))

    # Step 2: Load original VCF (with IDs)
    orig_vcf = pysam.VariantFile(ground_truth_vcf)

    # Step 3: Build a lookup dictionary from original VCF
    orig_lookup = {}
    for rec in orig_vcf.fetch():
        key = (rec.contig, rec.pos, rec.ref, tuple(rec.alts))
        orig_lookup[key] = rec.id

    # Step 4: Collect IDs based on TP and FN
    rows = []
    for rec in happy_vcf.fetch():
        for sample in rec.samples.values():
            bd = sample.get("BD")
            key = (rec.contig, rec.pos, rec.ref, tuple(rec.alts))
            match_id = orig_lookup.get(key)
            if not match_id:
                continue
            rows.append({
                "ID": match_id,
                "CHROM": rec.contig,
                "POS": rec.pos,
                "REF": rec.ref,
                "ALT": ",".join(rec.alts),
                "BD": bd,
                "DP": rec.info.get("QUERY_DP"),
                "RD": rec.info.get("QUERY_RD", None),
                "AD": rec.info.get("QUERY_AD", None),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"{output_prefix}.detailed.csv"), index=False)

compare_two_vcfs_with_hap_py(ground_truth_vcf=mutect2_vcf_file, test_vcf=varseek_denovo_vcf, reference_fasta=reference_genome_fasta, output_dir=happy_out, dry_run=False, user_docker=False, happy_env=happy_env)
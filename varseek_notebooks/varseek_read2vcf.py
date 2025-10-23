import shutil
import argparse
import subprocess
import sys
import os
import re
import logging
import shlex
import tempfile

logger = logging.getLogger(__name__)

def configure_logger(verbose_level, quiet):
    """Configure the logger based on verbosity and quiet flags."""
    if quiet:
        level = logging.CRITICAL
    elif verbose_level >= 2:
        level = logging.DEBUG
    elif verbose_level == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def run(cmd, check=True, shell=True, logger=logger):
    """Run a shell command and log it before execution."""
    logger.debug(cmd)
    subprocess.run(cmd, shell=shell, check=check)

def check_tool(tool):
    """Ensure that a required command-line tool is available."""
    if not shutil.which(tool) or not os.path.exists(tool):
        sys.exit(f"Error: required tool '{tool}' is not installed or not in PATH.")

def sanitize_ref_name(fasta_ref):
    """Strip .fa/.fasta/.fna(.gz) and replace dots with underscores."""
    ref_base = os.path.basename(fasta_ref)
    ref_base = re.sub(r"\.(fa|fasta|fna)(\.gz)?$", "", ref_base)
    return ref_base.replace(".", "_")

def read2vcf(
    inputs,
    fasta_ref,
    gtf=None,
    star_genome_index_dir="genome_index",
    bowtie2_genome_index_prefix="bowtie2_index",
    star_alignment_prefix="star_",
    bowtie2_alignment_dir="bowtie2_alignments",
    regions=None,
    output="out.vcf.gz",
    read_length=90,
    min_counts=3,
    aligner="STAR",
    bowtie2_seed_length=None,
    bowtie2_score_min=None,
    include=None,
    skip_indels=False,
    disable_baq=False,
    split_bam_by_n=False,
    disable_bcftools_norm=False,
    bcftools_call_prior=None,
    merge_bam_files=False,
    strip_version_numbers=False,
    tmp_dir=None,
    threads=1,
    overwrite=False,
    verbose=0,
    quiet=False,
):
    #* Configure logger
    configure_logger(verbose, quiet)

    #* Check tools
    for tool in ["bcftools"]:
        check_tool(tool)

    #* Validate flagged arguments
    if not output.endswith(".vcf") and not output.endswith(".vcf.gz"):
        sys.exit("Error: --output must end with .vcf or .vcf.gz")
    if fasta_ref:
        valid_fasta_extensions = [".fa", ".fasta", ".fa.gz", ".fasta.gz", ".fna", ".fna.gz"]
        if not any(fasta_ref.endswith(ext) for ext in valid_fasta_extensions):
            sys.exit(f"Error: --fasta-ref must be a FASTA file ending with {', '.join(valid_fasta_extensions)}")
        if not os.path.isfile(fasta_ref):
            fasta_dir = os.path.dirname(fasta_ref) or "."
            recommended_command = f"gget ref -r 111 -d -od {fasta_dir} -w dna human && gunzip {fasta_ref}.gz"
            sys.exit(f"Error: FASTA reference '{fasta_ref}' not found. Recommended command to download: {recommended_command}")
    if gtf:
        if not gtf.endswith(".gtf"):
            sys.exit("Error: --gtf must be a GTF file ending with .gtf")
        if not os.path.isfile(gtf):
            gtf_dir = os.path.dirname(gtf) or "."
            recommended_command = f"gget ref -r 111 -d -od {gtf_dir} -w gtf human && gunzip {gtf}.gz"
            sys.exit(f"Error: GTF file '{gtf}' not found. Recommended command to download: {recommended_command}")
    if regions:
        if not regions.endswith(".bed"):
            sys.exit("Error: --regions must be a BED file ending with .bed")
        if not os.path.isfile(regions):
            # recommended_command = f"awk '$3 == \"gene\" {{print $1, $4-1, $5, $10}}' OFS='\\t' {gtf} | sort -k1,1V -k2,2n -o {regions}"
            sys.exit(f"Error: regions BED file '{regions}' not found.")
    if min_counts < 2:
        min_counts = 0
        logger.warning("Filtering by a minimum count threshold is highly recommended. Additionally, indels observed once will not be output regardless of settings (bcftools mpileup behavior).")
    if not aligner in ["STAR", "bowtie2"]:
        sys.exit("Error: --aligner must be either 'STAR' or 'bowtie2'")
    if os.path.exists(output) and not overwrite:
        sys.exit(f"Error: output file '{output}' already exists. Use --overwrite to overwrite.")
    
    #* Validate inputs
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 2:
            sys.exit("Error: when providing multiple inputs, only two entries are allowed (R1 and R2 for paired-end reads)")
        fastq_files_1 = inputs[0].split(",")
        if len(inputs) == 2:
            fastq_files_2 = inputs[1].split(",")
            if len(fastq_files_1) != len(fastq_files_2):
                sys.exit("Error: number of R1 and R2 FASTQ files must be the same for paired-end reads")   
    else:
        sys.exit("Error: inputs must be a string or a list/tuple of strings")
    
    valid_fastq_extensions = [".fq", ".fastq", ".fq.gz", ".fastq.gz"]
    valid_bam_extensions = [".bam"]
    input_type = None
    for input_pair in inputs:
        files = input_pair.split(",")
        for file in files:
            if any(file.endswith(ext) for ext in valid_fastq_extensions):
                if input_type is None:
                    input_type = "fastq"
                elif input_type != "fastq":
                    sys.exit("Error: all inputs must be of the same type (either FASTQ or BAM)")
            elif any(file.endswith(ext) for ext in valid_bam_extensions):
                if input_type is None:
                    input_type = "bam"
                elif input_type != "bam":
                    sys.exit("Error: all inputs must be of the same type (either FASTQ or BAM)")
            else:
                sys.exit(f"Error: input file '{file}' must be a FASTQ or BAM file")
            if not os.path.isfile(file):
                sys.exit(f"Error: input file '{file}' not found")
    
    #* Define derivative variables
    output_type = "-Oz" if output.endswith(".gz") else "-Ov"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    do_filtering = (min_counts > 1) or (include is not None)
    if do_filtering:
        filter_expression = f"bcftools filter --threads {threads} {bcftools_verbosity}"
        if include:
            filter_expression += f" -i '{include}'"
        if min_counts > 1:
            filter_expression += f" -i 'INFO/AD[1] >= {min_counts}'"
    bcftools_verbosity = ""
    if not quiet:
        if verbose == 1:
            bcftools_verbosity = " -v"
        elif verbose >= 2:
            bcftools_verbosity = " -vv"

    #* Align reads if BAM doesn't exist
    bam_for_bcftools = None
    if input_type == "fastq":
        check_tool("samtools")
        if aligner == "STAR":
            check_tool("STAR")
            read_length_minus_one = read_length - 1
            bam_for_bcftools = f"{star_alignment_prefix}Aligned.sortedByCoord.out.bam"
            
            if not os.path.exists(bam_for_bcftools):
                #* Build STAR genome if needed
                if not os.path.isdir(star_genome_index_dir) or not os.listdir(star_genome_index_dir):
                    logger.info(f"Building STAR genome index at {star_genome_index_dir}...")
                    star_build_command = f"STAR --runThreadN {threads} --runMode genomeGenerate --genomeDir {star_genome_index_dir} --genomeFastaFiles {fasta_ref} --sjdbGTFfile {gtf} --sjdbOverhang {read_length_minus_one} --limitSjdbInsertNsj 1000000 --limitBAMsortRAM 0"
                    run(star_build_command)
                
                logger.info("Running STAR alignment...")
                inputs_star = " ".join(inputs)
                cmd = f"""
                STAR --runThreadN {threads} \
                    --genomeDir {star_genome_index_dir} \
                    --readFilesIn {inputs_star} \
                    --sjdbOverhang {read_length_minus_one} \
                    --outFileNamePrefix {star_alignment_prefix} \
                    --outSAMtype BAM SortedByCoordinate \
                    --outSAMmapqUnique 60 \
                    --twopassMode Basic \
                    --limitSjdbInsertNsj 1000000 \
                    --limitBAMsortRAM 0
                """
                if inputs[0].endswith(".gz"):
                    cmd += " --readFilesCommand zcat"
                run(cmd)

                #* Split spliced reads if requested
                if split_bam_by_n:
                    check_tool("gatk")
                    split_bam = f"{star_alignment_prefix}split_exons.sorted.bam"
                    if not os.path.exists(split_bam):  # TODO: rewrite without GATK dependency
                        split_cmd = f"gatk SplitNCigarReads -R {fasta_ref} -I {bam_for_bcftools} -O {split_bam} --create-output-bam-index"
                        if tmp_dir:
                            split_cmd += f" --tmp-dir {tmp_dir}"
                        run(split_cmd)
                    bam_for_bcftools = split_bam

        elif aligner == "bowtie2":
            check_tool("bowtie2")
            
            bowtie2_options = ""
            if bowtie2_seed_length is not None:
                bowtie2_options += f" -L {bowtie2_seed_length}"
            if bowtie2_score_min is not None:
                bowtie2_options += f" --score-min {bowtie2_score_min}"

            if merge_bam_files:
                bam_for_bcftools = os.path.join(bowtie2_alignment_dir, "aligned.sorted.bam")
                if not os.path.exists(bam_for_bcftools):
                    bowtie2_genome_index_file = f"{bowtie2_genome_index_prefix}.1.bt2"
                    if not os.path.exists(bowtie2_genome_index_file):
                        bowtie_build_command = f"bowtie2-build {fasta_ref} {bowtie2_genome_index_prefix}"
                        run(bowtie_build_command)
                    os.makedirs(bowtie2_alignment_dir, exist_ok=True)
                    if len(inputs) == 2:  # paired-end
                        bowtie2_align_command = f"bowtie2 --xeq --very-sensitive {bowtie2_options} --threads {threads} -x {bowtie2_genome_index_prefix} -1 {inputs[0]} -2 {inputs[1]} | samtools view -bS - | samtools sort -o {bam_for_bcftools}"
                    elif len(inputs) == 1:  # single-end
                        bowtie2_align_command = f"bowtie2 --xeq --very-sensitive {bowtie2_options} --threads {threads} -x {bowtie2_genome_index_prefix} -U {inputs[0]} | samtools view -bS - | samtools sort -o {bam_for_bcftools}"
                    logger.info("Running Bowtie2 alignment...")
                    run(bowtie2_align_command)
            else:
                bam_for_bcftools = []
                first_fastq = fastq_files_1[0]
                if not out_bam_dir:
                    out_bam_dir = os.path.dirname(first_fastq)
                fasta_ref_base = os.path.basename(fasta_ref)
                fasta_ref_base = re.sub(r"\.(fa|fasta|fna)(\.gz)?$", "", fasta_ref_base)
                fasta_ref_base = fasta_ref_base.replace(".", "_")
                if len(inputs) == 2:  # paired-end
                    for fastq1, fastq2 in zip(fastq_files_1, fastq_files_2):
                        fq_base = re.sub(r"\..*", "", os.path.basename(fastq1))
                        bam_out = os.path.join(out_bam_dir, f"{fq_base}_aligned_to_{fasta_ref_base}.bam")
                        bowtie2_align_command = f"bowtie2 --xeq --very-sensitive {bowtie2_options} --threads {threads} -x {bowtie2_genome_index_prefix} -1 {fastq1} -2 {fastq2} | samtools view -bS - | samtools sort -o {bam_out}"
                        run(bowtie2_align_command)
                        if os.path.exists(bam_out):
                            bam_for_bcftools.append(bam_out)
                        else:
                            logger.warning(f"Bowtie2 alignment for {fastq1} and {fastq2} did not produce expected BAM output '{bam_out}'")
                elif len(inputs) == 1:  # single-end
                    for fastq in fastq_files_1:
                        fq_base = re.sub(r"\..*", "", os.path.basename(fastq))
                        bam_out = os.path.join(out_bam_dir, f"{fq_base}_aligned_to_{fasta_ref_base}.bam")
                        bowtie2_align_command = f"bowtie2 --xeq --very-sensitive {bowtie2_options} --threads {threads} -x {bowtie2_genome_index_prefix} -U {fastq} | samtools view -bS - | samtools sort -o {bam_out}"
                        run(bowtie2_align_command)
                        if os.path.exists(bam_out):
                            bam_for_bcftools.append(bam_out)
                        else:
                            logger.warning(f"Bowtie2 alignment for {fastq} did not produce expected BAM output '{bam_out}'")
                else:
                    sys.exit("Error: invalid number of inputs for bowtie2 alignment")
                bam_for_bcftools = " ".join(bam_for_bcftools)
        else:
            sys.exit(f"Error: aligner '{aligner}' not supported")

    #* Index BAM
    assert isinstance(bam_for_bcftools, str)
    bam_files = shlex.split(bam_for_bcftools)
    for bam in bam_files:
        bai = bam + ".bai"
        if not os.path.exists(bai):
            run(f"samtools index -@ {threads} {bam}")
    
    #* bcftools mpileup
    if not os.path.exists(bam_for_bcftools):
        sys.exit("Error: BAM file for bcftools not found or generated")

    bcftools_cmd = f"bcftools mpileup --threads {threads} -A -f {fasta_ref} -a INFO/AD -Q 0 -d 10000 -Ou {bcftools_verbosity}"
    if regions:
        bcftools_cmd += f" -R {regions}"
    if disable_baq:
        bcftools_cmd += " -B"
    if skip_indels:
        bcftools_cmd += " -I"
    bcftools_cmd += f" {bam_for_bcftools}"

    #* bcftools filter
    if do_filtering:
        bcftools_cmd += f" | {filter_expression} -Ou"
    
    #* bcftools call
    bcftools_cmd += f" | bcftools call -m -A -v --threads {threads} {bcftools_verbosity}"
    if bcftools_call_prior:
        bcftools_cmd += f" --prior {bcftools_call_prior}"

    #* optional: bcftools norm and additional filter (must repeat after normalization)
    if not disable_bcftools_norm:
        bcftools_cmd += f" -Ou | bcftools norm -f {fasta_ref} -c s -d all -m -any --threads {threads} {bcftools_verbosity}"
        if do_filtering:
            bcftools_cmd += f" -Ou | {filter_expression}"

    bcftools_cmd += f" {output_type} -o {output}"

    run(bcftools_cmd)

    #* optional: strip version numbers
    if strip_version_numbers:
        tmp_fh = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf.gz" if output_type == "-Oz" else ".vcf")
        tmp_file = tmp_fh.name
        tmp_fh.close()
        if output_type == "-Oz":
            # compressed .vcf.gz
            logger.info(f"Stripping version numbers and recompressing {output}...")
            cmd = (
                f"zcat {output} "
                r"| awk 'BEGIN{{OFS=\"\t\"}} {{sub(/\.[0-9]+$/, \"\", $1); print}}' "
                f"| bgzip > {tmp_file}"
            )
            subprocess.run(cmd, shell=True, check=True)
        else:
            # uncompressed .vcf
            logger.info(f"Stripping version numbers in {output}...")
            cmd = (
                r"awk 'BEGIN{OFS=\"\t\"} {sub(/\.[0-9]+$/, \"\", $1); print}' "
                f"{output} > {tmp_file}"
            )
            subprocess.run(cmd, shell=True, check=True)

        shutil.move(tmp_file, output)

    #* index VCF
    run(f"bcftools index -f --threads {threads} {output}")

    logger.info(f"Program complete. VCF written to {output}")

def main():
    parser = argparse.ArgumentParser(description="STAR alignment + bcftools variant calling pipeline")
    parser.add_argument("inputs", nargs="+", help="Input FASTQs. If multiple fastqs, separate with commas. If paired fastqs, first pass in R1 files separated by commas, then space, then R2 files separated by commas.")
    parser.add_argument("-f", "--fasta-ref", required=True, help="Reference FASTA file")
    parser.add_argument("--gtf", default="", help="genome annotation GTF file")
    parser.add_argument("-x", "--star-genome-index-dir", default="genome_index", help="STAR or Bowtie2 genome index directory")
    parser.add_argument("--bowtie2-genome-index-prefix", default="bowtie2_index", help="prefix for Bowtie2 genome index files")
    parser.add_argument("--star-alignment-prefix", default="star_", help="prefix for STAR output BAM")
    parser.add_argument("--bowtie2-alignment-dir", default="bowtie2_alignments", help="directory for Bowtie2 output BAMs")
    parser.add_argument("--regions", default="", help="BED file of regions to restrict variant calling to")
    parser.add_argument("--out-bam-dir", default="", help="Output directory for BAM files (for Bowtie2 aligner only when not merging BAMs)")
    parser.add_argument("-o", "--output", default="out.vcf.gz", help="Output VCF file")
    parser.add_argument("--read-length", type=int, default=90, help="Read length")
    parser.add_argument("--min-counts", type=int, default=3, help="Minimum count threshold for filtering")
    parser.add_argument("--aligner", default="STAR", choices=["STAR", "bowtie2"], help="Aligner to use: STAR or bowtie2")
    parser.add_argument("--bowtie2-seed-length", type=int, default=None, help="Seed length for Bowtie2 aligner")
    parser.add_argument("--bowtie2-score-min", default=None, help="Score minimum for Bowtie2 aligner")
    parser.add_argument("-i", "--include", default="", help="bcftools filter expression")
    parser.add_argument("-I", "--skip-indels", action="store_true", help="Skip indels")
    parser.add_argument("--disable-baq", action="store_true", help="Disable BAQ computation in mpileup")
    parser.add_argument("--split-bam-by-n", action="store_true", help="Split BAM by N in CIGAR (spliced reads)")
    parser.add_argument("--merge-bam-files", action="store_true", help="Merge multiple BAM files into one for variant calling (Bowtie2 only)")
    parser.add_argument("--strip-version-numbers", action="store_true", help="Strip version numbers from chromosome names in output VCF")
    parser.add_argument("--disable-bcftools-norm", action="store_true", help="Disable running bcftools norm")
    parser.add_argument("--bcftools-call-prior", default="", help="Prior for bcftools call")
    parser.add_argument("--tmp-dir", default="/tmp", help="Temporary directory for intermediate files") 
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity (default logging.WARNING, -v logging.INFO, -vv for logging.DEBUG)") 
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output (overrides any verbose flag)") 
    args = parser.parse_args()

    read2vcf(
        inputs=args.inputs,
        fasta_ref=args.fasta_ref,
        gtf=args.gtf,
        star_genome_index_dir=args.star_genome_index_dir,
        bowtie2_genome_index_prefix=args.bowtie2_genome_index_prefix,
        star_alignment_prefix=args.star_alignment_prefix,
        bowtie2_alignment_dir=args.bowtie2_alignment_dir,
        regions=args.regions,
        out_bam_dir=args.out_bam_dir,
        output=args.output,
        read_length=args.read_length,
        min_counts=args.min_counts,
        aligner=args.aligner,
        bowtie2_seed_length=args.bowtie2_seed_length,
        bowtie2_score_min=args.bowtie2_score_min,
        include=args.include,
        skip_indels=args.skip_indels,
        disable_baq=args.disable_baq,
        split_bam_by_n=args.split_bam_by_n,
        merge_bam_files=args.merge_bam_files,
        strip_version_numbers=args.strip_version_numbers,
        disable_bcftools_norm=args.disable_bcftools_norm,
        bcftools_call_prior=args.bcftools_call_prior,
        tmp_dir=args.tmp_dir,
        threads=args.threads,
        overwrite=args.overwrite,
        verbose=args.verbose,
        quiet=args.quiet,
    )

if __name__ == "__main__":
    main()

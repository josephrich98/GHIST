import shutil
import argparse
import subprocess
import sys
import os
import logging

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

def read2vcf(
    inputs,
    fasta_ref,
    gtf=None,
    star_genome_dir="star_genome_index",
    star_alignment_prefix="star_",
    regions=None,
    output="out.vcf.gz",
    read_length=90,
    min_counts=3,
    include=None,
    skip_indels=False,
    disable_baq=False,
    split_bam_by_n=False,
    disable_bcftools_norm=False,
    bcftools_call_prior=None,
    tmp_dir=None,
    threads=1,
    verbose=1,
    quiet=False,
):
    #* Configure logger
    configure_logger(verbose, quiet)

    #* Check tools
    for tool in ["STAR", "samtools", "bcftools"]:
        check_tool(tool)

    #* Validate arguments
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
            recommended_command = f"awk '$3 == \"gene\" {{print $1, $4-1, $5, $10}}' OFS='\\t' {gtf} | sort -k1,1V -k2,2n -o {regions}"
            sys.exit(f"Error: regions BED file '{regions}' not found. Recommended command to generate: {recommended_command}")
    if min_counts < 2:
        min_counts = 0
        logger.warning("Filtering by a minimum count threshold is highly recommended. Additionally, indels observed once will not be output regardless of settings (bcftools mpileup behavior).")
    
    #* Define derivative variables
    read_length_minus_one = read_length - 1
    out_bam = f"{star_alignment_prefix}Aligned.sortedByCoord.out.bam"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    output_type = "-Oz" if output.endswith(".gz") else "-Ov"
    do_filtering = (min_counts > 1) or (include is not None)
    if do_filtering:
        filter_expression = f"bcftools filter --threads {threads}"
        if include:
            filter_expression += f" -i '{include}'"
        if min_counts > 1:
            filter_expression += f" -i 'INFO/AD[1] >= {min_counts}'"

    #* Align reads if BAM doesn't exist
    if not os.path.exists(out_bam):
        #* Build STAR genome if needed
        if not os.path.isdir(star_genome_dir) or not os.listdir(star_genome_dir):
            logger.info(f"Building STAR genome index at {star_genome_dir}...")
            star_build_command = f"STAR --runThreadN {threads} --runMode genomeGenerate --genomeDir {star_genome_dir} --genomeFastaFiles {fasta_ref} --sjdbGTFfile {gtf} --sjdbOverhang {read_length_minus_one} --limitSjdbInsertNsj 1000000 --limitBAMsortRAM 0"
            run(star_build_command)
        
        logger.info("Running STAR alignment...")
        inputs_star = " ".join(inputs)
        cmd = f"""
        STAR --runThreadN {threads} \
             --genomeDir {star_genome_dir} \
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

    #* Index BAM
    bai = out_bam + ".bai"
    if not os.path.exists(bai):
        run(f"samtools index -@ {threads} {out_bam}")

    #* Split spliced reads if requested
    if split_bam_by_n:
        check_tool("gatk")
        split_bam = f"{star_alignment_prefix}split_exons.sorted.bam"
        if not os.path.exists(split_bam):  # TODO: rewrite without GATK dependency
            split_cmd = f"gatk SplitNCigarReads -R {fasta_ref} -I {out_bam} -O {split_bam} --create-output-bam-index"
            if tmp_dir:
                split_cmd += f" --tmp-dir {tmp_dir}"
            run(split_cmd)
        out_bam = split_bam

    #* bcftools mpileup
    bcftools_cmd = f"bcftools mpileup --threads {threads} -A -f {fasta_ref} -a INFO/AD -Q 0 -d 10000 -Ou"
    if regions:
        bcftools_cmd += f" -R {regions}"
    if disable_baq:
        bcftools_cmd += " -B"
    if skip_indels:
        bcftools_cmd += " -I"
    bcftools_cmd += f" {out_bam}"

    #* bcftools filter
    if do_filtering:
        bcftools_cmd += f" | {filter_expression} -Ou"
    
    #* bcftools call
    bcftools_cmd += f" | bcftools call -m -A -v --threads {threads}"
    if bcftools_call_prior:
        bcftools_cmd += f" --prior {bcftools_call_prior}"

    #* optional: bcftools norm and additional filter (must repeat after normalization)
    if not disable_bcftools_norm:
        bcftools_cmd += f" -Ou | bcftools norm -f {fasta_ref} -c s -d all -m -any --threads {threads}"
        if do_filtering:
            bcftools_cmd += f" -Ou | {filter_expression}"

    bcftools_cmd += f" {output_type} -o {output}"

    run(bcftools_cmd)
    run(f"bcftools index -f --threads {threads} {output}")    

    logger.info(f"Program complete. VCF written to {output}")

def main():
    parser = argparse.ArgumentParser(description="STAR alignment + bcftools variant calling pipeline")
    parser.add_argument("inputs", nargs="+", help="Input FASTQs. If multiple fastqs, separate with commas. If paired fastqs, first pass in R1 files separated by commas, then space, then R2 files separated by commas.")
    parser.add_argument("-f", "--fasta-ref", required=True, help="Reference FASTA file")
    parser.add_argument("--gtf", default="", help="genome annotation GTF file")
    parser.add_argument("-x", "--star-genome-dir", default="star_genome_index", help="STAR genome index directory")
    parser.add_argument("--star-alignment-prefix", default="star_", help="prefix for STAR output BAM")
    parser.add_argument("--regions", default="", help="BED file of regions to restrict variant calling to")
    parser.add_argument("-o", "--output", default="out.vcf.gz", help="Output VCF file")
    parser.add_argument("--read-length", type=int, default=90, help="Read length")
    parser.add_argument("--min-counts", type=int, default=3, help="Minimum count threshold for filtering")
    parser.add_argument("-i", "--include", default="", help="bcftools filter expression")
    parser.add_argument("-I", "--skip-indels", action="store_true", help="Skip indels")
    parser.add_argument("--disable-baq", action="store_true", help="Disable BAQ computation in mpileup")
    parser.add_argument("--split-bam-by-n", action="store_true", help="Split BAM by N in CIGAR (spliced reads)")
    parser.add_argument("--disable-bcftools-norm", action="store_true", help="Disable running bcftools norm")
    parser.add_argument("--bcftools-call-prior", default="", help="Prior for bcftools call")
    parser.add_argument("--tmp-dir", default="/tmp", help="Temporary directory for intermediate files") 
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity (default logging.WARNING, -v logging.INFO, -vv for logging.DEBUG)") 
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output (overrides any verbose flag)") 
    args = parser.parse_args()

    read2vcf(
        inputs=args.inputs,
        fasta_ref=args.fasta_ref,
        threads=args.threads,
        star_genome_dir=args.star_genome_dir,
        output=args.output,
        min_counts=args.min_counts,
        include=args.include,
        skip_indels=args.skip_indels,
        disable_baq=args.disable_baq,
        split_bam_by_n=args.split_bam_by_n,
        regions=args.regions,
        gtf=args.gtf,
        read_length=args.read_length,
        star_alignment_prefix=args.star_alignment_prefix,
        disable_bcftools_norm=args.disable_bcftools_norm,
        bcftools_call_prior=args.bcftools_call_prior,
        tmp_dir=args.tmp_dir,
        verbose=args.verbose,
        quiet=args.quiet,
    )

if __name__ == "__main__":
    main()

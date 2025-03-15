import os
import argparse
import subprocess


def main(config):

    code_dir = config.code_dir
    os.chdir(code_dir)

    dir_output = config.dir_output
    gpu_id = config.gpu_id
    fp_he_img = config.fp_he_img
    dir_hovernet = config.dir_hovernet
    dir_xenium_outs = config.dir_xenium_outs
    n_processes = config.n_processes
    shell = config.shell

    # STEP 1
    command = f"python 1_get_gene_panel.py --fp_transcripts {dir_xenium_outs}/transcripts.csv.gz --dir_output {dir_output}"
    os.system(command)

    # # STEP 2 - OPTIONAL - for visualisation only
    # command = f"python 2_optional_get_summed_transcripts_image.py --fp_transcripts {dir_xenium_outs}/transcripts.csv.gz --dir_output {dir_output} --n_processes {n_processes}"
    # os.system(command)

    # STEP 3
    command = f"python 3_get_xenium_nuclei_seg_image.py --fp_boundaries {dir_xenium_outs}/nucleus_boundaries.csv.gz --dir_output {dir_output} --fp_he_img {fp_he_img} --crop_fraction 0.1  --n_processes {n_processes}"
    os.system(command)

    # STEP 4
    command = f"python 4_get_xenium_cell_gene_matrix.py --dir_feature_matrix {dir_xenium_outs}/cell_feature_matrix --dir_output {dir_output}"
    os.system(command)

    # STEP 5.1
    command = f"python 5_segment_nuclei_he_image.py --dir_output {dir_output} --fp_he_img {fp_he_img} --dir_hovernet {dir_hovernet} --gpu_id {gpu_id} --step 1"
    os.system(command)

    # STEP 5.2
    env_name = "hovernet"
    subprocess.run(f"conda activate {env_name} && python 5_segment_nuclei_he_image.py --dir_output {dir_output} --fp_he_img {fp_he_img} --dir_hovernet {dir_hovernet} --gpu_id {gpu_id} --step 2", shell=True, executable=f"/bin/{shell}")

    # STEP 5.3
    subprocess.run(f"conda activate ghist && python 5_segment_nuclei_he_image.py --dir_output {dir_output} --fp_he_img {fp_he_img} --dir_hovernet {dir_hovernet} --gpu_id {gpu_id} --step 3", shell=True, executable=f"/bin/{shell}")

    # STEP 6
    command = f"python 6_get_overlapping_nuclei.py --dir_output {dir_output} --n_processes {n_processes}"
    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--code_dir",
        default="data_processing",
        type=str,
        help="dir of data processing code",
    )
    parser.add_argument(
        "--dir_output",
        default="data_processing_breast_ILC_test",
        type=str,
        help="dir of processed data",
    )
    parser.add_argument(
        "--dir_xenium_outs",
        default="/dskh/nobackup/helenf/Xenium_breast_datasets/Xenium_V1_FFPE_Human_Breast_ILC_outs",
        type=str,
        help="dir of Xenium outputs",
    )
    parser.add_argument(
        "--fp_he_img",
        default="/dskh/nobackup/helenf/Xenium_breast_datasets/Xenium_V1_FFPE_Human_Breast_ILC_he_image.ome.tif",
        type=str,
        help="file path of H&E image",
    )
    parser.add_argument(
        "--dir_hovernet",
        default="/dskh/nobackup/helenf/hover_net",
        type=str,
        help="dir of hovernet code",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="which GPU to use",
    )
    parser.add_argument(
        "--n_processes",
        default=24,
        type=int,
        help="num cpus to use, or None for all cpus - 1",
    )
    parser.add_argument(
        "--shell",
        default="tcsh",
        type=str,
        help="the type of shell of your system",
    )

    config = parser.parse_args()
    main(config)

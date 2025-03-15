# Spatial gene expression at single-cell resolution from histology using deep learning with GHIST

For more details, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2024.07.02.601790v1).

The increased use of spatially resolved transcriptomics provides new biological insights into disease mechanisms. However, the high cost and complexity of these methods are barriers to broad clinical adoption. Consequently, methods have been created to predict spot-based gene expression from routinely-collected histology images. Recent benchmarking showed that current methodologies have limited accuracy and spatial resolution, constraining translational capacity. Here, we introduce GHIST, a deep learning-based framework that predicts spatial gene expression at single-cell resolution by leveraging subcellular spatial transcriptomics and synergistic relationships between multiple layers of biological information. We validated GHIST using public datasets and The Cancer Genome Atlas data, demonstrating its flexibility across different spatial resolutions and superior performance. Our results underscore the utility of in silico generation of single-cell spatial gene expression measurements and the capacity to enrich existing datasets with a spatially resolved omics modality, paving the way for scalable multi-omics analysis and new biomarker discoveries.  

![alt text](Figure1.png)

pip 
## Installation

> **Note**: A GPU with 24GB VRAM is strongly recommended for the deep learning component, and 32GB RAM for data processing.
We ran GHIST on a Linux system with a 24GB NVIDIA GeForce RTX 4090 GPU, Intel(R) Core(TM) i9-13900F CPU @ 5.60GHz with 32 threads, and 32GB RAM.

1. Clone repository:
```sh
git clone https://github.com/SydneyBioX/GHIST.git
```

2. Create virtual environment:
```sh
conda create --name ghist python=3.10
```

3. Activate virtual environment:
```sh
conda activate ghist
```

4. Install dependencies:
```sh
cd GHIST
pip install -r requirements.txt
```
Please install the ``stainlib`` package from https://github.com/sebastianffx/stainlib

Typically installation is expected to be completed within a few minutes.


## Demo

This small demo dataset is based on publicly available data provided by 10x Genomics (In Situ Sample 2): https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast. We will use a subset of the data and a previously saved checkpoint as a short demo.


### Running the demo

1. Unzip the file `data_demo/images.zip` and place the 2 `.tif` files under `data_demo`:
```sh
unzip data_demo/images.zip -d data_demo/
```

2. Download the saved model for this demo:
```sh
gdown --folder https://drive.google.com/drive/folders/1FJrJZ0f1y5MQ19aOoH-9F6iXOrvIQWha?usp=drive_link
```

3. Train:
```sh
python train.py --config_file configs/config_demo.json
```

4. Validation:
```sh
python inference.py --config_file configs/config_demo.json --mode val
```

5. Testing:
```sh
python inference.py --config_file configs/config_demo.json --mode test
```

The predictions are stored in ``experiments/{timestamp}/{mode}_output/``, and the csv files contains the predicted gene expressions for each cell, where the index is the cell ID that corresponds to the IDs from the nuclei segmentation image, and the columns are the genes. Demo is expected to be completed within a few minutes.


## Running GHIST:

Create a config file under ``./configs``, please see the demo config file as an example. Important parameters to update include:
- ``comps``: if ``avgexp``, ``celltype``, or ``neighb`` information is not available, set any of these missing parameters to ``false``. If ``celltype`` is ``false``, ``neighb`` will also be ``false`` as it depends on ``celltype``.
- ``cell_types``: list of cell types in your data. Ignored if ``celltype`` is ``false``.
- ``data_sources_train_val``: locations of data for training and validation.
- ``data_sources_test``: locations of data for testing. Set ``data_sources_test.fp_nuc_sizes`` to ``false`` if there's no file for nuclei sizes for the test data.
- ``regions_val.divisions``: which portion of the image is used for validation for each fold. The image is divided along the y axis at the specified intervals. The remainder is used for training. k lists of the start and end of partition, where k is the number of cross-validation folds. Set to ``[0.0, 0.0]`` as the list of a fold to use the whole image for training.
- ``regions_test.divisions``: which portion of the image is used for testing. Typically ``[0.0, 1.0]`` for all folds.


### Training the model
```sh
python train.py --config_file configs/FILENAME.json --resume_epoch EPOCH --fold_id FOLD --gpu_id GPU_NUM
```
- ``--resume_epoch`` specifies whether to train from scratch or resume from a checkpoint, e.g., ``--resume_epoch 10`` to resume from the saved checkpoint from epoch 10. Set to 0 for training from scratch.
- ``--fold_id`` specifies the cross-validation fold (1, 2, 3...)
- ``--gpu_id`` which GPU to use (0, 1, 2...)


### Predicting from the trained model

```sh
python inference.py --config_file configs/FILENAME.json --epoch EPOCH --mode MODE --fold_id FOLD --gpu_id GPU_NUM
```
- ``--epoch`` specifies which epoch to test, e.g., ``--epoch 10`` to use the model from epoch 10, ``--epoch -1`` to use the model from the last saved epoch
- ``--mode`` can be ``val`` or ``test``
- ``--fold_id`` specifies the cross-validation fold (1, 2, 3...)
- ``--gpu_id`` which GPU to use (0, 1, 2...)


### Output cell expressions

Predicted gene expressions individual cells may be found in the experiment directory, e.g.: ``experiments/fold1_2025_02_22_10_04_14/test_output/epoch_10_expr.csv``. An example is provided as ``example_output.csv`` to show the format.  


## Data processing

A demo script (``demo_data_processing.py``) is provided for steps to process Xenium and corresponding H&E image data for GHIST. Update the following arguments:
- ``--dir_output`` directory to store the processed data
- ``--dir_xenium_outs`` directory of Xenium outs (e.g., Xenium_V1_FFPE_Human_Breast_ILC_outs)
- ``--fp_he_img`` path to H&E image that is aligned to the Xenium data (e.g., Xenium_V1_FFPE_Human_Breast_ILC_he_image.ome.tif)
- ``--dir_hovernet`` directory where the Hover-Net code is on your server (please download from their repository)
- ``--shell`` the type of shell of your system (tcsh, bash, etc)

If you'd like to use cell type information, run your preferred cell annotation method on ``cell_gene_matrix_filtered.csv`` to get cell type labels. Then create a csv file with columns like the following (see ``data_demo/celltype_filtered.csv``):

```
c_id,ct
492184,T
519402,Malignant
491961,Macrophage
...etc
```

Example data may be downloaded from here (Tissue sample 2 for ILC): https://www.10xgenomics.com/datasets/ffpe-human-breast-with-pre-designed-panel-1-standard

Run ``conda init`` for your specific shell, e.g. for ``tcsh``:
```sh
conda init tcsh
```

During the H&E nuclei segmentation step using Hover-Net, we have encountered the error below, though the segmentation still ran OK:
``Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library. Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.``


## Citation

If GHIST has assisted you with your work, please kindly cite our paper:

- https://www.biorxiv.org/content/10.1101/2024.07.02.601790v1
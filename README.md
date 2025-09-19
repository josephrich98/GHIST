# Spatial gene expression at single-cell resolution from histology using deep learning with GHIST

For more details, please refer to our [paper](https://www.nature.com/articles/s41592-025-02795-z).

GHIST is a deep learning-based framework that predicts spatial gene expression at single-cell resolution from histology (H&E-stained) images by leveraging subcellular spatial transcriptomics and synergistic relationships between multiple layers of biological information.

![alt text](Figure1.png)


## Installation

> **Note**: A GPU with 24GB VRAM is strongly recommended for the deep learning component.
We ran GHIST on a Linux system with a 24GB NVIDIA GeForce RTX 4090 GPU, Intel(R) Core(TM) i9-13900F CPU @ 5.60GHz with 32 threads, and 32GB RAM.

1. Install [hovernet](https://github.com/vqdang/hover_net) in a separate environment  
git clone git@github.com:vqdang/hover_net.git
conda env create -f hover_net/environment.yml
conda activate hovernet
pip install torch==1.6.0 torchvision==0.7.0
conda deactivate

2. Install [GHIST](https://github.com/SydneyBioX/GHIST) in a separate environment
git clone git@github.com:josephrich98/GHIST.git
conda create --name ghist python=3.10
conda activate ghist
pip install -r GHIST/requirements.txt

3. Install [stainlib](https://github.com/sebastianffx/stainlib) in ghist environment
git clone git@github.com:sebastianffx/stainlib.git
pip install -e stainlib/

## Varseek notebooks

Please check out the notebooks in [varseek_notebooks](./varseek_notebooks/) for data pre-processing, training, validation, and prediction using GHIST.

## Tutorials

Please check out the examples in [tutorials](./tutorials/) for key use cases of GHIST, including:

- [Data pre-processing](./tutorials/1_data_preprocessing.ipynb)
- [Training and validation](./tutorials/2_training_and_validation.ipynb)
- [Prediction on data without ground truth](./tutorials/3_prediction.ipynb)


## Figures

Code for creating the output figures may be found in https://github.com/SydneyBioX/GHIST_figure


## Citation

If GHIST has assisted you with your work, please kindly cite our paper: https://www.nature.com/articles/s41592-025-02795-z
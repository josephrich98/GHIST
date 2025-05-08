# Spatial gene expression at single-cell resolution from histology using deep learning with GHIST

For more details, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2024.07.02.601790v1).

GHIST is a deep learning-based framework that predicts spatial gene expression at single-cell resolution from histology (H&E-stained) images by leveraging subcellular spatial transcriptomics and synergistic relationships between multiple layers of biological information.

![alt text](Figure1.png)


## Installation

> **Note**: A GPU with 24GB VRAM is strongly recommended for the deep learning component.
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


## Tutorials

Please check out the examples in [tutorials](./tutorials/) for key use cases of GHIST, including:

- [Data pre-processing](./tutorials/1_data_preprocessing.ipynb)
- [Training and validation](./tutorials/2_training_and_validation.ipynb)
- [Prediction on data without ground truth](./tutorials/3_prediction.ipynb)


## Figures

Code for creating the output figures may be found in https://github.com/SydneyBioX/GHIST_figure


## Citation

If GHIST has assisted you with your work, please kindly cite our paper:

- https://www.biorxiv.org/content/10.1101/2024.07.02.601790v1
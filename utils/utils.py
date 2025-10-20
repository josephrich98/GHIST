import os
import datetime as dt
import json
import collections
import re
import torch
from scipy.special import softmax
import numpy as np
import random
import matplotlib.pyplot as plt
import natsort
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter
import tifffile
import scipy
import pandas as pd
import glob
import h5py
import math
import zarr
from sklearn.manifold import TSNE


def get_device(gpu_id):
    gpu_str = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print("Using GPUs: {}".format(gpu_str))
    device = torch.device("cuda")

    return device


def sorted_alphanumeric(data):
    """
    Alphanumerically sort a list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def read_txt(fp):
    with open(fp) as file:
        lines = [line.rstrip() for line in file]
    return lines


def delete_file(path):
    """
    Delete file if exists
    """
    if os.path.exists(path):
        os.remove(path)


def get_files_list(path, ext_array=[".tif"]):
    """
    Get all files in a directory with a specific extension
    """
    files_list = list()
    dirs_list = list()

    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if any(x in file for x in ext_array):
                files_list.append(os.path.join(root, file))
                folder = os.path.dirname(os.path.join(root, file))
                if folder not in dirs_list:
                    dirs_list.append(folder)

    return files_list, dirs_list


def json_file_to_pyobj(filename):
    """
    Read json config file
    """

    def _json_object_hook(d):
        return collections.namedtuple("X", d.keys())(*d.values())

    def json2obj(data):
        return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def get_newest_id(exp_dir="experiments", fold_id=1):
    """Get the latest experiment ID based on its timestamp

    Parameters
    ----------
    exp_dir : str, optional
        Name of the directory that contains all the experiment directories, by default 'experiments'

    Returns
    -------
    exp_id : str
        Name of the latest experiment directory
    """
    folders = next(os.walk(exp_dir))[1]
    folders = natsort.natsorted(folders)
    # folders = [x for x in folders if mode in x]
    folders = [x for x in folders if ("fold" + str(fold_id) + "_") in x]
    folder_last = folders[-1]
    exp_id = folder_last.replace("\\", "/")
    return exp_id


def get_experiment_id(make_new, load_dir, fold_id, run_name):
    """
    Get timestamp ID of current experiment
    """
    run_name_dir_full = f"experiments/{run_name}"
    if make_new is False:
        if load_dir != "latest" and not os.path.exists(os.path.join(run_name_dir_full, load_dir)):
            raise ValueError(f"Experiment directory {os.path.join(run_name_dir_full, load_dir)} does not exist. Please set load_dir to 'latest' or an existing directory in {run_name_dir_full} (or check 'run_name'={run_name}).")
        if load_dir == "latest":
            timestamp = get_newest_id(run_name_dir_full, fold_id)
        else:
            timestamp = load_dir
    else:
        timestamp = (
            + "fold"
            + str(fold_id)
            + "_"
            + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )

    return timestamp


def is_valid_positive_int(s):
    try:
        return int(s) > 0
    except ValueError:
        return False

def set_seed(seed: int = 42):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_run_dir_full(config_file_name, fold_id):
    with open(f"configs/{config_file_name}", "r") as f:
        config_data = json.load(f)

    run_name = config_data["experiment_dirs"]["run_name"]
    load_dir = config_data["experiment_dirs"]["load_dir"]
    timestamp = get_experiment_id(make_new=False, load_dir=load_dir, fold_id=fold_id, run_name=run_name)

    run_dir_full = f"experiments/{run_name}/{timestamp}"
    return run_dir_full


def get_tif_dimensions(image: str) -> dict[str: int]:
    coordinate_to_dimension_dict = {}
    
    with tifffile.TiffFile(image) as tif:
        image = tif.series[0]
        print(f"Image axes: {image.axes}, shape: {image.shape}")
    
    for coordinate in image.axes:
        index = image.axes.index(coordinate)
        coordinate_to_dimension_dict[coordinate.lower()] = image.shape[index]
    
    return coordinate_to_dimension_dict

def determine_image_scale_factor(image_path: str, scale: int) -> tuple[dict[str: float], dict[str: float]]:
    """
    image_path: Path to the tif image file.
    scale: The scale factor to apply (e.g., 2 double size, 0.5 for half size).
    """
    original_dimension_dict = get_tif_dimensions(image_path)
    final_dimension_dict = {k: math.ceil(v * scale) for k, v in original_dimension_dict.items()}
    return original_dimension_dict, final_dimension_dict

def plot_downsampled(path, downsample_factor=8, title=None):
    # Open zarr array from the TIFF file
    z = tifffile.imread(path, aszarr=True)
    arr = zarr.open(z, mode="r")

    # Slice only every nth pixel *before* loading
    seg = arr[::downsample_factor, ::downsample_factor]

    # Convert to numpy (this triggers reading)
    seg = seg[...]

    plt.figure(figsize=(10, 10))
    plt.imshow(seg, cmap="gray")
    plt.axis("off")
    if title is None:
        title = f"{os.path.basename(path)} (downsampled {downsample_factor}×)"
    plt.title(title)
    plt.show()

def save_avg_expression(adata, df_out, avg_expression_path, cell_type_path = None):
    # Ensure cell IDs align
    df_out = df_out.set_index("c_id").loc[adata.obs_names]

    # Add cell type info to adata.obs
    adata.obs["ct"] = df_out["ct"].values
    X = adata.X

    # For sparse matrices, convert to CSR for efficient indexing
    if not isinstance(X, np.ndarray):
        X = X.tocsr()

    # Create a DataFrame of expression
    expr_df = pd.DataFrame(X.toarray() if not isinstance(X, np.ndarray) else X,
                        index=adata.obs_names,
                        columns=adata.var_names)

    # Add cell type column
    expr_df["ct"] = adata.obs["ct"].values

    # Group by cell type and compute mean
    ct_means = expr_df.groupby("ct").mean(numeric_only=True)

    # map cell types to integers
    ct_to_int = {ct: i for i, ct in enumerate(ct_means.index)}
    ct_means.index = ct_means.index.map(ct_to_int)

    if cell_type_path is None:
        cell_type_path = avg_expression_path.replace(".csv", "_celltype.csv")

    mapping_df = pd.DataFrame(list(ct_to_int.items()), columns=["int_id", "cell_type"])
    mapping_df.to_csv(cell_type_path, index=False, header=[None, "cell_type"])

    ct_means.index.name = None
    ct_means.to_csv(avg_expression_path, index=True)
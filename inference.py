import argparse
import logging
import os
import sys
import natsort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob

from model.adjustments import *
from dataio.dataset_input import DataProcessing
from model.model import Framework
from utils.utils import *

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import csv

from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata


def main(config):
    opts = json_file_to_pyobj(config.config_file)

    if os.path.basename(config.config_file) == "config_demo.json" and config.mode == "predict":
        demo_predict = True
    else:
        demo_predict = False

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    device = get_device(config.gpu_id)

    if config.mode == "val":
        opts_data_sources = opts.data_sources_train_val
    elif config.mode == "predict":
        opts_data_sources = opts.data_sources_predict
    else:
        sys.exit("Invalid --mode: choose either val or predict")

    # Create experiment directories
    make_new = False
    timestamp = get_experiment_id(
        make_new, opts.experiment_dirs.load_dir, config.fold_id
    )
    experiment_path = f"experiments/{timestamp}"
    model_dir = experiment_path + "/" + opts.experiment_dirs.model_dir
    if config.mode == "predict":
        predict_output_dir = (
            experiment_path + "/" + opts.experiment_dirs.predict_output_dir + "/"
        )
    else:
        predict_output_dir = (
            experiment_path + "/" + opts.experiment_dirs.val_output_dir + "/"
        )
    os.makedirs(predict_output_dir, exist_ok=True)

    # Set up the model
    logging.info("Initialising model")

    use_avgexp = opts.comps.avgexp
    use_celltype = opts.comps.celltype
    use_neighb = opts.comps.neighb if use_celltype else False
    use_variants = opts.comps.variants

    if use_celltype:
        classes = opts.data.cell_types
        n_classes = len(classes)
        print(classes)
        print(f"Num cell types {n_classes}")
    else:
        n_classes = 0
        classes = []

    fp_genes = os.path.join(experiment_path, "genes.txt")
    gene_names = read_txt(fp_genes)
    n_genes = len(gene_names)
    print(f"{n_genes} genes")

    if use_avgexp:
        if not opts_data_sources.fp_avgexp:
            sys.exit(f"Please provide {opts_data_sources}.fp_avgexp in config file")
        df_ref_raw = pd.read_csv(opts_data_sources.fp_avgexp, index_col=0)

        gene_names = natsort.natsorted(
            list(set(df_ref_raw.columns.tolist()) & set(gene_names))
        )

        df_ref = pd.DataFrame(0, index=df_ref_raw.index, columns=gene_names)
        for col in gene_names:
            df_ref[col] = df_ref_raw[col]

        n_ref = df_ref.shape[0]
        expr_ref = opts.data.expr_scale * df_ref.to_numpy()
        print("Avgexp shape ", expr_ref.shape)
        expr_ref_torch = torch.from_numpy(expr_ref).float().to(device)

    else:
        n_ref = None
        expr_ref_torch = None
    
    if use_variants:
        if not opts_data_sources.fp_variants:
            sys.exit(f"Please provide {opts_data_sources}.fp_variants in config file")
        df_var_raw = pd.read_csv(opts_data_sources.fp_variants, index_col=0)

        variant_names = natsort.natsorted(df_var_raw.columns.tolist())
        df_var = pd.DataFrame(0, index=df_var_raw.index, columns=variant_names)
        for col in variant_names:
            df_var[col] = df_var_raw[col]

        n_variants = df_var.shape[1]

        # # scaling (you can set opts.data.variant_scale = 1.0 if binary)
        # var_ref = opts.data.variant_scale * df_var.to_numpy()
        # print("Variants shape ", var_ref.shape)
        # var_ref_torch = torch.from_numpy(var_ref).float().to(device)
    else:
        n_variants = 0
        # var_ref_torch = None

    model = Framework(
        n_classes,
        n_genes,
        opts.model.emb_dim,
        device,
        n_ref,
        use_avgexp,
        use_celltype,
        use_neighb,
        use_variants,
        n_variants=n_variants,
    )

    # Get list of model files
    if demo_predict:
        saved_model_epochs = ["demo_predict"]
    elif config.epoch in ["last", "all"]:
        saved_model_paths = glob.glob(f"{model_dir}/epoch_*.pth")
        saved_model_paths = sorted_alphanumeric(saved_model_paths)
        saved_model_names = [
            (os.path.basename(x)).split(".")[0] for x in saved_model_paths
        ]
        saved_model_epochs = [x.split("_")[1] for x in saved_model_names]
        saved_model_epochs = list(set(saved_model_epochs))
        saved_model_epochs = sorted_alphanumeric(saved_model_epochs)
        if config.epoch == "all":
            saved_model_epochs = np.array(saved_model_epochs, dtype="int")
        elif config.epoch == "last":
            saved_model_epochs = np.array(saved_model_epochs[-1], dtype="int")
            saved_model_epochs = [saved_model_epochs]
    else:
        epoch_valid = is_valid_positive_int(config.epoch)
        if epoch_valid:
            saved_model_epochs = [int(config.epoch)]
        else:
            sys.exit("Invalid --epoch: specify an integer, or use 'last' for the most recent, or 'all' for all epochs")

    # Dataloader
    logging.info("Preparing data")

    if config.mode == "val":
        opts_regions = opts.regions_val
    else:
        opts_regions = opts.regions_predict

    predict_dataset = DataProcessing(
        opts_data_sources,
        opts.data,
        opts_regions,
        opts.comps,
        classes,
        gene_names,
        device,
        experiment_path,
        False,
        config.fold_id,
        mode=config.mode,
        demo_predict=demo_predict
    )
    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=opts.training.batch_size,
        shuffle=False,
        num_workers=opts.data.num_workers,
        drop_last=False,
    )

    n_predict_examples = len(dataloader)
    logging.info("Total number of patches: %d" % n_predict_examples)

    logging.info("Begin prediction")

    all_f1_ct = []
    all_corr_mean = []

    with torch.no_grad():

        for epoch_idx, predict_epoch in enumerate(saved_model_epochs):

            if demo_predict:
                load_path = f"{model_dir}/model.pth"
            else:
                load_path = model_dir + "/epoch_%d_model.pth" % (predict_epoch)

            # Restore model
            checkpoint = torch.load(load_path)

            model.load_state_dict(checkpoint["model_state_dict"])
            # epoch = checkpoint["epoch"]
            print("Predict using " + load_path)

            model.to(device)

            model = model.eval()

            pbar = tqdm(dataloader)

            all_gt_ct = []
            all_pr_ct = []
            all_pr_adj = []
            all_ids = []
            all_expr = None
            all_expr_gt = None
            all_comp_est = np.zeros(
                (n_predict_examples * opts.training.batch_size, n_classes)
            )
            all_comp_est_i = 0
            all_comp_est_by_cell = None
            all_area = None

            for (
                batch_nuclei,
                _,
                batch_he_img,
                batch_expr,  # val
                batch_n_cells,
                batch_ct,  # val and use_celltype
                patch_ids,
                batch_var,  # val and use_variants
            ) in pbar:

                batch_nuclei = batch_nuclei.to(device)
                batch_he_img = batch_he_img.to(device)
                batch_expr = batch_expr.to(device)  # all zeros placeholder if predict
                batch_n_cells = batch_n_cells.to(device)
                batch_ct = batch_ct.to(device)  # all zeros placeholder if predict
                patch_ids = patch_ids.to(device)
                batch_var = batch_var.to(device)  # all zeros placeholder if predict

                # assume no GT available, don't calculate performance
                if config.mode == "predict":
                    batch_ct_input = None
                    batch_expr_input = None
                else:
                    batch_ct_input = batch_ct.clone()
                    batch_expr_input = batch_expr.clone()

                (
                    out_cell_type,
                    _,
                    batch_ct_pc,  # all zeros placeholder if predict
                    out_expr,
                    out_expr_immune,
                    out_expr_invasive,
                    _,
                    _,
                    _,
                    _,
                    batch_expr_pc,  # all zeros placeholder if predict
                    comp_estimated,
                    batch_area,
                    patch_ids_pc,
                    batch_var_pc,  # all zeros placeholder if predict
                    fv_variants,
                    out_variants,  # use_variants
                ) = model(
                    batch_he_img,
                    batch_nuclei,
                    batch_n_cells,
                    expr_ref_torch,
                    batch_ct_input,
                    batch_expr_input,
                    patch_ids=patch_ids,
                    batch_var=batch_var
                )

                if out_expr.shape[0] == 0:
                    continue

                if use_neighb:
                    # neighbourhood compositions
                    comp_out_raw = torch.nn.functional.softmax(out_cell_type, dim=1)
                    comp_out_raw = torch.argmax(comp_out_raw, 1)
                    comp_out = torch.nn.functional.one_hot(
                        comp_out_raw, num_classes=n_classes
                    )
                    comp_out = comp_out.float()
                    comp_out = torch.mean(comp_out, 0)

                    # adjustments based on predicted composition
                    adjusted_out_cell_type = []
                    comp_estimated_sum = torch.zeros(n_classes).to(device)

                    target_cts = list(set(opts.data.immune_cts) | set(opts.data.invasive_cts))

                    for i_batch in range(batch_n_cells.shape[0]):
                        n_cells_batch = int(batch_n_cells[i_batch])

                        if n_cells_batch > 0:
                            idx_start = torch.sum(batch_n_cells[:i_batch]).item()
                            idx_end = idx_start + n_cells_batch

                            comp_out_raw_patch = torch.nn.functional.softmax(
                                out_cell_type[idx_start:idx_end, :], dim=1
                            )
                            comp_out_raw_patch = torch.argmax(comp_out_raw_patch, 1)
                            comp_out_patch = torch.nn.functional.one_hot(
                                comp_out_raw_patch, num_classes=n_classes
                            )
                            comp_out_patch = comp_out_patch.float()
                            comp_out_patch = torch.mean(comp_out_patch, 0)

                            # refinements based on predicted compositions

                            # mask highly confident cell predictions
                            pred_logits = F.softmax(
                                out_cell_type[idx_start:idx_end, :], dim=1
                            )

                            high_conf_imm = get_high_conf_cts(opts, pred_logits, "high_confidence_immune_cts")
                            
                            # cell type refinement

                            (
                                adjusted_out_cell_type_patch_immune,
                                idx_swapped_immune_all,
                                idx_swapped_immune,
                            ) = adjust_pr(
                                out_cell_type[idx_start:idx_end, :],
                                comp_estimated[i_batch, :],
                                comp_out_patch,
                                opts.data.cell_types,
                                opts.data.immune_cts,
                                target_cts=target_cts,
                                ignore_idx=high_conf_imm,
                                scale=config.alpha,
                                is_invasive=config.is_invasive
                            )

                            # ensure consistency with expressions
                            if len(idx_swapped_immune) > 0:
                                idx_swapped_immune = [
                                    x + idx_start for x in idx_swapped_immune
                                ]
                                out_expr[idx_swapped_immune] = out_expr_immune[
                                    idx_swapped_immune
                                ]

                            if config.is_invasive:
                                high_conf_invasive = get_high_conf_cts(opts, pred_logits, "high_confidence_invasive_cts")

                                # malignant
                                (
                                    adjusted_out_cell_type_patch_invasive,
                                    idx_swapped_invasive_all,
                                    idx_swapped_invasive,
                                ) = adjust_pr(
                                    out_cell_type[idx_start:idx_end, :],
                                    comp_estimated[i_batch, :],
                                    comp_out_patch,
                                    opts.data.cell_types,
                                    opts.data.invasive_cts,
                                    target_cts=target_cts,
                                    ignore_idx=high_conf_invasive,
                                    scale=10000,
                                    is_invasive=config.is_invasive
                                )

                                if len(idx_swapped_invasive) > 0:
                                    idx_swapped_invasive = [
                                        x + idx_start for x in idx_swapped_invasive
                                    ]
                                    out_expr[idx_swapped_invasive] = out_expr_invasive[
                                        idx_swapped_invasive
                                    ]

                            adjusted_out_cell_type_patch = (
                                adjusted_out_cell_type_patch_immune.copy()
                            )
                            if config.is_invasive:
                                for index in idx_swapped_invasive_all:
                                    adjusted_out_cell_type_patch[index] = (
                                        adjusted_out_cell_type_patch_invasive[index]
                                    )

                            adjusted_out_cell_type.extend(adjusted_out_cell_type_patch)

                            # neighbourhood compositions
                            comp_estimated_sum += n_cells_batch * comp_estimated[i_batch, :]

                            all_comp_est[all_comp_est_i, :] = (
                                n_cells_batch
                                * comp_estimated[i_batch, :].detach().cpu().numpy()
                            )

                            repeated_by_cells = np.tile(
                                all_comp_est[all_comp_est_i, :], (n_cells_batch, 1)
                            )

                            # also for each corresponding cell
                            if all_comp_est_by_cell is None:
                                all_comp_est_by_cell = repeated_by_cells.copy()
                            else:
                                all_comp_est_by_cell = np.vstack(
                                    (all_comp_est_by_cell, repeated_by_cells)
                                )

                            all_comp_est_i += 1

                    all_pr_adj.extend(adjusted_out_cell_type)

                # save predicted expr
                out_expr = (1 / opts.data.expr_scale) * out_expr.detach().cpu().numpy()
                out_expr_gt = (
                    1 / opts.data.expr_scale
                ) * batch_expr_pc.detach().cpu().numpy()

                if all_expr is None:
                    all_expr = out_expr.copy()
                    all_expr_gt = out_expr_gt.copy()
                else:
                    all_expr = np.vstack((all_expr, out_expr))
                    all_expr_gt = np.vstack((all_expr_gt, out_expr_gt))

                # nuclei areas
                if all_area is None:
                    all_area = batch_area.detach().cpu().numpy()
                else:
                    all_area = np.hstack((all_area, batch_area.detach().cpu().numpy()))

                # cell IDs
                patch_ids_pc = list(patch_ids_pc.detach().cpu().numpy())
                all_ids.extend(patch_ids_pc)

                if use_celltype:
                    # cell types auxiliary predictions and ground truth
                    batch_ct_pc = list(batch_ct_pc.detach().cpu().numpy())
                    out_cell_type_ct = torch.argmax(out_cell_type, dim=1)
                    out_cell_type_ct = list(out_cell_type_ct.detach().cpu().numpy())
                    all_gt_ct.extend(batch_ct_pc)
                    all_pr_ct.extend(out_cell_type_ct)

            # duplicates from overlapped regions - keep the prediction with largest area
            combined = np.vstack((np.array(all_ids), all_area, np.arange(len(all_ids))))
            combined = np.transpose(combined)
            combined_df = pd.DataFrame(
                combined, index=np.arange(len(all_ids)), columns=["id", "area", "index"]
            )
            sorted_df = combined_df.sort_values(by="area", ascending=False)
            # Drop duplicate rows based on the 'id' column, keeping only the first occurrence (largest area)
            unique_df = sorted_df.drop_duplicates(subset="id", keep="first")

            unique_indices = unique_df["index"].astype(int).tolist()

            unique_indices = sorted(unique_indices)
            all_ids = [all_ids[ui] for ui in unique_indices]

            if use_celltype:
                all_gt_ct = [all_gt_ct[ui] for ui in unique_indices]
                all_pr_ct = [all_pr_ct[ui] for ui in unique_indices]
            if use_neighb:
                all_pr_adj = [all_pr_adj[ui] for ui in unique_indices]

            all_expr = all_expr[unique_indices, :]
            all_expr_gt = all_expr_gt[unique_indices, :]

            # save predicted expr
            df_all_expr = pd.DataFrame(all_expr, index=all_ids, columns=gene_names)
            fp_expr_out = f"{predict_output_dir}/epoch_{predict_epoch}_expr.csv"
            df_all_expr.to_csv(fp_expr_out)
            print(
                f"Saved predicted expressions of {len(unique_indices)} cells to {fp_expr_out}"
            )

            df_all_expr_gt = pd.DataFrame(
                all_expr_gt, index=all_ids, columns=gene_names
            )

            if use_variants:
                out_variants_np = out_variants.detach().cpu().numpy()
                df_all_variants = pd.DataFrame(out_variants_np, index=all_ids, columns=variant_names)
                fp_var_out = f"{predict_output_dir}/epoch_{predict_epoch}_variants.csv"
                df_all_variants.to_csv(fp_var_out)
                print(f"Saved predicted variants of {len(unique_indices)} cells to {fp_var_out}")

            if config.mode == "val":

                print("***expression correlation***")
                all_rowwise = df_all_expr_gt.corrwith(df_all_expr, axis=0)
                mean_corr = np.nanmean(all_rowwise)
                print(f"PCC mean: {mean_corr}")
                all_corr_mean.append(mean_corr)

                if use_celltype:
                    f1 = f1_score(all_gt_ct, all_pr_ct, average=None)
                    f1_mean = np.mean(f1)
                    all_f1_ct.append(f1_mean)
                
                if use_variants:
                    # Binary (0/1) prediction vs truth
                    preds_bin = (out_variants_np > 0.5).astype(int)
                    gt_bin = batch_var_pc.detach().cpu().numpy()

                    variant_acc = (preds_bin == gt_bin).mean()
                    print(f"Variant prediction accuracy: {variant_acc:.4f}")


    if config.mode == "val":
        # best epoch, better performance should have lower ranks
        print("***best epoch***")
        rank_corr = rankdata(-np.array(all_corr_mean), method="min")
        if use_celltype:
            rank_f1 = rankdata(-np.array(all_f1_ct), method="min")
            ranksums = rank_f1 + rank_corr
        else:
            ranksums = rank_corr.copy()
        best_idx = np.argmin(ranksums)
        best_epoch = saved_model_epochs[best_idx]
        print(f"PCC best epoch {best_epoch} mean {all_corr_mean[best_idx]}")


logging.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--epoch",
        default="all",
        type=str,
        help="epoch to run: specify an integer, or use 'last' for the most recent, or 'all' for all epochs",
    )
    parser.add_argument(
        "--mode",
        default="predict",
        type=str,
        help="predict or val",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="which GPU to use",
    )
    parser.add_argument(
        "--alpha",
        default=1,
        type=float,
        help="hyperparameter for neighbourhood composition refinement",
    )
    parser.add_argument(
        "--is_invasive", action=argparse.BooleanOptionalAction, default=False
    )

    config = parser.parse_args()
    main(config)


import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast
import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import layer_info, expr_parameters

import sys
sys.path.append("..")

csv_dir = "reference_csv_files"
ref_files = ["Resnet_Conv5_2.csv"]
layers = [5]
our_test_dir = "./"


def get_spm_bank_and_size(stride, spatial_factor, rf_factor, spm_factor):
    prod = [x*y*z for x, y, z in zip(spatial_factor, rf_factor, spm_factor)]
    n, m, c, oy, ox, fy, fx = prod
    S = stride
    weights_SPM = m * c * fx * fy  # W
    ifmap_SPM = n * c * ((ox-1)*S + fx) * ((oy-1)*S + fy)  # I
    psum_SPM = n * m * ox * oy  # O

    BYTES_PER_DATA = 2
    TOTAL_BYTES_IN_BANK = 2048

    total_banks_weights = math.ceil(
        weights_SPM * BYTES_PER_DATA / TOTAL_BYTES_IN_BANK)
    total_banks_ifmap = math.ceil(
        ifmap_SPM * BYTES_PER_DATA / TOTAL_BYTES_IN_BANK)
    total_banks_psum = math.ceil(
        psum_SPM * BYTES_PER_DATA / TOTAL_BYTES_IN_BANK)

    data_banks_alloc_SPM = sum(
        [total_banks_weights, total_banks_ifmap, total_banks_psum])
    size_in_bytes = sum([weights_SPM, ifmap_SPM, psum_SPM])*BYTES_PER_DATA

    return data_banks_alloc_SPM, size_in_bytes


def get_joined_df(csv_dir, ref_files, our_test_dir):

    ref_dfs = []
    for ref_file_name in ref_files:
        df = pd.read_csv(csv_dir+"/"+ref_file_name)
        ref_dfs.append(df)

    for i, ref_df in enumerate(ref_dfs):
        print("processing", ref_files[i])
        layer_arguments = layer_info.resnet_parameters[layers[i]]
        env = expr_parameters.Environment()
        env.pe_pipeline_stages = 1
        layer = ConvLayer(env, **layer_arguments)

        loop_IVs = ["N", "M", "C", "Ox", "Oy", "Fx", "Fy"]

        ref_df["energy_from_dMazeRunner"] = np.nan
        ref_df["cycle_from_dMazeRunner"] = np.nan
        ref_df["edp_from_dMazeRunner"] = np.nan
        ref_df["energy_diff_percent"] = np.nan
        ref_df["layer"] = layers[i]

        ref_df.drop(ref_df[ref_df["Verify Tiling"]
                           != True].index, inplace=True)

        for index, row in ref_df.iterrows():
            if not row["Verify Tiling"]:
                continue

            #set tiling
            dram_tiling = ast.literal_eval(row["DRAM_tiling"])
            spm_tiling = ast.literal_eval(row["SPM_tiling"])
            rf_tiling = ast.literal_eval(row["RF_tiling"])
            spatial_tiling = ast.literal_eval(row["Spatial_tiling"])
            for idx in range(len(loop_IVs)):
                tiling_factor = [dram_tiling[idx], spm_tiling[idx],
                                 rf_tiling[idx], spatial_tiling[idx]]
                layer.set_tiling(loop_IVs[idx], tiling_factor)

            #find key from ordering
            spm_ordering = ast.literal_eval(row["SPM_schedule"])
            dram_ordering = ast.literal_eval(row["DRAM_schedule"])

            spm_ordering = tuple([iv.title() for iv in spm_ordering])
            dram_ordering = tuple([iv.title() for iv in dram_ordering])

            spm_reuse_factor = layer.determine_data_reuse(
                "SPM", user_ordering=spm_ordering)[0]
            dram_reuse_factor = layer.determine_data_reuse(
                "DRAM", user_ordering=dram_ordering)[0]

            spm_ordering = layer.get_ordering_from_reuse_factor(
                spm_reuse_factor, "SPM")
            dram_ordering = layer.get_ordering_from_reuse_factor(
                dram_reuse_factor, "DRAM")
            key = (dram_ordering, spm_ordering)

            cycles_of_all_orderings = layer.get_Cycles_One_Layer()
            energy_of_all_orderings = layer.get_Energy_One_Layer()

            ##### use the set of ordering that minimizes energy
            energy, dram_ordering, spm_ordering = layer.get_min_energy()
            key = (dram_ordering, spm_ordering)
            #####

            cycle, energy = cycles_of_all_orderings[key], energy_of_all_orderings[key]
            ref_df.at[index, "energy_from_dMazeRunner"] = energy
            ref_df.at[index, "cycle_from_dMazeRunner"] = cycle
            ref_df.at[index, "edp_from_dMazeRunner"] = energy * cycle

            # energy distribution
            energy_distributions_of_all_orderings = layer.get_Energy_Distribution()
            energy_MAC, energy_RF, energy_NOC, energy_SPM, energy_DRAM = energy_distributions_of_all_orderings[
                key]
            ref_df.at[index, "energy_MAC_dMazeRunner"] = energy_MAC
            ref_df.at[index, "energy_RF_dMazeRunner"] = energy_RF
            ref_df.at[index, "energy_NOC_dMazeRunner"] = energy_NOC
            ref_df.at[index, "energy_SPM_dMazeRunner"] = energy_SPM
            ref_df.at[index, "energy_DRAM_dMazeRunner"] = energy_DRAM

            energy_from_ref = row["Energy"]
            ref_df.at[index, "energy_diff_percent"] = 100 * \
                abs(energy-energy_from_ref)/energy_from_ref

            stride = layer.strides
            n_banks, size_in_bytes = get_spm_bank_and_size(
                stride, spatial_tiling, rf_tiling, spm_tiling)
            ref_df.at[index, "spm_banks_yang_et_al"] = n_banks
            ref_df.at[index, "spm_size_in_bytes_yang_et_al"] = size_in_bytes

    final_df = pd.DataFrame()

    for layer_index, ref_df in enumerate(ref_dfs):
        def change_dataflow_name(old_name):
            name = old_name.replace("IC", "C")
            name = name.replace("OC", "M")
            name = name.replace("ON", "N")
            tokens = name.strip().split("_")
            if len(tokens) not in [2, 4]:
                return None
            if len(tokens) == 4:
                tokens = tokens[:2]
            x, y = tokens
            return (y.title() + " | " + x.title())

        def parse_config(old_name):
            tokens = old_name.strip().split("_")
            if len(tokens) == 4:
                return tokens[2]+"_"+tokens[3]
            else:
                return None

        columns = list(ref_df.columns.values)
        columns.remove("layer")
        new_column_order = ["layer"] + columns
        ref_df = ref_df[new_column_order]

        ref_df["dataflow_str"] = ref_df["Data_flow_mechanism"].apply(
            change_dataflow_name)
        ref_df["dataflow_config"] = ref_df["Data_flow_mechanism"].apply(
            parse_config)

        final_df = pd.concat([final_df, ref_df])

    final_df = final_df.rename(index=str, columns={
        "dataflow_str": "dataflow",
        "Energy": "energy_theirs",
        "energy_from_dMazeRunner": "energy_dMazeRunner",
        "energy_best": "energy_optimal",
        "cycle_from_dMazeRunner": "cycle_dMazeRunner",
        "edp_from_dMazeRunner": "edp_dMazeRunner",
        "energy_diff_percent": "energy_diff_%",
    })

    final_df["energy_RF_diff_%"] = (
        final_df["Energy_RF"] - final_df["energy_RF_dMazeRunner"]) / final_df["energy_RF_dMazeRunner"] * 100
    final_df["energy_NOC_diff_%"] = (
        final_df["Energy_NoC"] - final_df["energy_NOC_dMazeRunner"]) / final_df["energy_NOC_dMazeRunner"] * 100
    final_df["energy_SPM_diff_%"] = (
        final_df["Energy_SPM"] - final_df["energy_SPM_dMazeRunner"]) / final_df["energy_SPM_dMazeRunner"] * 100
    final_df["energy_DRAM_diff_%"] = (
        final_df["Energy_DRAM"] - final_df["energy_DRAM_dMazeRunner"]) / final_df["energy_DRAM_dMazeRunner"] * 100

    return final_df


def get_enery_error(df):
    layer_error_df = pd.DataFrame()
    for i in np.sort(df["layer"].unique()):
        mask = df["layer"] == i
        layer_error_df.loc[i, "layer"] = i
        layer_error_df.loc[i,
                           "energy_diff_%_avg"] = df[mask]["energy_diff_%"].mean()
        layer_error_df.loc[i,
                           "energy_ratio_avg"] = df[mask]["energy_ratio"].mean()

    dataflow_error_df = pd.DataFrame()
    for i, dataflow in enumerate(df["dataflow"].unique()):
        mask = df["dataflow"] == dataflow
        dataflow_error_df.loc[i, "dataflow"] = dataflow
        dataflow_error_df.loc[i,
                              "energy_diff_%_avg"] = df[mask]["energy_diff_%"].mean()
        dataflow_error_df.loc[i,
                              "energy_ratio_avg"] = df[mask]["energy_ratio"].mean()
    dataflow_error_df = dataflow_error_df.sort_values("energy_diff_%_avg")

    return layer_error_df, dataflow_error_df


def get_energy_breakdown(df):
    layer_df = pd.DataFrame()
    for i in np.sort(df["layer"].unique()):
        mask = df["layer"] == i
        layer_df.loc[i, "layer"] = i
        layer_df.loc[i, "energy_RF_diff_%"] = df[mask]["energy_RF_diff_%"].mean()
        layer_df.loc[i, "energy_NOC_diff_%"] = df[mask]["energy_NOC_diff_%"].mean()
        layer_df.loc[i, "energy_SPM_diff_%"] = df[mask]["energy_SPM_diff_%"].mean()
        layer_df.loc[i, "energy_DRAM_diff_%"] = df[mask]["energy_DRAM_diff_%"].mean()
        layer_df.loc[i, "energy_total_diff_%"] = df[mask]["energy_diff_%"].mean()

    dataflow_df = pd.DataFrame()
    for i, dataflow in enumerate(df["dataflow"].unique()):
        mask = df["dataflow"] == dataflow
        dataflow_df.loc[i, "dataflow"] = dataflow
        dataflow_df.loc[i,
                        "energy_RF_diff_%"] = df[mask]["energy_RF_diff_%"].mean()
        dataflow_df.loc[i,
                        "energy_NOC_diff_%"] = df[mask]["energy_NOC_diff_%"].mean()
        dataflow_df.loc[i,
                        "energy_SPM_diff_%"] = df[mask]["energy_SPM_diff_%"].mean()
        dataflow_df.loc[i,
                        "energy_DRAM_diff_%"] = df[mask]["energy_DRAM_diff_%"].mean()
        dataflow_df.loc[i,
                        "energy_total_diff_%"] = df[mask]["energy_diff_%"].mean()

    return layer_df, dataflow_df


def get_columns(t=None):
    if t == "energy":
        columns = ["layer", "dataflow", "dataflow_config",
                   "energy_diff_%", "energy_theirs", "energy_dMazeRunner", "energy_optimal", "energy_ratio"]
    elif t == "energy_breakdown":
        columns = ["layer", "dataflow", "Energy_RF", "energy_MAC_dMazeRunner", "energy_RF_dMazeRunner", "energy_RF_diff_%",
                   "Energy_NoC", "energy_NOC_dMazeRunner", "energy_NOC_diff_%",
                   "Energy_SPM", "energy_SPM_dMazeRunner", "energy_SPM_diff_%",
                   "Energy_DRAM", "energy_DRAM_dMazeRunner", "energy_DRAM_diff_%",
                   "energy_theirs", "energy_dMazeRunner"]
        columns = ["layer", "dataflow", "dataflow_config", "energy_MAC_dMazeRunner", "Energy_RF", "energy_RF_dMazeRunner",
                   "Energy_NoC", "energy_NOC_dMazeRunner",
                   "Energy_SPM", "energy_SPM_dMazeRunner",
                   "Energy_DRAM", "energy_DRAM_dMazeRunner",
                   "energy_theirs", "energy_dMazeRunner"]
    elif t == "simple":
        columns = ["layer", "dataflow", "dataflow_config",
                   "energy_dMazeRunner", "energy_optimal", "energy_ratio",
                   "cycle_dMazeRunner", "cycle_optimal", "cycle_ratio",
                   "edp_dMazeRunner", "edp_optimal", "edp_ratio",
                   "spm_banks_yang_et_al", "spm_size_in_bytes_yang_et_al"]

    else:
        columns = ["layer", "dataflow", "dataflow_config", "energy_diff_%",
                   "energy_theirs", "energy_dMazeRunner", "energy_optimal", "energy_ratio",
                   "cycle_dMazeRunner", "cycle_optimal", "cycle_ratio", "edp_ratio"]
    return columns


def get_energy_breakdown_xls(df):
    mask = df["layer"] == 5
    columns = ["Energy_RF", "Energy_NoC", "Energy_SPM", "Energy_DRAM"]
    df[mask][columns].to_xls("out.xls")


def main():

    df = get_joined_df(csv_dir, ref_files, our_test_dir)

    columns = get_columns("energy_breakdown")
    df_final = df[columns].rename(index=str, columns={
        "energy_dMazeRunner": "energy_total_dMazeRunner",
        "energy_optimal": "energy_optimal_from_dMazeRunner",
        "cycle_dMazeRunner": "cycle_from_yang_et_al",
        "cycle_optimal": "cycle_optimal_from_dMazeRunner",
        "edp_dMazeRunner": "edp_from_yang_et_al",
        "edp_optimal": "edp_optimal_from_dMazeRunner",
        "Energy_RF": "energy_RF_yang_et_al",
        "Energy_NoC": "energy_NOC_yang_et_al",
        "Energy_SPM": "energy_SPM_yang_et_al",
        "Energy_DRAM": "energy_DRAM_yang_et_al",
        "energy_theirs": "energy_total_yang_et_al",
    })

    df_final.to_excel("out.xlsx")


if __name__ == "__main__":
    main()

import tvm
import multiprocessing as mp
import time
import datetime
import sys
import random
import os
from pprint import pprint

from functools import reduce
import operator
import math
import tqdm
import pickle
import concurrent.futures
import numpy as np
import pandas as pd

from dMazeRunner.common import layer_info, expr_parameters
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner import evaluate_all_combinations

expr_name = ""

energy_rf = {
    16: 0.03,
    32: 0.06,
    64: 0.12,
    128: 0.24,
    256: 0.48,
    512: 0.96,
    1024: 1.92,
}

energy_spm = {
    16: 4,
    32: 6,
    64: 9,
    128: 13.5,
    256: 20.25,
    512: 30.375,
    1024: 45.5625,
}

n_PEs = [256]

TOTAL_BYTES_IN_BANK = 2048

def main(args):
    layer_index = args[0]
    for n_PE in n_PEs:
        for spm_size, spm_energy in energy_spm.items():
            for rf_size, rf_energy in energy_rf.items():
                spm_banks = spm_size * 1024 / 2 / TOTAL_BYTES_IN_BANK
                expr_name = "layer{}_{}_PE+{}_SPM+{}_RF".format(layer_index, n_PE, spm_size, rf_size)
                print("evaluating", expr_name)
                args = [layer_index, layer_index, expr_name]
                env_config = {
                    "cgra_size": n_PE,
                    "spm_banks": spm_banks,
                    "rf_size": rf_size,
                    "spm_energy": spm_energy,
                    "rf_energy": rf_energy
                }
                evaluate_all_combinations.main(args, env_config=env_config)


def parse(layer_index):
    columns = ("spm_size", "rf_size", "n_PE", "edp", "total_evaluated",
        "start_time", "end_time")
    final_df = pd.DataFrame(columns=columns)
    for n_PE in n_PEs:
        for spm_size, spm_energy in energy_spm.items():
            for rf_size, rf_energy in energy_rf.items():
                expr_name = "layer{}_{}_PE+{}_SPM+{}_RF".format(layer_index, n_PE, spm_size, rf_size)
                file_name = "out/{}/layer_{}_out_dict.p".format(expr_name, layer_index)
                df = pickle.load(open(file_name, "rb"))
                config = df["config"]
                result_df = df["result"]

                start_time = config["start_time"]
                end_time = config["end_time"]
                if len(result_df) > 0:
                    min_row_index = result_df["edp"].idxmin()
                    min_row = result_df.loc[min_row_index]
                    min_edp =min_row["edp"]
                    total_evaluated = sum(result_df["evaluated"])
                    print(expr_name, min_edp, total_evaluated, start_time, end_time)
                else:
                    min_edp, total_evaluated = None, 0
                    print(expr_name, None)

                final_df.loc[len(final_df)] = [spm_size, rf_size, n_PE, min_edp,
                    total_evaluated, start_time, end_time]

    final_df = final_df.astype({
        "spm_size": int,
        "rf_size": int,
        "n_PE": int,
        "total_evaluated": int,
    })

    file_name = "dse_output_layer_{}.xlsx".format(layer_index)
    final_df.to_excel(file_name)



def print_help():
    print("usage:")
    print("python tune_memory.py run <layer_index>")
    print("python tune_memory.py parse <layer_index>")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print_help()
        exit()

    if sys.argv[1] == "parse":
        parse(int(sys.argv[2]))
    elif sys.argv[1] == "run":
        main(sys.argv[2:])
    else:
        print_help()

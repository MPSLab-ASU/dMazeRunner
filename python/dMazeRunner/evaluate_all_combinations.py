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

from functools import lru_cache

from dMazeRunner.common import layer_info, expr_parameters
from dMazeRunner.dataflow import ConvLayer

expr_name = ""

IV_ORDER = ["N", "M", "C", "Ox", "Oy", "Fx", "Fy"]
CGRA_SIZE = 256
RF_SIZE = 256
BYTES_PER_DATA = 2			# same for all data operands
TOTAL_BANKS_IN_SPM = 32 	# banks for one of the double buffers
TOTAL_BYTES_IN_BANK = 2048

PE_UTILIZATION = 0.8
RF_UTILIZATION = 0.8
SPM_UTILIZATION = 0.6

PRUNE_NO_FEATURE_DRAM = True
PRUNE_NO_REDUCTION = True

def factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def of_bucket(lst, depth=0):
	""" return all combinations of items in buckets """
	for item in lst[0]:
		if len(lst) > 1:
			for result in of_bucket(lst[1:], depth+1):
				yield (item, ) + result
		else:
			yield (item, )


def valid_spatial(spatial_factor):
    n, m, c, ox, oy, fx, fy = spatial_factor
    pe_in_use = reduce(operator.mul, spatial_factor)
    size_constraint = pe_in_use <= CGRA_SIZE
    utilization_constraint = (pe_in_use/CGRA_SIZE) >= PE_UTILIZATION
    no_reduction = fx == 1 and fy == 1 and c == 1 if PRUNE_NO_REDUCTION else True
    return size_constraint and utilization_constraint and no_reduction


def valid_rf(stride, rf_factor):
    regs_alloc = 0
    n, m, c, ox, oy, fy, fx = rf_factor
    S = stride
    regs_alloc += n * c * ((ox-1)*S + fx) * ((oy-1)*S + fy) #I
    regs_alloc += m * c * fx * fy #W
    regs_alloc += n * m * ox * oy #O

    size_constraint = regs_alloc <= RF_SIZE
    utilization_constraint = (regs_alloc/RF_SIZE) >= RF_UTILIZATION
    return size_constraint and utilization_constraint


def valid_spm(stride, spatial_rf_factor, spm_factor):
    prod = [ x*y for x,y in zip(spatial_rf_factor, spm_factor)]
    n, m, c, oy, ox, fy, fx = prod
    S = stride
    weights_SPM = n * c * ((ox-1)*S + fx) * ((oy-1)*S + fy) #I
    ifmap_SPM = m * c * fx * fy #W
    psum_SPM = n * m * ox * oy #O

    total_banks_weights = math.ceil(weights_SPM * BYTES_PER_DATA / TOTAL_BYTES_IN_BANK)
    total_banks_ifmap = math.ceil(ifmap_SPM * BYTES_PER_DATA / TOTAL_BYTES_IN_BANK)
    total_banks_psum = math.ceil(psum_SPM * BYTES_PER_DATA / TOTAL_BYTES_IN_BANK)

    data_banks_alloc_SPM = sum([total_banks_weights, total_banks_ifmap, total_banks_psum])
    data_alloc_SPM = sum([weights_SPM, ifmap_SPM, psum_SPM]) * BYTES_PER_DATA

    bank_constraint = data_banks_alloc_SPM <= TOTAL_BANKS_IN_SPM
    size_constraint = data_alloc_SPM <= TOTAL_BANKS_IN_SPM * TOTAL_BYTES_IN_BANK
    utilization_constraint = (data_alloc_SPM / (TOTAL_BANKS_IN_SPM * TOTAL_BYTES_IN_BANK)) >= SPM_UTILIZATION

    return bank_constraint and size_constraint and utilization_constraint


def valid_dram(dram_factor):
    n, m, c, ox, oy, fx, fy = dram_factor
    no_feature_constraint = fx == 1 and fy == 1 if PRUNE_NO_FEATURE_DRAM else True
    return no_feature_constraint


#def run_thread(layer_index, tc_list, spatial_factor):
def run_thread(params):
    layer_index, tc_list, spatial_factor, env_config = params

    if env_config != None:
        env = expr_parameters.Environment(
            rf_energy=env_config["rf_energy"],
            spm_energy=env_config["spm_energy"],
        )
    else:
        env = expr_parameters.Environment()

    env.pe_pipeline_stages = 1
    layer_args = layer_info.resnet_parameters[layer_index]
    layer = ConvLayer(env, **layer_args)
    stride = layer.strides

    tc_list_after_spatial = [ int(x/y) for x, y in zip(tc_list, spatial_factor) ]
    tc_list_factors_spatial = [ factors(tc) for tc in tc_list_after_spatial ]

    min_energy = float("inf")
    min_energy_sequence = None
    min_edp = float("inf")
    min_edp_sequence = None

    evaluated = 0

    for rf_factor in of_bucket(tc_list_factors_spatial):
        if not valid_rf(stride, rf_factor):
            continue

        tc_list_after_rf = [ int(x/y) for x, y in zip(tc_list_after_spatial, rf_factor)]
        tc_list_factors_rf = [ factors(tc) for tc in tc_list_after_rf]

        for spm_factor in of_bucket(tc_list_factors_rf):
            spatial_rf_factor = ( x*y for x, y in zip(spatial_factor, rf_factor))
            if not valid_spm(stride, spatial_rf_factor, spm_factor):
                continue

            tc_list_after_spm = tuple([ int(x/y) for x, y in zip(tc_list_after_rf, spm_factor) ])
            dram_factor = tc_list_after_spm
            if not valid_dram(dram_factor):
                continue
            #assert tc_list == [ x*y*z*w for x,y,z,w in zip(spatial_factor, rf_factor, spm_factor, dram_factor)]

            evaluated += 1

            for idx in range(len(IV_ORDER)):
                tiling_factor = [dram_factor[idx], spm_factor[idx], rf_factor[idx], spatial_factor[idx]]
                layer.set_tiling(IV_ORDER[idx], tiling_factor)

            edp, energy, cycle = layer.get_min_edp_energy_cycle()
            if edp < min_edp:
                min_edp = edp
                min_edp_sequence = (dram_factor, spm_factor, rf_factor, spatial_factor)

            if energy < min_energy:
                min_energy = energy
                min_energy_sequence = (dram_factor, spm_factor, rf_factor, spatial_factor)

    """
    file_name = "_".join([ str(i) for i in spatial_factor ])
    file_name = str(layer_index) + "_" + file_name
    output_file = "out/" + expr_name + "/" + file_name + ".txt"
    with open(output_file, "w") as outFile:
        line = ",".join([ str(item) for item in [min_edp, min_edp_sequence, min_energy, min_energy_sequence] ])
        outFile.write(line)
    """

    return min_edp, min_edp_sequence, min_energy, min_energy_sequence, evaluated


def dump_result(result_df, layer_index, start_time, end_time):
    print(result_df)
    config = {
        "PE_UTILIZATION": PE_UTILIZATION,
        "RF_UTILIZATION": RF_UTILIZATION,
        "SPM_UTILIZATION": SPM_UTILIZATION,
        "PRUNE_NO_FEATURE_DRAM": PRUNE_NO_FEATURE_DRAM,
        "PRUNE_NO_REDUCTION": PRUNE_NO_REDUCTION,
        "RF_SIZE": RF_SIZE,
        "SPM_BANKS": TOTAL_BANKS_IN_SPM,
        "CGRA_SIZE": CGRA_SIZE,
        "start_time": start_time,
        "end_time": end_time,
        "total_evaluations": sum( result_df["evaluated"] ),
    }
    dump_dict = {
        "config": config,
        "result": result_df
    }
    pickle.dump(dump_dict, open("out/"+expr_name+"/layer_"+str(layer_index)+"_out_dict.p", "wb"))


def generate_combinations(layer_index, tc_list, env_config=None):
    #pool = mp.Pool(processes=mp.cpu_count()*2)
    tc_list_factors = [ factors(tc) for tc in tc_list ]
    spatial_factors = [ factor for factor in of_bucket(tc_list_factors) if valid_spatial(factor) ]
    params = [ (layer_index, tc_list, spatial_factor, env_config) for spatial_factor in spatial_factors ]

    output_dir_name = "out"
    os.makedirs(output_dir_name, exist_ok=True)

    output_dir_name = "out/"+expr_name
    os.makedirs(output_dir_name, exist_ok=True)

    pbar = tqdm.tqdm(total=len(params))
    columns = ["edp", "edp_seq", "energy", "energy_seq", "evaluated"]
    result_df = pd.DataFrame(columns=columns)

    multi_processing = True
    if multi_processing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for ret_tuple in executor.map(run_thread, params):
                pbar.update()
                if ret_tuple[1] == None:
                    continue
                result_df.loc[len(result_df)] = ret_tuple
    else:
        for param in params:
            ret_tuple = run_thread(param)
            pbar.update()
            if ret_tuple[1] == None:
                continue
            result_df.loc[len(result_df)] = ret_tuple


    return result_df


def run_dataflow(layer_index, env_config=None):
    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    layer_args = layer_info.resnet_parameters[layer_index]
    layer = ConvLayer(env, **layer_args)

    base_TCs = layer.base_TCs
    tcs = [ base_TCs[iv] for iv in IV_ORDER ]
    return generate_combinations(layer_index, tcs, env_config)


def main(argv, env_config=None):
    if len(argv) != 3:
        usage = "python run_expr.py <layer_index_start> <layer_index_end> <expr_name>"
        example = "python run_expr.py 4 5 test will evaluate all 21 dataflow for layer 4 and 5 and save the result in directory test"
        print("usage:", usage)
        print("example:", example)
        exit()

    n_layer = len(layer_info.resnet_parameters)
    layer_index_start = int(argv[0])
    layer_index_end = min(int(argv[1])+1, n_layer)

    global expr_name
    expr_name = argv[2]

    if env_config != None:
        global CGRA_SIZE, TOTAL_BANKS_IN_SPM, RF_SIZE
        CGRA_SIZE = env_config["cgra_size"]
        TOTAL_BANKS_IN_SPM = env_config["spm_banks"]
        RF_SIZE = env_config["rf_size"]

    layers_to_evaluate = range(layer_index_start, layer_index_end)
    for layer_index in layers_to_evaluate:
        start_time = datetime.datetime.now()
        result_df = run_dataflow(layer_index, env_config)
        end_time = datetime.datetime.now()
        dump_result(result_df, layer_index, start_time, end_time)


if __name__ == "__main__":
    main(sys.argv[1:])

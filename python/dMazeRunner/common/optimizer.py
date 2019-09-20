import tvm
from dMazeRunner.dataflow import ConvLayer, GemmLayer
from dMazeRunner.common import layer_info, expr_parameters
import multiprocessing as mp
import time
import datetime
import sys
import random
import os
from pprint import pprint
from copy import deepcopy

from functools import reduce
import operator
import math
import tqdm
import pickle
import concurrent.futures
import numpy as np
import pandas as pd

from functools import lru_cache

expr_name = ""
PARAMS = None

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


def valid_spatial(layer, spatial_factor, expr_params=None):
    if expr_params != None:
        global PARAMS
        PARAMS = expr_params
    pe_in_use = reduce(operator.mul, spatial_factor)
    size_constraint = pe_in_use <= PARAMS.CGRA_SIZE
    utilization_constraint = (pe_in_use/PARAMS.CGRA_SIZE) >= PARAMS.PE_UTILIZATION
    no_reduction = True

    if PARAMS.PRUNE_NO_REDUCTION and type(layer) == ConvLayer:
        n, m, c, ox, oy, fx, fy = spatial_factor
        no_reduction = fx == 1 and fy == 1 and c == 1 if PARAMS.PRUNE_NO_REDUCTION else True

    elif PARAMS.PRUNE_NO_REDUCTION and type(layer) == GemmLayer:
        x, y, k = spatial_factor
        no_reduction = k == 1 if PARAMS.PRUNE_NO_REDUCTION else True

    return size_constraint and utilization_constraint and no_reduction


def valid_rf(layer, stride, rf_factor):
    regs_alloc = 0

    if type(layer) == ConvLayer:
        n, m, c, ox, oy, fy, fx = rf_factor
        S = stride
        regs_alloc += n * c * ((ox-1)*S + fx) * ((oy-1)*S + fy) #I
        regs_alloc += m * c * fx * fy #W
        regs_alloc += n * m * ox * oy #O

    elif type(layer) == GemmLayer:
        x, y, k = rf_factor
        regs_alloc += x * y
        regs_alloc += k * y
        regs_alloc += x * k

    size_constraint = regs_alloc <= PARAMS.RF_SIZE
    utilization_constraint = (regs_alloc/PARAMS.RF_SIZE) >= PARAMS.RF_UTILIZATION
    return size_constraint and utilization_constraint


def valid_spm(layer, stride, spatial_rf_factor, spm_factor):
    prod = [ x*y for x,y in zip(spatial_rf_factor, spm_factor)]

    if type(layer) == ConvLayer:
        n, m, c, oy, ox, fy, fx = prod
        S = stride
        weights_SPM = n * c * ((ox-1)*S + fx) * ((oy-1)*S + fy) #I
        ifmap_SPM = m * c * fx * fy #W
        psum_SPM = n * m * ox * oy #O
        elements = [weights_SPM, ifmap_SPM, psum_SPM]

        """
        total_banks_weights = math.ceil(weights_SPM * PARAMS.BYTES_PER_DATA / PARAMS.TOTAL_BYTES_IN_BANK)
        total_banks_ifmap = math.ceil(ifmap_SPM * PARAMS.BYTES_PER_DATA / PARAMS.TOTAL_BYTES_IN_BANK)
        total_banks_psum = math.ceil(psum_SPM * PARAMS.BYTES_PER_DATA / PARAMS.TOTAL_BYTES_IN_BANK)

        data_banks_alloc_SPM = sum([total_banks_weights, total_banks_ifmap, total_banks_psum])
        data_alloc_SPM = sum([weights_SPM, ifmap_SPM, psum_SPM]) * PARAMS.BYTES_PER_DATA
        """

    elif type(layer) == GemmLayer:
        x, y, k = prod
        A_SPM = x * k
        B_SPM = k * y
        C_SPM = x * y
        elements = [A_SPM, B_SPM, C_SPM]

    banks = [ math.ceil(element * PARAMS.BYTES_PER_DATA / PARAMS.TOTAL_BYTES_IN_BANK) for element in elements ]
    data_banks_alloc_SPM = sum(banks)
    data_alloc_SPM = sum(elements) * PARAMS.BYTES_PER_DATA

    bank_constraint = data_banks_alloc_SPM <= PARAMS.TOTAL_BANKS_IN_SPM
    size_constraint = data_alloc_SPM <= PARAMS.TOTAL_BANKS_IN_SPM * PARAMS.TOTAL_BYTES_IN_BANK
    utilization_constraint = (data_alloc_SPM / (PARAMS.TOTAL_BANKS_IN_SPM * PARAMS.TOTAL_BYTES_IN_BANK)) >= PARAMS.SPM_UTILIZATION

    return bank_constraint and size_constraint and utilization_constraint


def valid_dram(layer, dram_factor):
    if type(layer) == ConvLayer:
        n, m, c, ox, oy, fx, fy = dram_factor
        no_feature_constraint = fx == 1 and fy == 1 if PARAMS.PRUNE_NO_FEATURE_DRAM else True
        return no_feature_constraint
    elif type(layer) == GemmLayer:
        return True


#def run_thread(layer_index, tc_list, spatial_factor):
def run_thread(params, count_only=False):
    if len(params) == 4:
        layer, tc_list, spatial_factor, env_config = params
    else:
        layer_index, tc_list, spatial_factor, env_config, expr_params = params
        global PARAMS
        PARAMS = expr_params

    if env_config != None:
        PARAMS.env.rf_energy = env_config["rf_energy"]
        PARAMS.env.spm_energy = env_config["spm_energy"]

    if len(params) == 5:
        layer_args = layer_info.resnet_parameters[layer_index]
        layer = ConvLayer(PARAMS.env, **layer_args)

    try:
        stride = layer.strides
    except:
        stride = None

    tc_list_after_spatial = [ int(x/y) for x, y in zip(tc_list, spatial_factor) ]
    tc_list_factors_spatial = [ factors(tc) for tc in tc_list_after_spatial ]

    min_energy = float("inf")
    min_energy_sequence = None
    min_edp = float("inf")
    min_edp_sequence = None
    min_cycle = float("inf")
    min_cycle_sequence = None

    evaluated = 0

    for rf_factor in of_bucket(tc_list_factors_spatial):
        if not valid_rf(layer, stride, rf_factor):
            continue

        tc_list_after_rf = [ int(x/y) for x, y in zip(tc_list_after_spatial, rf_factor)]
        tc_list_factors_rf = [ factors(tc) for tc in tc_list_after_rf]

        for spm_factor in of_bucket(tc_list_factors_rf):
            spatial_rf_factor = ( x*y for x, y in zip(spatial_factor, rf_factor))
            if not valid_spm(layer, stride, spatial_rf_factor, spm_factor):
                continue

            tc_list_after_spm = tuple([ int(x/y) for x, y in zip(tc_list_after_rf, spm_factor) ])
            dram_factor = tc_list_after_spm
            if not valid_dram(layer, dram_factor):
                continue
            #assert tc_list == [ x*y*z*w for x,y,z,w in zip(spatial_factor, rf_factor, spm_factor, dram_factor)]

            evaluated += 1
            if count_only:
                continue

            IV_ORDER = layer._default_loop_order
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

            if cycle < min_cycle:
                min_cycle = cycle
                min_cycle_sequence = (dram_factor, spm_factor, rf_factor, spatial_factor)

    """
    file_name = "_".join([ str(i) for i in spatial_factor ])
    file_name = str(layer_index) + "_" + file_name
    output_file = "out/" + expr_name + "/" + file_name + ".txt"
    with open(output_file, "w") as outFile:
        line = ",".join([ str(item) for item in [min_edp, min_edp_sequence, min_energy, min_energy_sequence] ])
        outFile.write(line)
    """

    if count_only:
        return evaluated

    return min_edp, min_edp_sequence, min_energy, min_energy_sequence, min_cycle, min_cycle_sequence, evaluated


def find_optimal(model, env_config=None):
    if type(model) in [ConvLayer, GemmLayer]:
        layer = model
    else:
        return -1
    base_TCs = layer.base_TCs
    tc_list = [ base_TCs[iv] for iv in model._default_loop_order ]
    tc_list_factors = [ factors(tc) for tc in tc_list ]
    spatial_factors = [ factor for factor in of_bucket(tc_list_factors) if valid_spatial(layer, factor) ]
    params = [ (deepcopy(layer), tc_list, spatial_factor, env_config) for spatial_factor in spatial_factors ]

    pbar = tqdm.tqdm(total=len(params))
    columns = ["edp", "edp_seq", "energy", "energy_seq", "cycle", "cycle_seq", "evaluated"]
    result_df = pd.DataFrame(columns=columns)

    records = []

    multi_processing = True
    if multi_processing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for ret_tuple in executor.map(run_thread, params):
                pbar.update()
                if ret_tuple[1] == None:
                    continue
                records.append(ret_tuple)
                #result_df.loc[len(result_df)] = ret_tuple
    else:
        for param in params:
            ret_tuple = run_thread(param)
            pbar.update()
            if ret_tuple[1] == None:
                continue
            records.append(ret_tuple)
            #result_df.loc[len(result_df)] = ret_tuple

    result_df = pd.DataFrame.from_records(records, columns=columns)
    return result_df


def get_num_evaluations(model, _params, env_config=None):
    if type(model) in [ConvLayer, GemmLayer]:
        layer = model
    else:
        return -1

    if _params == None:
        _params = expr_parameters.ExprParameters()

    global PARAMS
    PARAMS = _params

    base_TCs = layer.base_TCs
    tc_list = [ base_TCs[iv] for iv in model._default_loop_order ]
    tc_list_factors = [ factors(tc) for tc in tc_list ]
    spatial_factors = [ factor for factor in of_bucket(tc_list_factors) if valid_spatial(layer, factor) ]
    params = [ (deepcopy(layer), tc_list, spatial_factor, env_config) for spatial_factor in spatial_factors ]

    total_evaluated = 0
    for param in params:
        evaluated = run_thread(param, count_only=True)
        total_evaluated += evaluated

    return total_evaluated


def optimize(model, params=None, env_config=None):
    if params == None:
        params = expr_parameters.ExprParameters()

    global PARAMS
    PARAMS = params

    result_df = find_optimal(model, env_config)
    result_df.energy = result_df.energy.astype(float)

    min_edp_row = result_df.iloc[result_df["edp"].idxmin()]
    min_edp, min_edp_seq = min_edp_row["edp"], min_edp_row["edp_seq"]

    min_energy_row = result_df.iloc[result_df["energy"].idxmin()]
    min_energy, min_energy_seq = min_energy_row["energy"], min_energy_row["energy_seq"]

    min_cycle_row = result_df.iloc[result_df["cycle"].idxmin()]
    min_cycle, min_cycle_seq = min_cycle_row["cycle"], min_cycle_row["cycle_seq"]

    min_vals = [min_edp, min_energy, min_cycle]
    min_seqs = [min_edp_seq, min_energy_seq, min_cycle_seq]
    types = ["edp", "energy", "cycle"]
    min_orderings = []

    for min_val, min_seq, t in zip(min_vals, min_seqs, types):
        tiling_factors = zip(*min_seq)

        for idx, tiling_factor in zip(model._default_loop_order, tiling_factors):
            model.set_tiling(idx, tiling_factor)

        if t == "edp":
            val, dram_ordering, spm_ordering = model.get_min_edp()
        elif t == "energy":
            val, dram_ordering, spm_ordering = model.get_min_energy()
        elif t == "cycle":
            val, dram_ordering, spm_ordering = model.get_min_cycle()

        assert(min_val == val)
        dram_ordering_parsed = [ ordering.split("_")[0] for ordering in dram_ordering ]
        spm_ordering_parsed = [ ordering.split("_")[0] for ordering in spm_ordering ]
        min_orderings.append((dram_ordering_parsed, spm_ordering_parsed))

    """
    print(result_df)
    print("min_edp:", min_edp)
    print("min_edp_seq:", min_edp_seq)
    print("min_energy:", min_energy)
    print("min_energy_seq:", min_energy_seq)
    """

    min_edp_ordering, min_energy_ordering, min_cycle_ordering = min_orderings

    return {
        "min_edp": min_edp,
        "min_edp_seq": min_edp_seq,
        "min_edp_ordering": min_edp_ordering,
        "min_energy": min_energy,
        "min_energy_seq": min_energy_seq,
        "min_energy_ordering": min_energy_ordering,
        "min_cycle": min_cycle,
        "min_cycle_seq": min_cycle_seq,
        "min_cycle_ordering": min_cycle_ordering
    }

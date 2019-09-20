from matplotlib import pyplot as plt
import nnvm
import tvm
import numpy as np
from mxnet.gluon.model_zoo.vision import get_model

import sys
import argparse
import datetime

from dMazeRunner.dataflow import get_dataflow
import dMazeRunner.common.optimizer as optimizer

from dMazeRunner.common import layer_info, expr_parameters
from dMazeRunner.dataflow import ConvLayer

def main():
    args = parse_arguments()
    layer_arguments = layer_info.resnet_parameters[int(args.layer)]
    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    layer = ConvLayer(env=env, **layer_arguments)
    print("compiling layer", args.layer)

    params = expr_parameters.ExprParameters(env)

    if args.opt_utilization:
        params.PE_UTILIZATION = 0.8
        params.RF_UTILIZATION = 0.8
        params.SPM_UTILIZATION = 0.5

    params.PRUNE_NO_FEATURE_DRAM = True if args.opt_no_feature_dram else False
    params.PRUNE_NO_REDUCTION = True if args.opt_no_spatial_reduction else False

    start_time = datetime.datetime.now()
    result = optimizer.optimize(layer, params)
    end_time = datetime.datetime.now()

    delta = end_time - start_time
    num_evaluations = optimizer.get_num_evaluations(layer, params)
    print("Optimized energy-delay product (EDP): %.4E" %  result["min_edp"])
    print("Execution method for optimized EDP: \nTiling factors (DRAM, SPM, RF, SPATIAL): ", result["min_edp_seq"])
    print("Ordering (DRAM, SPM):", result["min_edp_ordering"])
    print("Minimized energy: %.4E" % result["min_energy"])
    print("Execution method for minimized energy: \nTiling factors (DRAM, SPM, RF, SPATIAL): ", result["min_energy_seq"])
    print("Ordering (DRAM, SPM):", result["min_energy_ordering"])
    print("Minimized execution cycles: %.4E" % result["min_cycle"])
    print("Execution method for minimized execution cycles: \nTiling factors (DRAM, SPM, RF, SPATIAL): ", result["min_cycle_seq"])
    print("Ordering (DRAM, SPM):", result["min_cycle_ordering"])
    print("Execution methods evaluated:", num_evaluations)
    print("Time spent in exploration: {} seconds".format(delta.total_seconds()))

    return

def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--layer-index", dest="layer")
    parser.add_argument("--opt-utilization",
        action="store_true",
        help="Prune execution methods featuring low utilizations of RF, SPM, and PEs.")
    parser.add_argument("--opt-no-feature-dram", action="store_true",
        help="Prune execution methods that access non-contiguous 2D features from DRAM.")
    parser.add_argument("--opt-no-spatial-reduction", action="store_true",
        help="Prune execution methods that require inter-PE communication for reduction.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

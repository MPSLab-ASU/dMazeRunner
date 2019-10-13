from matplotlib import pyplot as plt
#from keras.applications.resnet50 import preprocess_input
import nnvm
import tvm
import numpy as np
from mxnet.gluon.model_zoo.vision import get_model
#from tvm.contrib.download import download_testdata

import sys
import argparse
import datetime
import json

from dMazeRunner.dataflow import get_dataflow, ConvLayer, GemmLayer
from dMazeRunner.common import expr_parameters
import dMazeRunner.common.optimizer as optimizer
from dMazeRunner.common import print_summary

supported_frontends = ["mxnet", "keras"]

def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} exists, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except:
        import urllib
        urllib.urlretrieve(url, path)


def download_block_from_mxnet(args):
    import mxnet as mx
    block = get_model(args.model, pretrained=True)
    sym, params = nnvm.frontend.from_mxnet(block)

    target = 'llvm'
    shape = (1, 3, 224, 224) #input shape; need to be obtained from the model
    shape_dict = {'data': shape}
    model_layers = get_dataflow(sym, target, shape_dict, params, args.batch_size)
    return model_layers


def download_block_from_keras(args):
    import keras
    supported_models = ["resnet50"]

    if args.model == "resnet50":
        weights_url = ''.join(['https://github.com/fchollet/deep-learning-models/releases/',
                            'download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'])
        weights_file = 'resnet50_weights.h5'
        download(weights_url, weights_file)
        keras_resnet50 = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
                                                            input_shape=(224, 224, 3), classes=1000)
        keras_resnet50.load_weights('resnet50_weights.h5')

        from PIL import Image
        from keras.applications.resnet50 import preprocess_input
        img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
        download(img_url, 'cat.png')
        img = Image.open('cat.png').resize((224, 224))
        # input preprocess
        data = np.array(img)[np.newaxis, :].astype('float32')
        data = preprocess_input(data).transpose([0, 3, 1, 2])

        shape_dict = {'input_1': data.shape}
        sym, params = nnvm.frontend.from_keras(keras_resnet50)
        target = 'llvm'

    else:
        print("not supported model; supported models:", supported_models)
        exit()

    model_layers = get_dataflow(sym, target, shape_dict, params, args.batch_size)
    return model_layers


def adjust_optimization_strategies(params):
    retVal = True

    if params.PE_UTILIZATION > 0 and params.RF_UTILIZATION > 0:
        params.PE_UTILIZATION = params.PE_UTILIZATION - 0.1
        params.RF_UTILIZATION = params.RF_UTILIZATION - 0.1

    if params.PE_UTILIZATION <= params.THRESHOLD_BEGIN_REDUCE_SPM_UTIL or params.RF_UTILIZATION <= params.THRESHOLD_BEGIN_REDUCE_SPM_UTIL:
        params.SPM_UTILIZATION = params.SPM_UTILIZATION - 0.1

    if params.PE_UTILIZATION <= params.THRESHOLD_DISABLE_OPTS or params.RF_UTILIZATION <= params.THRESHOLD_DISABLE_OPTS:
        params.PRUNE_NO_FEATURE_DRAM = False
        params.PRUNE_NO_REDUCTION = False
        params.MIN_EXEC_METHODS = 1

    if params.PE_UTILIZATION < 0 or params.RF_UTILIZATION < 0 or params.SPM_UTILIZATION < 0:
        retVal = False

    return retVal


def main():
    args = parse_arguments()

    if args.list_models:
        print_summary.print_supported_models(args)
        exit()

    try:
        download_func = eval("download_block_from_" + args.frontend)
        model_layers = download_func(args)

    except Exception as ex:
        print("Failed to download model", args.model)
        print("Error:", ex)
        exit()

    print("\n"+"="*40+"\n")

    if args.list_layers:
        for i, layer in enumerate(model_layers):
            if type(layer) == dict:
                layer_name = layer["name"]
                layer_type = "Special_Function"
            else:
                layer_name = layer.name
                layer_type = str(type(layer)).split(".")[-1][:-2]
            print("{}: {} ({})".format(i, layer_name, layer_type))
        exit()

    if args.arch_spec:
        with open(args.arch_spec) as jsonFile:
            # Parameters used by analytical model of dataflow execution
            json_data = json.load(jsonFile)
            env_params = json_data["arch_details"]
            env = expr_parameters.Environment(**env_params)
            # Params used by map-space generator and optimizer
            expr_params = json_data["arch_basic"]
            params = expr_parameters.ExprParameters(**expr_params)
    else:
        env = expr_parameters.Environment()
        params = expr_parameters.ExprParameters(env)

    params.PRUNE_NO_FEATURE_DRAM = True if args.opt_no_feature_dram else False
    params.PRUNE_NO_REDUCTION = True if args.opt_no_spatial_reduction else False
    params.MIN_EXEC_METHODS = int(args.min_exec_methods) if int(args.min_exec_methods) > 1 else 1

    if args.auto_optimize:
        params.PE_UTILIZATION = 0.8
        params.RF_UTILIZATION = 0.8
        params.SPM_UTILIZATION = 0.5
        params.PRUNE_NO_FEATURE_DRAM = True
        params.PRUNE_NO_REDUCTION = True
        params.MIN_EXEC_METHODS = 100

    if args.min_resource_utilization:
        threshold_resource_util = args.min_resource_utilization
        params.PE_UTILIZATION = threshold_resource_util[0]
        params.RF_UTILIZATION = threshold_resource_util[1]
        params.SPM_UTILIZATION = threshold_resource_util[2]

    try:
        layers = [model_layers[int(args.layer)]]
        layer = layers[0]
        if type(layer) not in [ConvLayer, GemmLayer]:
            try:
                layer_type = layer.name
            except:
                layer_type = layer["name"]
                print("The requeted layer ({}) is not yet supported for analyzing execution on PE-array of dataflow accelerator.".format(layer_type))
                print("To list all the layers of a model, please use option --list-layers.")
                exit()
    except:
        if (args.layer) and int(args.layer) >= len(model_layers):
            print("Specified model does not feature any layer numbered as specified index.")
            print("To list all the layers of a model, please use option --list-layers.")
            exit()
        elif not (args.layer):
            print("Evaluating entire model for optimized dataflow acceleration.")
            layers = model_layers
        else:
            exit()

    total_energy = 0
    total_cycles = 0
    total_edp = 0
    total_evaluations = 0
    total_time = datetime.timedelta(0)

    for i, layer in enumerate(layers):
        if type(layer) not in [ConvLayer, GemmLayer]:
            continue
        layer.env = env
        layer_type = str(type(layer)).split(".")[-1][:-2]
        print("\n")
        print("compiling layer {}: {} ({})".format(i, layer.name, layer_type))
        print(layer)

        result = None
        skip_layer = False
        start_time = datetime.datetime.now()

        result = optimizer.optimize(layer, params)
        num_evaluations = optimizer.get_num_evaluations(layer, params)

        # Adaptive search for finding efficient execution methods
        while (result == None) or (num_evaluations < params.MIN_EXEC_METHODS):
            if args.auto_optimize:
                adjust_opts_success = adjust_optimization_strategies(params)
                if adjust_opts_success == False:
                    print("Optimizer did not find any execution method to evaluate. Please try with another application model and/or target architecture instead.")
                    skip_layer = True
                    break
                result = optimizer.optimize(layer, params)
                num_evaluations = optimizer.get_num_evaluations(layer, params)
            else:
                print("Optimizer did not find any execution method to evaluate. Please try some different optimization strategy, e.g., apply smaller pruning factors, or try auto-optimizer.")
                skip_layer = True
                break

        if skip_layer == True:
            continue

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

        total_edp += result["min_edp"]
        total_energy += result["min_energy"]
        total_cycles += result["min_cycle"]
        total_evaluations += num_evaluations
        total_time += delta

    if len(layers) > 1:
        print("\n")
        print("="*20, "Summary", "="*20)
        print("Optimized Total EDP: %.4E" %  total_edp)
        print("Optimized Total Energy: %.4E" %  total_energy)
        print("Optimized Total Exeuction Cycles: %.4E" %  total_cycles)
        print("Total execution methods evaluated: {}".format(total_evaluations))
        print("Search space exploration done in %.2f" % (total_time.total_seconds())  + " seconds.")

    return

def parse_arguments():
    parser = argparse.ArgumentParser(description=
        "Search space optimizer for mapping loops on coarse-grained programmable dataflow accelerators.")

    # Command-line arguments about specifying front-end, model, layer.
    parser.add_argument("--frontend", dest="frontend",
        choices=supported_frontends,
        help="Specify a front-end for optimizing dataflow acceleration of ML model")
    parser.add_argument("--model", dest="model",
        help="Specify a ML model to be optimized for dataflow acceleration. \n \
            To list all supported models for a specified front-end, please use --list-models")
    parser.add_argument('--list-models', action='store_true', dest="list_models",
        help="Lists all supported models for a specified front-end.")
    parser.add_argument("--layer-index", dest="layer", default=None,
        help="Specify a layer number for optimizing layer execution on dataflow accelerator. \n \
        To list all layers of a model, please use --list-layers.")
    parser.add_argument("--list-layers", action="store_true",
        help="Lists all layers of a model.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1,
        help="Specify batch size for executing a model or a layer. Default is 1.")

    # Set architecture specifications
    parser.add_argument("--arch-spec",
        help="Path of the file containing architecture specification.")

    # Auto-optimizer. By default OFF. When OFF without any optimization
    # strategy, it leads to a brute-force search of mappings.
    # When ON, applies optimization strategies for search-space-reduction.
    parser.add_argument("--auto-optimize", action="store_true",
        help="Automatically optimize mappings through searching a reduced mapping-space.")

    # Strategies for search-space optimization
    parser.add_argument("--opt-min-utilization",
        action="store", dest="min_resource_utilization", type=float, nargs=3,
        help="Prune execution methods featuring lower resource utilization. \n \
            Threshold for utilizations is specified by 3 arguments (PEs RF SPM). \n \
            For example, arguments \' 0.8 0.7 0.5 \' refer to utilization factors of Processing Elements, RegFiles, and scratchpad memory, respectively. \
            Each utilization factor ranges between 0.0 and 1.0. Factor 0.0 implies no search-reduction based on utilization.")
    parser.add_argument("--opt-no-feature-dram", action="store_true",
        help="Discard execution methods that access non-contiguous 2D data of a tensor from DRAM.")
    parser.add_argument("--opt-no-spatial-reduction", action="store_true",
        help="Discard execution methods that require inter-PE communication to perform a reduction operation for the output.")
    parser.add_argument("--opt-min-exec-methods", action="store", dest="min_exec_methods", type=int, default=1,
        help="Explore at least specified number of efficient execution methods.")


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

import json
import sys
from copy import copy
import math
import nnvm
import tvm
from dMazeRunner.dataflow import get_dataflow, ConvLayer, GemmLayer, DWConvLayer
from dMazeRunner.common.expr_parameters import Environment, ExprParameters

class LayerInfo:
    def __init__(self, layer_spec_dict, layer_type):
        self.name = layer_spec_dict["name"]
        self.layer_type = layer_type
        size_A = layer_spec_dict["size_tensor_A"] #ifmaps
        size_B = layer_spec_dict["size_tensor_B"] #filters

        self.layer_spec_dict = copy(layer_spec_dict)
        self.layer_spec_dict.pop("name", None)

        if layer_type == "CONV":
            strides = layer_spec_dict["strides"]
            m, c, fx, fy = size_B
            n, _c, ix, iy = size_A
            assert(c == _c)
            ox = math.floor((ix - fx) // strides[0]) + 1
            oy = math.floor((iy - fy) // strides[1]) + 1
            #out_size = math.floor((in_size - kernel + 2*pad) // stride) + 1

            self.layer_parameter = {
                "name": self.name,
                # "index": index,
                "channels": c,
                "kernel_size": f"[{fx}, {fy}]",
                "padding": "[0, 0]",
                "strides": f"[{strides[0]}, {strides[1]}]",
                "output_shape": [n, m, ox, oy],
                "input_shape": [n, c, ix, iy],
                "batch_size": n,
                "instances": layer_spec_dict["instances"] if "instances" in layer_spec_dict else 1,
            }

        if layer_type == "GEMM":
            m, k = size_A
            k, n = size_B
            self.layer_parameter = {
                "name": self.name,
                # "index": index,
                "M": m,
                "N": n,
                "K": k,
                "instances": layer_spec_dict["instances"] if "instances" in layer_spec_dict else 1,
            }
        # print(self.layer_parameter["instances"])

        if layer_type == "DW_CONV":
            strides = layer_spec_dict["strides"]
            m, c, fx, fy = size_B
            n, _c, ix, iy = size_A
            # assert(c == _c)
            ox = math.floor((ix - fx) // strides[0]) + 1
            oy = math.floor((iy - fy) // strides[1]) + 1
            #out_size = math.floor((in_size - kernel + 2*pad) // stride) + 1

            self.layer_parameter = {
                "name": self.name,
                # "index": index,
                "channels": _c,
                "kernel_size": f"[{fx}, {fy}]",
                "padding": "[0, 0]",
                "strides": f"[{strides[0]}, {strides[1]}]",
                "output_shape": [n, _c, ox, oy],
                "input_shape": [n, _c, ix, iy],
                "batch_size": n,
                "instances": layer_spec_dict["instances"] if "instances" in layer_spec_dict else 1,
            }


def compare_layer_shapes(layerA_type, layerA_params, layerB_params):
    retVal = True
    params = []
    if layerA_type == GemmLayer:
        params = ["base_TCs"] # ["M", "N", "K"]
    elif layerA_type == ConvLayer:
        params = ["channels", "kernel_size", "padding", "strides", "output_shape", "input_shape", "batch_size"]
    elif layerA_type == DWConvLayer:
        params = ["channels", "kernel_size", "padding", "strides", "output_shape", "input_shape", "batch_size"]
    else:
        return False
        # raise ValueError("Type of the layer being parsed is unsupported.")

    for item in params:
        retVal = retVal and (getattr(layerA_params, item) == getattr(layerB_params, item))
        # (layerA_params[item] == layerB_params[item])
        # print(f"item: {item}, comparison: {(layerA_params[item] == layerB_params[item])}, values: {layerA_params[item]}, {layerB_params[item]}")
    return retVal


def open_file(filepath):
    try:
        file = open(filepath, 'rb')
    except OSError:
        raise Exception("Cannot open or read the file:", filepath)
    return file


def filter_redundant_layers(model_layers):
    all_unique_layers = []
    for layer in model_layers:
        layer_with_same_config = False
        for prev_layer in all_unique_layers:
            if type(layer) != type(prev_layer): continue
            if compare_layer_shapes(type(layer), layer, prev_layer):
                layer_with_same_config = True
                # prev_layer["instances"] += layer["instances"]
                total_instances = getattr(prev_layer, "instances") + getattr(layer, "instances")
                setattr(prev_layer, "instances", total_instances)
                break
        if not layer_with_same_config:
            all_unique_layers.append(layer)
    return all_unique_layers


def get_info_all_layers(model_spec_file):
    assert model_spec_file != None
    with open_file(model_spec_file) as inFile:
        model_spec = json.load(inFile)
    layer_info_list = []
    for layer_type in layer_types:
        if layer_type in model_spec:
            for spec in model_spec[layer_type]:
                layer_info = LayerInfo(spec, layer_type)
                layer_info_list.append(layer_info)
                # same_shaped_layer = False
                # for prev_layer_info in layer_info_list:
                #     if compare_layer_shapes(layer_info.layer_type, prev_layer_info.layer_type, layer_info.layer_parameter, prev_layer_info.layer_parameter):
                #         same_shaped_layer = True
                #         prev_layer_info.layer_parameter["instances"] += layer_info.layer_parameter["instances"]
                #         break
                # if not same_shaped_layer:
                #     layer_info_list.append(layer_info)

    return layer_info_list

layer_types = ["CONV", "GEMM", "DW_CONV"]
# layer_types = ["CONV", "GEMM"]
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


def download_block_from_mxnet(args, model):
    import mxnet as mx
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model(model, pretrained=True)
    sym, params = nnvm.frontend.from_mxnet(block)

    target = 'llvm'
    shape = (1, 3, 224, 224) #input shape; need to be obtained from the model
    shape_dict = {'data': shape}
    model_layers = get_dataflow(sym, target, shape_dict, params, args.batch_size)
    return model_layers


def download_block_from_keras(args, model):
    import keras
    supported_models = ["resnet50"]

    if model == "resnet50":
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


def parse_arguments_model(parser):
    # Command-line arguments about specifying front-end, model, layer.
    parser.add_argument("--frontend", dest="frontend",
        choices=supported_frontends,
        help="Specify a front-end for optimizing dataflow acceleration of ML model")
    parser.add_argument("--models", dest="models", nargs='*',
        help="Specify one or more ML models to be optimized for dataflow acceleration. \n \
            To list all supported models for a specified front-end, please use --list-models. \n \Separate names of the models by space.")
    parser.add_argument('--list-models', action='store_true', dest="list_models",
        help="Lists all supported models for a specified front-end.")
    parser.add_argument("--layer-index", dest="layer", default=None,
        help="Specify a layer number for a model for optimizing layer execution on dataflow accelerator. \n \
        To list all layers of a model, please use --list-layers.")
    parser.add_argument("--list-layers", action="store_true", help="Lists all layers of a model.")
    parser.add_argument("--filter-redundant-layers", action="store_true", default=True, help="Specify filtering redundant layers of a model.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1,
        help="Specify batch size for executing a model or a layer. Default is 1.")
    parser.add_argument("--files-model-spec", nargs='*', help="Path of the files containing information about layers of the DNN model. \n \ Separate multiple paths by space.")
    parser.add_argument("--operations", choices=['GEMM', 'CONV', 'DW_CONV'], default=['GEMM', 'CONV', 'DW_CONV'], nargs='*', help="Define kind of the operation to be performed on sparse tensors. Supported operations are: 'GEMM', 'CONV', 'DW_CONV'.")


def parse_model_from_lib(args):
    from dMazeRunner.common import print_summary

    # Obtain all layers of multiple models
    layers_models = []
    # Object containing layers of a single model
    model_layers = None

    if args.list_models:
        print_summary.print_supported_models(args)
        exit()

    for model in args.models:
        try:
            download_func = eval("download_block_from_" + args.frontend)
            model_layers = download_func(args, model)

        except Exception as ex:
            print("Failed to download model", args.models)
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

        if (args.filter_redundant_layers) and (not (args.layer)):
            unique_layers = filter_redundant_layers(model_layers)
        else:
            unique_layers = model_layers

        layers_models.append(unique_layers)

    return layers_models


def parse_model_from_file(args):
    files_spec_models = args.files_model_spec
    layers_models = []
    for file in files_spec_models:
        info_all_layers = get_info_all_layers(file)
        layers = []

        if args.list_layers:
            for layer_idx, layer_info in enumerate(info_all_layers):
                layer_info = info_all_layers[layer_idx].layer_parameter
                layer_type = info_all_layers[layer_idx].layer_type
                print("{}: {} ({})".format(layer_idx, layer_info["name"], layer_type))
            exit()

        for layer_idx, layer_info in enumerate(info_all_layers):
            layer_info = info_all_layers[layer_idx].layer_parameter
            layer_type = info_all_layers[layer_idx].layer_type
            layer_info["index"] = layer_idx
            if layer_type == "CONV":
                layer = ConvLayer(Environment(), **layer_info)
            elif layer_type == "GEMM":
                layer = GemmLayer(Environment(), **layer_info)
            elif layer_type == "DW_CONV":
                layer = DWConvLayer(Environment(), **layer_info)

            layers.append(layer)

        unique_layers = filter_redundant_layers(layers)
        layers_models.append(unique_layers)

    return layers_models


def get_specified_layers(args, layers_models):
    # TODO: Do only if args.layer is specified else it is all layers.
    needed_layers = layers_models
    if not (args.layer):
        needed_layers = layers_models
    else:
        model_layers = layers_models[0]
        try:
            layer = model_layers[int(args.layer)]
            needed_layers = [[layer]]
            if type(layer) not in [ConvLayer, GemmLayer, DWConvLayer]:
                try:
                    layer_name = layer.name
                except:
                    layer_name = layer["name"]
                print("The requested layer ({}) is not yet supported for analyzing execution on PE-array of dataflow accelerator.".format(layer_name))
                print("To list all the layers of a model, please use option --list-layers.")
                exit()
        except:
            if (args.layer) and (int(args.layer) >= len(model_layers)):
                print("Specified model does not feature any layer numbered as specified index.")
                print("To list all the layers of a model, please use option --list-layers.")
                exit()
            # elif not (args.layer):
            #     print("Evaluating entire models for optimized dataflow acceleration.")
            #     needed_layers = layers_models
            else:
                exit()

    return needed_layers


def get_supported_layers(layers_models, supported_operations=["CONV", "GEMM", "DW_CONV"]):
    supported_layers_models = []
    for model_layers in layers_models:
        model_supported_layers = []
        for layer in model_layers:
            if "CONV" in supported_operations and type(layer) == ConvLayer:
                model_supported_layers.append(layer)
            elif "GEMM" in supported_operations and type(layer) == GemmLayer:
                model_supported_layers.append(layer)
            elif "DW_CONV" in supported_operations and type(layer) == DWConvLayer:
                model_supported_layers.append(layer)
        supported_layers_models.append(model_supported_layers)

    return supported_layers_models


def parse_layers(args):
    if args.files_model_spec != None:
        all_layers = parse_model_from_file(args)
    else:
        all_layers = parse_model_from_lib(args)

    needed_layers = get_specified_layers(args, all_layers)
    supported_layers = get_supported_layers(needed_layers, args.operations)

    return supported_layers

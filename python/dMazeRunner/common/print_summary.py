from matplotlib import pyplot as plt
#from keras.applications.resnet50 import preprocess_input
import nnvm
import tvm
import numpy as np
from mxnet.gluon.model_zoo.vision import get_model

import sys
import argparse

from dMazeRunner.dataflow import get_dataflow
import dMazeRunner.common.optimizer as optimizer

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
    return block

def download_block_from_keras(args):
    import keras
    supported_models = ["resnet50"]

    if args.model == "resnet50":
        block = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
                                                            input_shape=(224, 224, 3), classes=1000)
    else:
        print("not supported model; supported models:", supported_models)
        exit()

    return block

def print_supported_models(args):
    temp = args.model
    if args.frontend == "mxnet":
        try:
            args.model = "xyz"
            download_func = eval("download_block_from_" + args.frontend)
            block = download_func(args)
        except ValueError as errormsg:
            args.model = temp
            print(errormsg)
    elif args.frontend == "keras":
        supported_models = ["resnet50"]
        # supported_models = [
        #     "Xception",
        #     "VGG16",
        #     "VGG19",
        #     "ResNet",
        #     "ResNetV2",
        #     "ResNeXt",
        #     "InceptionV3",
        #     "InceptionResNetV2",
        #     "MobileNet",
        #     "MobileNetV2",
        #     "DenseNet",
        #     "NASNet"]
        print(supported_models)

def print_summary(block, frontend):
    if frontend == "mxnet":
        sym, params = nnvm.frontend.from_mxnet(block)
        if (sym):
            print(sym.debug_str())
        else:
            print(block)
    elif frontend == "keras":
        sym, params = nnvm.frontend.from_keras(block)
        if (sym):
            print(sym.debug_str())
        else:
            block.summary()

def main():
    args = parse_arguments()

    if(args.list_models):
        print_supported_models(args)

    if(args.summarize_model):
        try:
            download_func = eval("download_block_from_" + args.frontend)
            block = download_func(args)
        except:
            print("Failed to download model", args.model)
            exit()

        print_summary(block, args.frontend)


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--frontend", dest="frontend",
        choices=supported_frontends)
    parser.add_argument("--model", dest="model")
    parser.add_argument('--list-models', action='store_true', dest="list_models")
    parser.add_argument('--summarize-model', action='store_true', dest="summarize_model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

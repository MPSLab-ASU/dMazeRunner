import tvm
import argparse
import json
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters


def get_sample_layer(env):
    args = {
        #  ResNet Conv5_2 layer
        "name": "test_conv2d12_from_resnet18_v1",
        "channels": 512,
        "kernel_size": "[3, 3]",
        "padding": "[1, 1]",
        "strides": "[1, 1]",
        "output_shape": [1, 512, 7, 7],
        "input_shape": [1, 512, 7, 7],
        "batch_size": 4,
    }
    layer = ConvLayer(env, **args)
    layer.set_tiling("N", [1, 1, 4, 1])
    layer.set_tiling("M", [4, 4, 16, 2])
    layer.set_tiling("C", [32, 1, 8, 2])
    layer.set_tiling("Ox", [1, 1, 1, 7])
    layer.set_tiling("Oy", [1, 1, 1, 7])
    layer.set_tiling("Fx", [1, 3, 1, 1])
    layer.set_tiling("Fy", [1, 3, 1, 1])

    return layer

def get_energy(layer):
    #
    # Energy estimation of the execution method with specified tiling factor.
    # For minimizing energy, the best ordering is chosen implicitly during analysis.
    # To list energy consumptions of execution methods with different loop-orderings,
    # please use layer.get_Energy_One_Layer() instead.
    #
    energy, _, _ = layer.get_min_energy()
    print("Estimated Min Energy (pJ): %.4E" % energy)


def get_performance(layer):
    #
    # Execution cycle estimation of the execution method with specified tiling factor.
    # For minimizing cycles, the best ordering is chosen implicitly during analysis.
    #
    # To list performance of execution methods with different loop-orderings,
    # please use layer.get_Cycles_One_Layer() instead.
    #
    cycles, _, _ = layer.get_min_cycle()
    print("Estimated Min Execution Cycles: %.4E" % cycles)


def get_EDP(layer):
    #
    # Energy-Delay Product (EDP) estimation of the execution method with
    # specified tiling factor. For minimizing EDP, the best ordering is
    # chosen implicitly during analysis.
    #
    # Since min_EDP may come from different loop-ordering, as compared to
    # the best ordering for min_energy, or min_cycles,
    # min_edp may not equals to min_energy * min_cycles.
    #
    # To list EDP of execution methods with different loop-orderings,
    # please use layer.get_EDP_One_Layer() instead.
    #
    edp, _, _ = layer.get_min_edp()
    print("Estimated Min Energy-Delay Product: %.4E" % edp)


def main():
    parser = argparse.ArgumentParser(description=
        "Analyzing execution method for mapping loops on coarse-grained programmable dataflow accelerators.")

    # Set architecture specifications
    parser.add_argument("--arch-spec",
        help="Path of the file containing architecture specification.")

    args = parser.parse_args()
    if args.arch_spec:
        with open(args.arch_spec) as jsonFile:
            # Parameters used by analytical model of dataflow execution
            json_data = json.load(jsonFile)
            env_params = json_data["arch_details"]
            env = expr_parameters.Environment(**env_params)
    else:
        env = expr_parameters.Environment()
        params = expr_parameters.ExprParameters(env)

    layer = get_sample_layer(env)
    get_EDP(layer)
    get_energy(layer)
    get_performance(layer)

if __name__ == "__main__":
    main()

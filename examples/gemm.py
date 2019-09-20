import tvm
import argparse
import json
from dMazeRunner.dataflow import GemmLayer
from dMazeRunner.common import expr_parameters

def get_sample_layer(env):
    params = { # C(M,N) = A(M, K) . B(K, N)
        "name": "gemm_example",
        "M": 1000,
        "K": 1000,
        "N": 1000,
    }
    # Tiling factors (DRAM, SPM, RF, SPATIAL):  ((250, 2, 20), (2, 1, 1), (1, 4, 50), (2, 125, 1))
    # Ordering (DRAM, SPM): (['Y', 'K', 'X'], ['Y', 'X', 'K'])

    layer = GemmLayer(env, **params)
    layer.set_tiling("X", [250, 2,  1, 2])
    layer.set_tiling("Y", [2,   1,  4, 125])
    layer.set_tiling("K", [20,  1, 50, 1])

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

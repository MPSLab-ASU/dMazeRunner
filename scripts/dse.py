import argparse
import concurrent
import tqdm
import pickle
import datetime
import openpyxl
import pandas as pd

from dMazeRunner.common import layer_info, optimizer, expr_parameters
from dMazeRunner.dataflow import ConvLayer
import dse_parameters

MIN_EVALUATIONS = 10
MAX_EVALUATIONS = 500

def find_opt_parameters(params):
    layer, n_PE, spm_size, rf_size, spm_energy, rf_energy = params
    TOTAL_BYTES_IN_BANK = 2048
    spm_banks = spm_size * 1024 / 2 / TOTAL_BYTES_IN_BANK
    effective_rf_size = rf_size/2

    global_config = expr_parameters.ExprParameters()

    global_config.CGRA_SIZE = n_PE
    global_config.RF_SIZE = rf_size
    global_config.BYTES_PER_DATA = 2
    global_config.TOTAL_BANKS_IN_SPM = spm_banks
    global_config.TOTAL_BYTES_IN_BANK = TOTAL_BYTES_IN_BANK
    global_config.PE_UTILIZATION = 0.99
    global_config.RF_UTILIZATION = 0.99
    global_config.SPM_UTILIZATION = 0.99
    global_config.PRUNE_NO_FEATURE_DRAM = True
    global_config.PRUNE_NO_REDUCTION = True

    env_config = {
        "rf_energy": rf_energy,
        "spm_energy": spm_energy,
    }

    def update_global_config(prev_global_config, increase):
        global_config = prev_global_config
        STEP = -0.03

        min_utilization = min([global_config.PE_UTILIZATION, global_config.RF_UTILIZATION, global_config.SPM_UTILIZATION])
        if abs(min_utilization) < abs(STEP):
            if global_config.PRUNE_NO_FEATURE_DRAM:
                global_config.PRUNE_NO_FEATURE_DRAM = False
            elif global_config.PRUNE_NO_REDUCTION:
                global_config.PRUNE_NO_REDUCTION = False
            else:
                global_config = None #No configuration available

        else:
            if increase:
                STEP *= -1
            global_config.PE_UTILIZATION += STEP
            global_config.RF_UTILIZATION += STEP
            global_config.SPM_UTILIZATION += STEP

        return global_config

    while global_config != None:
        num_evaluations = optimizer.get_num_evaluations(layer, global_config, env_config)
        if num_evaluations < MIN_EVALUATIONS:
            global_config = update_global_config(global_config, increase=False)
            continue
        break

    """
    if num_evaluations < MIN_EVALUATIONS:
        increase = False
    if num_evaluations > MAX_EVALUATIONS:
        increase = True

    while global_config != None:
        global_config = update_global_config(global_config, increase=increase)

    while global_config != None:
        num_evaluations = optimizer.get_num_evaluations(layer, global_config, env_config)
        #print(num_evaluations)

        if num_evaluations < MIN_EVALUATIONS:
            if not prev_increase:
                break
            global_config = update_global_config(global_config, increase=False)
            prev_increase = False
            continue

        if num_evaluations > MAX_EVALUATIONS:
            if prev_increase:
                break
            global_config = update_global_config(global_config, increase=True)
            prev_increase = True
            continue

        break # MIN_EVALUATIONS <= num_evaluations <= MAX_EVALUATIONS
    """

    key = (n_PE, spm_size, rf_size)
    return key, global_config, env_config, num_evaluations


def pass_config(n_PE, rf_size, spm_size):
    effective_rf_size = rf_size/2
    cond1 = (effective_rf_size * n_PE) > (65536*4)
    cond2 = (spm_size*1024) < ((rf_size * n_PE)/16)
    cond3 = (spm_size*1024) > (16 * rf_size * n_PE)
    return not cond1 and not cond2 and not cond3


def find_config(args):
    config_file_name = args.config_file_name

    n_PEs = dse_parameters.n_PEs
    energy_rf = dse_parameters.energy_rf
    energy_spm = dse_parameters.energy_spm

    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    layer_args = layer_info.resnet_parameters[args.layer_index]
    layer = ConvLayer(env, **layer_args)

    opt_parameters = {}

    """
    energy_rf = {
        256: 0.48,
    }

    energy_spm = {
        4096: 99.0984375,
    }

    n_PEs = [64, 128, 256]
    """

    #pbar = tqdm.tqdm(total=len(n_PEs)*len(energy_spm)*len(energy_rf))

    params = []
    for n_PE in n_PEs:
        for spm_size, spm_energy in energy_spm.items():
            for rf_size, rf_energy in energy_rf.items():
                if pass_config(n_PE, rf_size, spm_size):
                    continue
                param = (layer, n_PE, spm_size, rf_size, spm_energy, rf_energy)
                params.append(param)

    pbar = tqdm.tqdm(total=len(params))

    multiprocessing = True
    if multiprocessing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for ret_tuple in executor.map(find_opt_parameters, params):
                key, global_config, env_config, num_evaluations = ret_tuple
                pbar.update()
                opt_parameters[key] = (global_config, env_config, num_evaluations)
    else:
        for param in params:
            key, global_config, env_config, num_evaluations = find_opt_parameters(param)
    """
    key = (n_PE, spm_size, rf_size)
    opt_parameters[key], num_evaluations = find_opt_parameters(layer, n_PE, spm_size, rf_size, spm_energy, rf_energy)
    print(key, num_evaluations)
    """

    print(opt_parameters)
    pickle.dump(opt_parameters, open(config_file_name+".p", "wb"))


def run_dse(args):
    config_file_name = args.config_file_name + ".p"
    global_configs = pickle.load(open(config_file_name, "rb"))

    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    layer_args = layer_info.resnet_parameters[args.layer_index]
    layer = ConvLayer(env, **layer_args)

    columns = ("spm_size", "rf_size", "n_PE", "edp", "total_evaluated",
        "start_time", "end_time")
    final_df = pd.DataFrame(columns=columns)
    records = []

    print("evaluating {} design points".format(len(global_configs)))
    i = 1
    for key, (global_config, env_config, num_evaluations) in global_configs.items():
        n_PE, spm_size, rf_size = key
        if global_config == None:
            min_edp = -1
            start_time, end_time = None, None
        else:
            start_time = datetime.datetime.now()
            result = optimizer.optimize(layer, global_config, env_config)
            min_edp = result["min_edp"]
            end_time = datetime.datetime.now()

        records.append((spm_size, rf_size, n_PE, min_edp, num_evaluations, start_time, end_time))
        """
        final_df.loc[len(final_df)] = [spm_size, rf_size, n_PE, min_edp,
            num_evaluations, start_time, end_time]
        """

        print("{}/{}, evaluated: {}\n".format(i, len(global_configs), num_evaluations))
        i += 1

    final_df = pd.DataFrame.from_records(records, columns=columns)
    final_df = final_df.astype({
        "spm_size": int,
        "rf_size": int,
        "n_PE": int,
        "total_evaluated": int,
    })
    pickle.dump(final_df, open("dse_output"+config_file_name+".p", "wb"))

    file_name = "dse_output_layer_{}.xlsx".format(config_file_name)
    final_df.to_excel(file_name)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("option", choices=["find_config", "run_dse"])
    parser.add_argument("--layer-index", dest="layer_index", type=int, required=True)
    parser.add_argument("--config-file", dest="config_file_name", required=True,
        help="if option=find_config, the result is saved as CONFIG_FILE_NAME. When option=run_dse, the script reads CONFIG_FILE_NAME.")
    args = parser.parse_args()

    if args.option == "find_config":
        find_config(args)
    elif args.option == "run_dse":
        run_dse(args)
    else:
        print("invalid option")


if __name__ == "__main__":
    main()

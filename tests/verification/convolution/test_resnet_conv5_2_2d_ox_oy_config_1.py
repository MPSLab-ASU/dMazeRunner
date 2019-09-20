import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters


spm_orderings = ()
dram_orderings = ()

def setup():
    layer = get_sample_layer()
    dicts = [
        {'O':9, 'I':1, 'W':1},
        {'O':3, 'I':1, 'W':1},
        {'O':1, 'I':4, 'W':1},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts ]
    global spm_orderings
    spm_orderings = tuple(orders)
    assert(len(spm_orderings) == 3)

    dicts = [
        {'O':32, 'I':1, 'W':1},
        {'O':1, 'I':4, 'W':1},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "DRAM") for d in dicts ]
    global dram_orderings
    dram_orderings = tuple(orders)
    assert(len(dram_orderings) == 2)

def get_sample_layer():
    env = expr_parameters.Environment()
    args = {
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

def test_commonFunctions():
    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()
    assert data_RF["I"] == 32
    assert data_RF["W"] == 128
    assert data_RF["O"] == 64

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["I"] == 3136
    assert data_PE_array["W"] == 512
    assert data_PE_array["O"] == 6272

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["I"] == 5184
    assert data_SPM["W"] == 18432
    assert data_SPM["O"] == 25088



def test_TotalEnergy():
    layer = get_sample_layer()
    energy_ops = layer.computeEnergyCost_UsefulOps()
    assert abs(energy_ops - 34681651.2) < 1.0e-5

    energy_RF = layer.computeEnergyCost_RF()
    assert abs(energy_RF - 1775700541.44) < 1.0e-5

    key_1_spm, key_2_spm, key_3_spm = spm_orderings
    key_1_dram, key_2_dram = dram_orderings

    energy_SPM_pass = layer.get_Energy_One_SPM_Pass()
    assert abs(energy_SPM_pass[(key_1_dram, key_1_spm)] - 4932887.2) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_1_spm)] - 4932887.2) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_2_spm)] - 6736965.12) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_2_spm)] - 6736965.12) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_3_spm)] - 10667439.36) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_3_spm)] - 10667439.36) < 1

    total_SPM_pass = layer.get_TripCounts_all_IVs("DRAM")
    assert abs(total_SPM_pass - 128) == 0

    energy_DRAM = layer.get_Energy_DataCommunication_DRAM()
    assert abs(energy_DRAM[(key_1_dram, key_1_spm)] - 624640000) < 1
    assert abs(energy_DRAM[(key_2_dram, key_1_spm)] - 1769472000) < 1
    assert abs(energy_DRAM[(key_1_dram, key_2_spm)] - 624640000) < 1
    assert abs(energy_DRAM[(key_2_dram, key_2_spm)] - 1769472000) < 1
    assert abs(energy_DRAM[(key_1_dram, key_3_spm)] - 624640000) < 1
    assert abs(energy_DRAM[(key_2_dram, key_3_spm)] - 1769472000) < 1

    energy = layer.get_Energy_One_Layer()
    for d in dram_orderings:
        for s in spm_orderings:
            expected_energy = round(energy_ops + energy_RF + (energy_SPM_pass[(d,s)] * total_SPM_pass) + energy_DRAM[(d,s)])
            assert abs(energy[(d,s)] - expected_energy) == 0


def test_Performance():
    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 512

    cycles = layer.calculate_Cycles_DataDistribution_NOC()

    assert cycles[layer._O.name] == 6272
    assert cycles[layer._I.name] == 3136
    assert cycles[layer._W.name] == 512

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 6272

    key_1_spm, key_2_spm, key_3_spm = spm_orderings
    key_1_dram, key_2_dram = dram_orderings

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][0][1] == 3136
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][0][1] == 3136
    assert comm_cycles_dict[(key_1_dram, key_2_spm)][0][1] == 3136
    assert comm_cycles_dict[(key_2_dram, key_2_spm)][0][1] == 3136
    assert comm_cycles_dict[(key_1_dram, key_3_spm)][0][1] == 6272
    assert comm_cycles_dict[(key_2_dram, key_3_spm)][0][1] == 6272

    # Check stall cycles for each RF pass
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_1_dram, key_2_spm)][1][1] == 0
    assert comm_cycles_dict[(key_2_dram, key_2_spm)][1][1] == 0
    assert comm_cycles_dict[(key_1_dram, key_3_spm)][1][1] == 6784
    assert comm_cycles_dict[(key_2_dram, key_3_spm)][1][1] == 6784

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    assert dict_cycles_spm_pass[(key_1_dram, key_1_spm)] == 152576
    assert dict_cycles_spm_pass[(key_2_dram, key_1_spm)] == 152576
    assert dict_cycles_spm_pass[(key_1_dram, key_2_spm)] == 231936
    assert dict_cycles_spm_pass[(key_2_dram, key_2_spm)] == 231936
    assert dict_cycles_spm_pass[(key_1_dram, key_3_spm)] == 470016
    assert dict_cycles_spm_pass[(key_2_dram, key_3_spm)] == 470016

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["I"]["dma_invocations"] == 4
    assert dma_accesses["I"]["data_accessed_dma"] == 1296
    assert dma_accesses["W"]["dma_invocations"] == 128
    assert dma_accesses["W"]["data_accessed_dma"] == 144
    assert dma_accesses["O"]["dma_invocations"] == 4
    assert dma_accesses["O"]["data_accessed_dma"] == 6272

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 572
    assert dma_cycles["W"] == 7296
    assert dma_cycles["O"] == 2064

    dict_cycles = layer.get_Cycles_One_Layer()
    assert dict_cycles[(key_1_dram, key_1_spm)] == 19529728
    assert dict_cycles[(key_2_dram, key_1_spm)] == 19529728
    assert dict_cycles[(key_1_dram, key_2_spm)] == 29687808
    assert dict_cycles[(key_2_dram, key_2_spm)] == 29687808
    assert dict_cycles[(key_1_dram, key_3_spm)] == 60162048
    assert dict_cycles[(key_2_dram, key_3_spm)] == 60162048

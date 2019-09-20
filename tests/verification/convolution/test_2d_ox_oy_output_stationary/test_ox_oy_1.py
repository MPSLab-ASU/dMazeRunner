import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters

spm_orderings = ()
dram_orderings = ()


def setup():
    layer = get_sample_layer()
    dicts = [
        {"O": 16, "I": 1, "W": 1},
    ]
    orders = [layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts]
    global spm_orderings
    spm_orderings = tuple(orders)
    assert(len(spm_orderings) == 1)

    dicts = [
        {'O': 8, 'I': 1, 'W': 1},
        {'O': 1, 'I': 16, 'W': 1},
        {'O': 1, 'I': 1, 'W': 4},
    ]
    orders = [layer.get_ordering_from_reuse_factor(d, "DRAM") for d in dicts]
    global dram_orderings
    dram_orderings = tuple(orders)
    assert(len(dram_orderings) == 3)

def get_sample_layer():
    env = expr_parameters.Environment()
    args = {
        "name": "test_conv2d12_from_resnet18_v1",
        "channels": 256,
        "kernel_size": "[3, 3]",
        "padding": "[1, 1]",
        "strides": "[1, 1]",
        "output_shape": [1, 256, 14, 14],
        "input_shape": [1, 256, 14, 14],
        "batch_size": 4,
    }
    layer = ConvLayer(env, batch=4, **args)
    layer.set_tiling("N", [4, 1, 1, 1])
    layer.set_tiling("M", [16, 1, 8, 2])
    layer.set_tiling("C", [8, 16, 1, 2])
    layer.set_tiling("Ox", [1, 1, 2, 7])
    layer.set_tiling("Oy", [1, 1, 2, 7])
    layer.set_tiling("Fx", [1, 1, 3, 1])
    layer.set_tiling("Fy", [1, 1, 3, 1])
    return layer


def test_commonFunctions():
    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()

    assert data_RF["I"] == 16
    assert data_RF["W"] == 72
    assert data_RF["O"] == 32

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["I"] == 512
    assert data_PE_array["W"] == 288
    assert data_PE_array["O"] == 3136

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["I"] == 8192
    assert data_SPM["W"] == 4608
    assert data_SPM["O"] == 3136


def test_TotalEnergy():
    layer = get_sample_layer()

    energy_ops = layer.computeEnergyCost_UsefulOps()
    assert abs(energy_ops - 34681651.2) < 1.0e-5

    energy_RF = layer.computeEnergyCost_RF()
    assert abs(energy_RF - 1775700541.44) < 1.0e-5

    key_1_spm = spm_orderings[0]
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    energy_SPM_pass = layer.get_Energy_One_SPM_Pass()
    assert abs(energy_SPM_pass[(key_1_dram, key_1_spm)] - 724736) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_1_spm)] - 837490.88) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_1_spm)] - 837490.88) < 1

    total_SPM_pass = layer.get_TripCounts_all_IVs("DRAM")
    assert abs(total_SPM_pass - 512) == 0

    energy_DRAM = layer.get_Energy_DataCommunication_DRAM()
    assert abs(energy_DRAM[(key_1_dram, key_1_spm)] - 1354966200.32) < 1
    assert abs(energy_DRAM[(key_2_dram, key_1_spm)] - 1126400000) < 1
    assert abs(energy_DRAM[(key_3_dram, key_1_spm)] - 1558937600) < 1

    energy = layer.get_Energy_One_Layer()
    for d in dram_orderings:
        for s in spm_orderings:
            expected_energy = round(energy_ops + energy_RF + (energy_SPM_pass[(d,s)] * total_SPM_pass) + energy_DRAM[(d,s)])
            assert abs(energy[(d,s)] - expected_energy) == 0


def test_Performance():
    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 288

    cycles = layer.calculate_Cycles_DataDistribution_NOC()

    assert cycles[layer._O.name] == 3136
    assert cycles[layer._I.name] == 512
    assert cycles[layer._W.name] == 288

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 3136

    key_1_spm = spm_orderings[0]
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][0][1] == 512
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][0][1] == 512
    assert comm_cycles_dict[(key_3_dram, key_1_spm)][0][1] == 512

    # Check stall cycles for each RF pass
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_3_dram, key_1_spm)][1][1] == 0

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    assert dict_cycles_spm_pass[(key_1_dram, key_1_spm)] == 8192
    assert dict_cycles_spm_pass[(key_2_dram, key_1_spm)] == 14240
    assert dict_cycles_spm_pass[(key_3_dram, key_1_spm)] == 14240

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["I"]["dma_invocations"] == 1
    assert dma_accesses["I"]["data_accessed_dma"] == 8192
    assert dma_accesses["W"]["dma_invocations"] == 16
    assert dma_accesses["W"]["data_accessed_dma"] == 288
    assert dma_accesses["O"]["dma_invocations"] == 1
    assert dma_accesses["O"]["data_accessed_dma"] == 3136

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 660
    assert dma_cycles["W"] == 1088
    assert dma_cycles["O"] == 281

    dict_cycles = layer.get_Cycles_One_Layer()
    assert dict_cycles[(key_1_dram, key_1_spm)] == 4581376
    assert dict_cycles[(key_2_dram, key_1_spm)] == 7290880
    assert dict_cycles[(key_3_dram, key_1_spm)] == 7290880

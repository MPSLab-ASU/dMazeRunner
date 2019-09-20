import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters

spm_orderings = ()
dram_orderings = ()

def setup():
    layer = get_sample_layer()
    dicts = [
        {"O": 12, "I": 1, "W": 1},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts ]
    global spm_orderings
    spm_orderings = tuple(orders)
    assert(len(spm_orderings) == 1)

    dicts = [
        {"O": 4, "I": 1, "W": 1},
        {"O": 1, "I": 4, "W": 1},
        {"O": 1, "I": 1, "W": 2},
        {"O": 1, "I": 1, "W": 5},
        {"O": 1, "I": 1, "W": 10},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "DRAM") for d in dicts ]
    global dram_orderings
    dram_orderings = tuple(orders)
    assert(len(dram_orderings) == 5)


def get_sample_layer():
    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    args = {
        "name": "test_conv2d12_from_resnet18_v1",
        "channels": 32,
        "kernel_size": "[3, 3]",
        "padding": "[1, 1]",
        "strides": "[1, 1]",
        "output_shape": [1, 32, 10, 10],
        "input_shape": [1, 32, 10, 10],
        "batch_size": 8,
    }
    layer = ConvLayer(env, **args)
    layer.set_tiling("N", [2, 1, 2, 2])
    layer.set_tiling("M", [4, 1, 2, 4])
    layer.set_tiling("C", [4, 4, 1, 2])
    layer.set_tiling("Ox", [5, 1, 2, 1])
    layer.set_tiling("Oy", [1, 1, 5, 2])
    layer.set_tiling("Fx", [1, 3, 1, 1])
    layer.set_tiling("Fy", [1, 1, 3, 1])

    for level in layer._loop_IVs:
        layer.set_ordering(level, ["N", "C", "Fx", "Fy", "M", "Ox", "Oy"])

    return layer


def test_commonFunctions():

    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()
    assert data_RF["I"] == 28
    assert data_RF["W"] == 6
    assert data_RF["O"] == 40

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["I"] == 192
    assert data_PE_array["W"] == 48
    assert data_PE_array["O"] == 640

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["I"] == 1536
    assert data_SPM["W"] == 576
    assert data_SPM["O"] == 640


def test_TotalEnergy():
    layer = get_sample_layer()

    energy_ops = layer.computeEnergyCost_UsefulOps()
    assert abs(energy_ops - 552960) < 1.0e-5

    energy_RF = layer.computeEnergyCost_RF()
    assert abs(energy_RF - 28311552) < 1.0e-5

    key_1_spm = spm_orderings[0]
    key_1_dram, key_2_dram, key_3_dram, key_4_dram, key_5_dram = dram_orderings

    energy_SPM_pass = layer.get_Energy_One_SPM_Pass()
    assert abs(energy_SPM_pass[(key_1_dram, key_1_spm)] - 64992) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_1_spm)] - 88003.2) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_1_spm)] - 88003.2) < 1
    assert abs(energy_SPM_pass[(key_4_dram, key_1_spm)] - 88003.2) < 1
    assert abs(energy_SPM_pass[(key_5_dram, key_1_spm)] - 88003.2) < 1

    total_SPM_pass = layer.get_TripCounts_all_IVs("DRAM")
    assert abs(total_SPM_pass - 160) == 0

    energy_DRAM = layer.get_Energy_DataCommunication_DRAM()
    assert abs(energy_DRAM[(key_1_dram, key_1_spm)] - 73227648) < 1
    assert abs(energy_DRAM[(key_2_dram, key_1_spm)] - 66560000) < 1
    assert abs(energy_DRAM[(key_3_dram, key_1_spm)] - 94208000) < 1
    assert abs(energy_DRAM[(key_4_dram, key_1_spm)] - 88678400) < 1
    assert abs(energy_DRAM[(key_5_dram, key_1_spm)] - 86835200) < 1

    energy = layer.get_Energy_One_Layer()
    for d in dram_orderings:
        for s in spm_orderings:
            expected_energy = round(energy_ops + energy_RF + (energy_SPM_pass[(d,s)] * total_SPM_pass) + energy_DRAM[(d,s)])
            assert abs(energy[(d,s)] - expected_energy) == 0


def test_Performance():
    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 120

    cycles = layer.calculate_Cycles_DataDistribution_NOC()

    assert cycles[layer._I.name] == 192
    assert cycles[layer._W.name] == 48
    assert cycles[layer._O.name] == 640

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 640

    key_1_spm = spm_orderings[0]
    key_1_dram, key_2_dram, key_3_dram, key_4_dram, key_5_dram = dram_orderings

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][0][1] == 192
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][0][1] == 192
    assert comm_cycles_dict[(key_3_dram, key_1_spm)][0][1] == 192
    assert comm_cycles_dict[(key_4_dram, key_1_spm)][0][1] == 192
    assert comm_cycles_dict[(key_5_dram, key_1_spm)][0][1] == 192

    # Check stall cycles for each RF pass
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_3_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_4_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_5_dram, key_1_spm)][1][1] == 0

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    assert dict_cycles_spm_pass[(key_1_dram, key_1_spm)] == 2304
    assert dict_cycles_spm_pass[(key_2_dram, key_1_spm)] == 3512
    assert dict_cycles_spm_pass[(key_3_dram, key_1_spm)] == 3512
    assert dict_cycles_spm_pass[(key_4_dram, key_1_spm)] == 3512
    assert dict_cycles_spm_pass[(key_5_dram, key_1_spm)] == 3512

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["I"]["dma_invocations"] == 32
    assert dma_accesses["I"]["data_accessed_dma"] == 48
    assert dma_accesses["W"]["dma_invocations"] == 8
    assert dma_accesses["W"]["data_accessed_dma"] == 72
    assert dma_accesses["O"]["dma_invocations"] == 32
    assert dma_accesses["O"]["data_accessed_dma"] == 20

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 1600
    assert dma_cycles["W"] == 408
    assert dma_cycles["O"] == 1504

    dict_cycles = layer.get_Cycles_One_Layer()
    assert dict_cycles[(key_1_dram, key_1_spm)] == 465280
    assert dict_cycles[(key_2_dram, key_1_spm)] == 622080
    assert dict_cycles[(key_3_dram, key_1_spm)] == 769920
    assert dict_cycles[(key_4_dram, key_1_spm)] == 750336
    assert dict_cycles[(key_5_dram, key_1_spm)] == 743808

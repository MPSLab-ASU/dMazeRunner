import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters

spm_orderings = ()
dram_orderings = ()

def setup():
    layer = get_sample_layer()
    dicts = [
        {"O": 2, "I": 1, "W": 1},
        {"O": 1, "I": 2, "W": 1},
        {"O": 1, "I": 1, "W": 2},
        {"O": 1, "I": 1, "W": 5},
        {"O": 1, "I": 1, "W": 10},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts ]
    global spm_orderings
    spm_orderings = tuple(orders)
    assert(len(spm_orderings) == 5)

    dicts = [
        {"O": 1, "I": 4, "W": 1},
        {"O": 1, "I": 1, "W": 2},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "DRAM") for d in dicts ]
    global dram_orderings
    dram_orderings = tuple(orders)
    assert(len(dram_orderings) == 2)


def get_sample_layer():
    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    args = {
        "name": "test_conv2d12_from_resnet18_v1",
        "channels": 16,
        "kernel_size": "[3, 3]",
        "padding": "[0, 0]",
        "strides": "[2, 2]",
        "output_shape": [1, 16, 5, 5],
        "input_shape": [1, 16, 11, 11],
        "batch_size": 8,
    }
    layer = ConvLayer(env, **args)
    layer.set_tiling("N", [2, 2, 2, 1])
    layer.set_tiling("M", [4, 2, 2, 1])
    layer.set_tiling("C", [1, 2, 1, 8])
    layer.set_tiling("Ox", [1, 1, 5, 1])
    layer.set_tiling("Oy", [1, 5, 1, 1])
    layer.set_tiling("Fx", [1, 1, 3, 1])
    layer.set_tiling("Fy", [1, 1, 3, 1])

    for level in layer._loop_IVs:
        layer.set_ordering(level, ["N", "C", "Fx", "Fy", "M", "Ox", "Oy"])

    return layer


def test_commonFunctions():

    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()
    assert data_RF["I"] == 66
    assert data_RF["W"] == 18
    assert data_RF["O"] == 20

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["I"] == 528
    assert data_PE_array["W"] == 144
    assert data_PE_array["O"] == 20

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["I"] == 7744
    assert data_SPM["W"] == 576
    assert data_SPM["O"] == 400


def test_TotalEnergy():
    layer = get_sample_layer()

    energy_ops = layer.computeEnergyCost_UsefulOps()
    assert abs(energy_ops - 34560) < 1.0e-5

    energy_RF = layer.computeEnergyCost_RF()
    assert abs(energy_RF - 1769472) < 1.0e-5

    key_1_spm, key_2_spm, key_3_spm, key_4_spm, key_5_spm = spm_orderings
    key_1_dram, key_2_dram = dram_orderings

    energy_SPM_pass = layer.get_Energy_One_SPM_Pass()
    assert abs(energy_SPM_pass[(key_1_dram, key_1_spm)] - 442914) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_1_spm)] - 442914) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_2_spm)] - 305508) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_2_spm)] - 305508) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_3_spm)] - 424548) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_3_spm)] - 424548) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_4_spm)] - 397764) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_4_spm)] - 397764) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_5_spm)] - 388836) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_5_spm)] - 388836) < 1

    total_SPM_pass = layer.get_TripCounts_all_IVs("DRAM")
    assert abs(total_SPM_pass - 8) == 0

    energy_DRAM = layer.get_Energy_DataCommunication_DRAM()
    assert abs(energy_DRAM[(key_1_dram, key_1_spm)] - 4659200) < 1
    assert abs(energy_DRAM[(key_1_dram, key_2_spm)] - 4659200) < 1
    assert abs(energy_DRAM[(key_1_dram, key_3_spm)] - 4659200) < 1
    assert abs(energy_DRAM[(key_1_dram, key_4_spm)] - 4659200) < 1
    assert abs(energy_DRAM[(key_1_dram, key_5_spm)] - 4659200) < 1
    assert abs(energy_DRAM[(key_2_dram, key_1_spm)] - 13491200) < 1
    assert abs(energy_DRAM[(key_2_dram, key_2_spm)] - 13491200) < 1
    assert abs(energy_DRAM[(key_2_dram, key_3_spm)] - 13491200) < 1
    assert abs(energy_DRAM[(key_2_dram, key_4_spm)] - 13491200) < 1
    assert abs(energy_DRAM[(key_2_dram, key_5_spm)] - 13491200) < 1

    energy = layer.get_Energy_One_Layer()
    for d in dram_orderings:
        for s in spm_orderings:
            expected_energy = round(energy_ops + energy_RF + (energy_SPM_pass[(d,s)] * total_SPM_pass) + energy_DRAM[(d,s)])
            assert abs(energy[(d,s)] - expected_energy) == 0


def test_Performance():
    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 180

    cycles = layer.calculate_Cycles_DataDistribution_NOC()

    assert cycles[layer._I.name] == 528
    assert cycles[layer._W.name] == 144
    assert cycles[layer._O.name] == 20

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 140

    key_1_spm, key_2_spm, key_3_spm, key_4_spm, key_5_spm = spm_orderings
    key_1_dram, key_2_dram = dram_orderings

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][0][1] == 528
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][0][1] == 528
    assert comm_cycles_dict[(key_1_dram, key_2_spm)][0][1] == 144
    assert comm_cycles_dict[(key_2_dram, key_2_spm)][0][1] == 144
    assert comm_cycles_dict[(key_1_dram, key_3_spm)][0][1] == 528
    assert comm_cycles_dict[(key_2_dram, key_3_spm)][0][1] == 528
    assert comm_cycles_dict[(key_1_dram, key_4_spm)][0][1] == 528
    assert comm_cycles_dict[(key_2_dram, key_4_spm)][0][1] == 528
    assert comm_cycles_dict[(key_1_dram, key_5_spm)][0][1] == 528
    assert comm_cycles_dict[(key_2_dram, key_5_spm)][0][1] == 528

    # Check stall cycles for each RF pass
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_1_dram, key_2_spm)][1][1] == 320
    assert comm_cycles_dict[(key_2_dram, key_2_spm)][1][1] == 320
    assert comm_cycles_dict[(key_1_dram, key_3_spm)][1][1] == 320
    assert comm_cycles_dict[(key_2_dram, key_3_spm)][1][1] == 320
    assert comm_cycles_dict[(key_1_dram, key_4_spm)][1][1] == 320
    assert comm_cycles_dict[(key_2_dram, key_4_spm)][1][1] == 320
    assert comm_cycles_dict[(key_1_dram, key_5_spm)][1][1] == 320
    assert comm_cycles_dict[(key_2_dram, key_5_spm)][1][1] == 320

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    assert dict_cycles_spm_pass[(key_1_dram, key_1_spm)] == 27520
    assert dict_cycles_spm_pass[(key_2_dram, key_1_spm)] == 27520
    assert dict_cycles_spm_pass[(key_1_dram, key_2_spm)] == 26240
    assert dict_cycles_spm_pass[(key_2_dram, key_2_spm)] == 26240
    assert dict_cycles_spm_pass[(key_1_dram, key_3_spm)] == 33920
    assert dict_cycles_spm_pass[(key_2_dram, key_3_spm)] == 33920
    assert dict_cycles_spm_pass[(key_1_dram, key_4_spm)] == 33920
    assert dict_cycles_spm_pass[(key_2_dram, key_4_spm)] == 33920
    assert dict_cycles_spm_pass[(key_1_dram, key_5_spm)] == 33920
    assert dict_cycles_spm_pass[(key_2_dram, key_5_spm)] == 33920

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["I"]["dma_invocations"] == 1
    assert dma_accesses["I"]["data_accessed_dma"] == 7744
    assert dma_accesses["W"]["dma_invocations"] == 1
    assert dma_accesses["W"]["data_accessed_dma"] == 576
    assert dma_accesses["O"]["dma_invocations"] == 4
    assert dma_accesses["O"]["data_accessed_dma"] == 100

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 627
    assert dma_cycles["W"] == 89
    assert dma_cycles["O"] == 212

    dict_cycles = layer.get_Cycles_One_Layer()
    assert dict_cycles[(key_1_dram, key_1_spm)] == 27520*8
    assert dict_cycles[(key_2_dram, key_1_spm)] == 27520*8
    assert dict_cycles[(key_1_dram, key_2_spm)] == 26240*8
    assert dict_cycles[(key_2_dram, key_2_spm)] == 26240*8
    assert dict_cycles[(key_1_dram, key_3_spm)] == 33920*8
    assert dict_cycles[(key_2_dram, key_3_spm)] == 33920*8
    assert dict_cycles[(key_1_dram, key_4_spm)] == 33920*8
    assert dict_cycles[(key_2_dram, key_4_spm)] == 33920*8
    assert dict_cycles[(key_1_dram, key_5_spm)] == 33920*8
    assert dict_cycles[(key_2_dram, key_5_spm)] == 33920*8

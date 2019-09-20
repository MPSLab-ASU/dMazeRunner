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
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts ]
    global spm_orderings
    spm_orderings = tuple(orders)
    assert(len(spm_orderings) == 3)

    dicts = [
        {"O": 2, "I": 1, "W": 1},
        {"O": 1, "I": 2, "W": 1},
        {"O": 1, "I": 1, "W": 2},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "DRAM") for d in dicts ]
    global dram_orderings
    dram_orderings = tuple(orders)
    assert(len(dram_orderings) == 3)


def get_sample_layer():
    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    args = {
        "name": "test_conv2d12_from_resnet18_v1",
        "channels": 8,
        "kernel_size": "[3, 3]",
        "padding": "[1, 1]",
        "strides": "[1, 1]",
        "output_shape": [1, 16, 10, 10],
        "input_shape": [1, 8, 10, 10],
        "batch_size": 4,
    }
    layer = ConvLayer(env, **args)
    layer.set_tiling("N", [2, 1, 2, 1])
    layer.set_tiling("M", [2, 2, 1, 4])
    layer.set_tiling("C", [2, 2, 2, 1])
    layer.set_tiling("Ox", [1, 2, 5, 1])
    layer.set_tiling("Oy", [1, 1, 10, 1])
    layer.set_tiling("Fx", [1, 1, 3, 1])
    layer.set_tiling("Fy", [1, 1, 3, 1])

    for level in layer._loop_IVs:
        layer.set_ordering(level, ["N", "C", "Fx", "Fy", "M", "Ox", "Oy"])

    return layer


def test_commonFunctions():

    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()
    assert data_RF["I"] == 336
    assert data_RF["W"] == 18
    assert data_RF["O"] == 100

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["I"] == 336
    assert data_PE_array["W"] == 72
    assert data_PE_array["O"] == 400

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["I"] == 1152
    assert data_SPM["W"] == 288
    assert data_SPM["O"] == 1600


def test_TotalEnergy():
    layer = get_sample_layer()

    energy_ops = layer.computeEnergyCost_UsefulOps()
    assert abs(energy_ops - 34560) < 1.0e-5

    energy_RF = layer.computeEnergyCost_RF()
    assert abs(energy_RF - 1769472) < 1.0e-5

    key_1_spm, key_2_spm, key_3_spm = spm_orderings
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    energy_SPM_pass = layer.get_Energy_One_SPM_Pass()
    assert abs(energy_SPM_pass[(key_1_dram, key_1_spm)] - 116320) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_1_spm)] - 116320) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_1_spm)] - 116320) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_2_spm)] - 137024) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_2_spm)] - 137024) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_2_spm)] - 137024) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_3_spm)] - 161456) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_3_spm)] - 161456) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_3_spm)] - 161456) < 1

    total_SPM_pass = layer.get_TripCounts_all_IVs("DRAM")
    assert abs(total_SPM_pass - 8) == 0

    energy_DRAM = layer.get_Energy_DataCommunication_DRAM()
    assert abs(energy_DRAM[(key_1_dram, key_1_spm)] - 3584000) < 1
    assert abs(energy_DRAM[(key_1_dram, key_2_spm)] - 3584000) < 1
    assert abs(energy_DRAM[(key_1_dram, key_3_spm)] - 3584000) < 1
    assert abs(energy_DRAM[(key_2_dram, key_1_spm)] - 5222400) < 1
    assert abs(energy_DRAM[(key_2_dram, key_2_spm)] - 5222400) < 1
    assert abs(energy_DRAM[(key_2_dram, key_3_spm)] - 5222400) < 1
    assert abs(energy_DRAM[(key_3_dram, key_1_spm)] - 5913600) < 1
    assert abs(energy_DRAM[(key_3_dram, key_2_spm)] - 5913600) < 1
    assert abs(energy_DRAM[(key_3_dram, key_3_spm)] - 5913600) < 1

    energy = layer.get_Energy_One_Layer()
    for d in dram_orderings:
        for s in spm_orderings:
            expected_energy = round(energy_ops + energy_RF + (energy_SPM_pass[(d,s)] * total_SPM_pass) + energy_DRAM[(d,s)])
            assert abs(energy[(d,s)] - expected_energy) == 0


def test_Performance():
    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 1800

    cycles = layer.calculate_Cycles_DataDistribution_NOC()

    assert cycles[layer._I.name] == 336
    assert cycles[layer._W.name] == 72
    assert cycles[layer._O.name] == 400

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 0

    key_1_spm, key_2_spm, key_3_spm = spm_orderings
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][0][1] == 336
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][0][1] == 336
    assert comm_cycles_dict[(key_3_dram, key_1_spm)][0][1] == 336
    assert comm_cycles_dict[(key_1_dram, key_2_spm)][0][1] == 400
    assert comm_cycles_dict[(key_2_dram, key_2_spm)][0][1] == 400
    assert comm_cycles_dict[(key_3_dram, key_2_spm)][0][1] == 400
    assert comm_cycles_dict[(key_1_dram, key_3_spm)][0][1] == 400
    assert comm_cycles_dict[(key_2_dram, key_3_spm)][0][1] == 400
    assert comm_cycles_dict[(key_3_dram, key_3_spm)][0][1] == 400

    # Check stall cycles for each RF pass
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_3_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_1_dram, key_2_spm)][1][1] == 1800
    assert comm_cycles_dict[(key_2_dram, key_2_spm)][1][1] == 1800
    assert comm_cycles_dict[(key_3_dram, key_2_spm)][1][1] == 1800
    assert comm_cycles_dict[(key_1_dram, key_3_spm)][1][1] == 1800
    assert comm_cycles_dict[(key_2_dram, key_3_spm)][1][1] == 1800
    assert comm_cycles_dict[(key_3_dram, key_3_spm)][1][1] == 1800

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    assert dict_cycles_spm_pass[(key_1_dram, key_1_spm)] == 21600
    assert dict_cycles_spm_pass[(key_2_dram, key_1_spm)] == 21600
    assert dict_cycles_spm_pass[(key_3_dram, key_1_spm)] == 21600
    assert dict_cycles_spm_pass[(key_1_dram, key_2_spm)] == 17600
    assert dict_cycles_spm_pass[(key_2_dram, key_2_spm)] == 17600
    assert dict_cycles_spm_pass[(key_3_dram, key_2_spm)] == 17600
    assert dict_cycles_spm_pass[(key_1_dram, key_3_spm)] == 17600
    assert dict_cycles_spm_pass[(key_2_dram, key_3_spm)] == 17600
    assert dict_cycles_spm_pass[(key_3_dram, key_3_spm)] == 17600

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["I"]["dma_invocations"] == 2
    assert dma_accesses["I"]["data_accessed_dma"] == 576
    assert dma_accesses["W"]["dma_invocations"] == 8
    assert dma_accesses["W"]["data_accessed_dma"] == 36
    assert dma_accesses["O"]["dma_invocations"] == 2
    assert dma_accesses["O"]["data_accessed_dma"] == 800

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 178
    assert dma_cycles["W"] == 392
    assert dma_cycles["O"] == 212

    dict_cycles = layer.get_Cycles_One_Layer()
    assert dict_cycles[(key_1_dram, key_1_spm)] == 172800
    assert dict_cycles[(key_2_dram, key_1_spm)] == 172800
    assert dict_cycles[(key_3_dram, key_1_spm)] == 172800
    assert dict_cycles[(key_1_dram, key_2_spm)] == 140800
    assert dict_cycles[(key_2_dram, key_2_spm)] == 140800
    assert dict_cycles[(key_3_dram, key_2_spm)] == 140800
    assert dict_cycles[(key_1_dram, key_3_spm)] == 140800
    assert dict_cycles[(key_2_dram, key_3_spm)] == 140800
    assert dict_cycles[(key_3_dram, key_3_spm)] == 140800

import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters

spm_orderings = ()
dram_orderings = ()

def setup():
    layer = get_sample_layer()
    dicts = [
        {"O": 4, "I": 1, "W": 1},
        {"O": 2, "I": 1, "W": 1},
        {"O": 1, "I": 4, "W": 1},
        {"O": 1, "I": 1, "W": 2},
        {"O": 1, "I": 1, "W": 5},
        {"O": 1, "I": 1, "W": 10},
    ]
    orders = [layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts]
    global spm_orderings
    spm_orderings = tuple(orders)
    assert(len(spm_orderings) == 6)

    dicts = [
        {"O": 2, "I": 1, "W": 1},
        {"O": 4, "I": 1, "W": 1},
        {"O": 8, "I": 1, "W": 1},
        {"O": 1, "I": 2, "W": 1},
        {"O": 1, "I": 1, "W": 2},
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
        "channels": 64,
        "kernel_size": "[6, 6]",
        "padding": "[2, 2]",
        "strides": "[2, 2]",
        "output_shape": [1, 64, 5, 5],
        "input_shape": [1, 64, 10, 10],
        "batch_size": 8,
    }
    layer = ConvLayer(env, **args)
    layer.set_tiling("N", [2, 2, 2, 1])
    layer.set_tiling("M", [2, 4, 2, 4])
    layer.set_tiling("C", [4, 2, 2, 4])
    layer.set_tiling("Ox", [1, 5, 1, 1])
    layer.set_tiling("Oy", [1, 1, 5, 1])
    layer.set_tiling("Fx", [1, 2, 3, 1])
    layer.set_tiling("Fy", [2, 1, 3, 1])

    for level in layer._loop_IVs:
        layer.set_ordering(level, ["N", "C", "Fx", "Fy", "M", "Ox", "Oy"])

    return layer


def test_commonFunctions():

    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()
    assert data_RF["I"] == 132
    assert data_RF["W"] == 36
    assert data_RF["O"] == 20

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["I"] == 528
    assert data_PE_array["W"] == 576
    assert data_PE_array["O"] == 80

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["I"] == 9856
    assert data_SPM["W"] == 9216
    assert data_SPM["O"] == 3200


def test_TotalEnergy():

    layer = get_sample_layer()
    energy_ops = layer.computeEnergyCost_UsefulOps()
    assert abs(energy_ops - 2211840) < 1.0e-5

    energy_RF = layer.computeEnergyCost_RF()
    assert abs(energy_RF - 113246208) < 1.0e-5

    key_1_spm = spm_orderings[0]
    key_2_spm = spm_orderings[1]
    key_3_spm = spm_orderings[2]
    key_4_spm = spm_orderings[3]
    key_5_spm = spm_orderings[4]
    key_6_spm = spm_orderings[5]

    key_1_dram = dram_orderings[0]
    key_2_dram = dram_orderings[1]
    key_3_dram = dram_orderings[2]
    key_4_dram = dram_orderings[3]
    key_5_dram = dram_orderings[4]

    energy_SPM_pass = layer.get_Energy_One_SPM_Pass()
    assert abs(energy_SPM_pass[(key_1_dram, key_1_spm)] - 3391568) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_2_spm)] - 3538336) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_3_spm)] - 2469632) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_4_spm)] - 3117632) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_5_spm)] - 2689088) < 1
    assert abs(energy_SPM_pass[(key_1_dram, key_6_spm)] - 2546240) < 1

    assert abs(energy_SPM_pass[(key_2_dram, key_1_spm)] - 3391568) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_2_spm)] - 3538336) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_3_spm)] - 2469632) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_4_spm)] - 3117632) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_5_spm)] - 2689088) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_6_spm)] - 2546240) < 1

    assert abs(energy_SPM_pass[(key_3_dram, key_1_spm)] - 3391568) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_2_spm)] - 3538336) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_3_spm)] - 2469632) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_4_spm)] - 3117632) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_5_spm)] - 2689088) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_6_spm)] - 2546240) < 1

    assert abs(energy_SPM_pass[(key_4_dram, key_1_spm)] - 3391568) < 1
    assert abs(energy_SPM_pass[(key_4_dram, key_2_spm)] - 3538336) < 1
    assert abs(energy_SPM_pass[(key_4_dram, key_3_spm)] - 2469632) < 1
    assert abs(energy_SPM_pass[(key_4_dram, key_4_spm)] - 3117632) < 1
    assert abs(energy_SPM_pass[(key_4_dram, key_5_spm)] - 2689088) < 1
    assert abs(energy_SPM_pass[(key_4_dram, key_6_spm)] - 2546240) < 1

    assert abs(energy_SPM_pass[(key_5_dram, key_1_spm)] - 3391568) < 1
    assert abs(energy_SPM_pass[(key_5_dram, key_2_spm)] - 3538336) < 1
    assert abs(energy_SPM_pass[(key_5_dram, key_3_spm)] - 2469632) < 1
    assert abs(energy_SPM_pass[(key_5_dram, key_4_spm)] - 3117632) < 1
    assert abs(energy_SPM_pass[(key_5_dram, key_5_spm)] - 2689088) < 1
    assert abs(energy_SPM_pass[(key_5_dram, key_6_spm)] - 2546240) < 1

    total_SPM_pass = layer.get_TripCounts_all_IVs("DRAM")
    assert abs(total_SPM_pass - 32) == 0

    energy_DRAM = layer.get_Energy_DataCommunication_DRAM()
    # print(layer.determine_data_reuse("DRAM"))
    # print(energy_DRAM[(key_1_dram, key_1_spm)])
    assert abs(energy_DRAM[(key_1_dram, key_1_spm)] - 139980800) < 1
    assert abs(energy_DRAM[(key_1_dram, key_2_spm)] - 139980800) < 1
    assert abs(energy_DRAM[(key_1_dram, key_3_spm)] - 139980800) < 1
    assert abs(energy_DRAM[(key_1_dram, key_4_spm)] - 139980800) < 1
    assert abs(energy_DRAM[(key_1_dram, key_5_spm)] - 139980800) < 1
    assert abs(energy_DRAM[(key_1_dram, key_6_spm)] - 139980800) < 1

    assert abs(energy_DRAM[(key_2_dram, key_1_spm)] - 129740800) < 1
    assert abs(energy_DRAM[(key_2_dram, key_2_spm)] - 129740800) < 1
    assert abs(energy_DRAM[(key_2_dram, key_3_spm)] - 129740800) < 1
    assert abs(energy_DRAM[(key_2_dram, key_4_spm)] - 129740800) < 1
    assert abs(energy_DRAM[(key_2_dram, key_5_spm)] - 129740800) < 1
    assert abs(energy_DRAM[(key_2_dram, key_6_spm)] - 129740800) < 1

    assert abs(energy_DRAM[(key_3_dram, key_1_spm)] - 124620800) < 1
    assert abs(energy_DRAM[(key_3_dram, key_2_spm)] - 124620800) < 1
    assert abs(energy_DRAM[(key_3_dram, key_3_spm)] - 124620800) < 1
    assert abs(energy_DRAM[(key_3_dram, key_4_spm)] - 124620800) < 1
    assert abs(energy_DRAM[(key_3_dram, key_5_spm)] - 124620800) < 1
    assert abs(energy_DRAM[(key_3_dram, key_6_spm)] - 124620800) < 1

    assert abs(energy_DRAM[(key_4_dram, key_1_spm)] - 128921600) < 1
    assert abs(energy_DRAM[(key_4_dram, key_2_spm)] - 128921600) < 1
    assert abs(energy_DRAM[(key_4_dram, key_3_spm)] - 128921600) < 1
    assert abs(energy_DRAM[(key_4_dram, key_4_spm)] - 128921600) < 1
    assert abs(energy_DRAM[(key_4_dram, key_5_spm)] - 128921600) < 1
    assert abs(energy_DRAM[(key_4_dram, key_6_spm)] - 128921600) < 1

    assert abs(energy_DRAM[(key_5_dram, key_1_spm)] - 130969600) < 1
    assert abs(energy_DRAM[(key_5_dram, key_2_spm)] - 130969600) < 1
    assert abs(energy_DRAM[(key_5_dram, key_3_spm)] - 130969600) < 1
    assert abs(energy_DRAM[(key_5_dram, key_4_spm)] - 130969600) < 1
    assert abs(energy_DRAM[(key_5_dram, key_5_spm)] - 130969600) < 1
    assert abs(energy_DRAM[(key_5_dram, key_6_spm)] - 130969600) < 1

    energy = layer.get_Energy_One_Layer()
    for d in dram_orderings:
        for s in spm_orderings:
            expected_energy = round(energy_ops + energy_RF + (energy_SPM_pass[(d,s)] * total_SPM_pass) + energy_DRAM[(d,s)])
            assert abs(energy[(d,s)] - expected_energy) == 0


def test_Performance():

    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 360

    cycles = layer.calculate_Cycles_DataDistribution_NOC()

    assert cycles[layer._O.name] == 80
    assert cycles[layer._I.name] == 528
    assert cycles[layer._W.name] == 576

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 240

    key_1_spm = spm_orderings[0]
    key_2_spm = spm_orderings[1]
    key_3_spm = spm_orderings[2]
    key_4_spm = spm_orderings[3]
    key_5_spm = spm_orderings[4]
    key_6_spm = spm_orderings[5]

    key_1_dram = dram_orderings[0]
    key_2_dram = dram_orderings[1]
    key_3_dram = dram_orderings[2]
    key_4_dram = dram_orderings[3]
    key_5_dram = dram_orderings[4]

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][0][1] == 576
    assert comm_cycles_dict[(key_1_dram, key_2_spm)][0][1] == 576
    assert comm_cycles_dict[(key_1_dram, key_3_spm)][0][1] == 576
    assert comm_cycles_dict[(key_1_dram, key_4_spm)][0][1] == 528
    assert comm_cycles_dict[(key_1_dram, key_5_spm)][0][1] == 528
    assert comm_cycles_dict[(key_1_dram, key_6_spm)][0][1] == 528

    assert comm_cycles_dict[(key_2_dram, key_1_spm)][0][1] == 576
    assert comm_cycles_dict[(key_2_dram, key_2_spm)][0][1] == 576
    assert comm_cycles_dict[(key_2_dram, key_3_spm)][0][1] == 576
    assert comm_cycles_dict[(key_2_dram, key_4_spm)][0][1] == 528
    assert comm_cycles_dict[(key_2_dram, key_5_spm)][0][1] == 528
    assert comm_cycles_dict[(key_2_dram, key_6_spm)][0][1] == 528

    assert comm_cycles_dict[(key_3_dram, key_1_spm)][0][1] == 576
    assert comm_cycles_dict[(key_3_dram, key_2_spm)][0][1] == 576
    assert comm_cycles_dict[(key_3_dram, key_3_spm)][0][1] == 576
    assert comm_cycles_dict[(key_3_dram, key_4_spm)][0][1] == 528
    assert comm_cycles_dict[(key_3_dram, key_5_spm)][0][1] == 528
    assert comm_cycles_dict[(key_3_dram, key_6_spm)][0][1] == 528

    assert comm_cycles_dict[(key_4_dram, key_1_spm)][0][1] == 576
    assert comm_cycles_dict[(key_4_dram, key_2_spm)][0][1] == 576
    assert comm_cycles_dict[(key_4_dram, key_3_spm)][0][1] == 576
    assert comm_cycles_dict[(key_4_dram, key_4_spm)][0][1] == 528
    assert comm_cycles_dict[(key_4_dram, key_5_spm)][0][1] == 528
    assert comm_cycles_dict[(key_4_dram, key_6_spm)][0][1] == 528

    assert comm_cycles_dict[(key_5_dram, key_1_spm)][0][1] == 576
    assert comm_cycles_dict[(key_5_dram, key_2_spm)][0][1] == 576
    assert comm_cycles_dict[(key_5_dram, key_3_spm)][0][1] == 576
    assert comm_cycles_dict[(key_5_dram, key_4_spm)][0][1] == 528
    assert comm_cycles_dict[(key_5_dram, key_5_spm)][0][1] == 528
    assert comm_cycles_dict[(key_5_dram, key_6_spm)][0][1] == 528

    # Check stall cycles for each RF pass
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_1_dram, key_2_spm)][1][1] == 0
    assert comm_cycles_dict[(key_1_dram, key_3_spm)][1][1] == 600
    assert comm_cycles_dict[(key_1_dram, key_4_spm)][1][1] == 600
    assert comm_cycles_dict[(key_1_dram, key_5_spm)][1][1] == 600
    assert comm_cycles_dict[(key_1_dram, key_6_spm)][1][1] == 600

    assert comm_cycles_dict[(key_2_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_2_dram, key_2_spm)][1][1] == 0
    assert comm_cycles_dict[(key_2_dram, key_3_spm)][1][1] == 600
    assert comm_cycles_dict[(key_2_dram, key_4_spm)][1][1] == 600
    assert comm_cycles_dict[(key_2_dram, key_5_spm)][1][1] == 600
    assert comm_cycles_dict[(key_2_dram, key_6_spm)][1][1] == 600

    assert comm_cycles_dict[(key_3_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_3_dram, key_2_spm)][1][1] == 0
    assert comm_cycles_dict[(key_3_dram, key_3_spm)][1][1] == 600
    assert comm_cycles_dict[(key_3_dram, key_4_spm)][1][1] == 600
    assert comm_cycles_dict[(key_3_dram, key_5_spm)][1][1] == 600
    assert comm_cycles_dict[(key_3_dram, key_6_spm)][1][1] == 600

    assert comm_cycles_dict[(key_4_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_4_dram, key_2_spm)][1][1] == 0
    assert comm_cycles_dict[(key_4_dram, key_3_spm)][1][1] == 600
    assert comm_cycles_dict[(key_4_dram, key_4_spm)][1][1] == 600
    assert comm_cycles_dict[(key_4_dram, key_5_spm)][1][1] == 600
    assert comm_cycles_dict[(key_4_dram, key_6_spm)][1][1] == 600

    assert comm_cycles_dict[(key_5_dram, key_1_spm)][1][1] == 0
    assert comm_cycles_dict[(key_5_dram, key_2_spm)][1][1] == 0
    assert comm_cycles_dict[(key_5_dram, key_3_spm)][1][1] == 600
    assert comm_cycles_dict[(key_5_dram, key_4_spm)][1][1] == 600
    assert comm_cycles_dict[(key_5_dram, key_5_spm)][1][1] == 600
    assert comm_cycles_dict[(key_5_dram, key_6_spm)][1][1] == 600

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    assert dict_cycles_spm_pass[(key_1_dram, key_1_spm)] == 116160
    assert dict_cycles_spm_pass[(key_1_dram, key_2_spm)] == 140160
    assert dict_cycles_spm_pass[(key_1_dram, key_3_spm)] == 188160
    assert dict_cycles_spm_pass[(key_1_dram, key_4_spm)] == 184320
    assert dict_cycles_spm_pass[(key_1_dram, key_5_spm)] == 182016
    assert dict_cycles_spm_pass[(key_1_dram, key_6_spm)] == 181248

    assert dict_cycles_spm_pass[(key_2_dram, key_1_spm)] == 116160
    assert dict_cycles_spm_pass[(key_2_dram, key_2_spm)] == 140160
    assert dict_cycles_spm_pass[(key_2_dram, key_3_spm)] == 188160
    assert dict_cycles_spm_pass[(key_2_dram, key_4_spm)] == 184320
    assert dict_cycles_spm_pass[(key_2_dram, key_5_spm)] == 182016
    assert dict_cycles_spm_pass[(key_2_dram, key_6_spm)] == 181248

    assert dict_cycles_spm_pass[(key_3_dram, key_1_spm)] == 116160
    assert dict_cycles_spm_pass[(key_3_dram, key_2_spm)] == 140160
    assert dict_cycles_spm_pass[(key_3_dram, key_3_spm)] == 188160
    assert dict_cycles_spm_pass[(key_3_dram, key_4_spm)] == 184320
    assert dict_cycles_spm_pass[(key_3_dram, key_5_spm)] == 182016
    assert dict_cycles_spm_pass[(key_3_dram, key_6_spm)] == 181248

    assert dict_cycles_spm_pass[(key_4_dram, key_1_spm)] == 116160
    assert dict_cycles_spm_pass[(key_4_dram, key_2_spm)] == 140160
    assert dict_cycles_spm_pass[(key_4_dram, key_3_spm)] == 188160
    assert dict_cycles_spm_pass[(key_4_dram, key_4_spm)] == 184320
    assert dict_cycles_spm_pass[(key_4_dram, key_5_spm)] == 182016
    assert dict_cycles_spm_pass[(key_4_dram, key_6_spm)] == 181248

    assert dict_cycles_spm_pass[(key_5_dram, key_1_spm)] == 116160
    assert dict_cycles_spm_pass[(key_5_dram, key_2_spm)] == 140160
    assert dict_cycles_spm_pass[(key_5_dram, key_3_spm)] == 188160
    assert dict_cycles_spm_pass[(key_5_dram, key_4_spm)] == 184320
    assert dict_cycles_spm_pass[(key_5_dram, key_5_spm)] == 182016
    assert dict_cycles_spm_pass[(key_5_dram, key_6_spm)] == 181248

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["I"]["dma_invocations"] == 896
    assert dma_accesses["I"]["data_accessed_dma"] == 11
    assert dma_accesses["W"]["dma_invocations"] == 3072
    assert dma_accesses["W"]["data_accessed_dma"] == 3
    assert dma_accesses["O"]["dma_invocations"] == 4
    assert dma_accesses["O"]["data_accessed_dma"] == 800

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 42112
    assert dma_cycles["W"] == 141312
    assert dma_cycles["O"] == 424

    dict_cycles = layer.get_Cycles_One_Layer()
    assert dict_cycles[(key_1_dram, key_1_spm)] == 5883136
    assert dict_cycles[(key_1_dram, key_2_spm)] == 5883136
    assert dict_cycles[(key_1_dram, key_3_spm)] == 6021120
    assert dict_cycles[(key_1_dram, key_4_spm)] == 5898240
    assert dict_cycles[(key_1_dram, key_5_spm)] == 5883136
    assert dict_cycles[(key_1_dram, key_6_spm)] == 5883136

    assert dict_cycles[(key_2_dram, key_1_spm)] == 5876352
    assert dict_cycles[(key_2_dram, key_2_spm)] == 5876352
    assert dict_cycles[(key_2_dram, key_3_spm)] == 6021120
    assert dict_cycles[(key_2_dram, key_4_spm)] == 5898240
    assert dict_cycles[(key_2_dram, key_5_spm)] == 5876352
    assert dict_cycles[(key_2_dram, key_6_spm)] == 5876352

    assert dict_cycles[(key_3_dram, key_1_spm)] == 5872960
    assert dict_cycles[(key_3_dram, key_2_spm)] == 5872960
    assert dict_cycles[(key_3_dram, key_3_spm)] == 6021120
    assert dict_cycles[(key_3_dram, key_4_spm)] == 5898240
    assert dict_cycles[(key_3_dram, key_5_spm)] == 5872960
    assert dict_cycles[(key_3_dram, key_6_spm)] == 5872960

    assert dict_cycles[(key_4_dram, key_1_spm)] == 5222912
    assert dict_cycles[(key_4_dram, key_2_spm)] == 5222912
    assert dict_cycles[(key_4_dram, key_3_spm)] == 6021120
    assert dict_cycles[(key_4_dram, key_4_spm)] == 5898240
    assert dict_cycles[(key_4_dram, key_5_spm)] == 5860608
    assert dict_cycles[(key_4_dram, key_6_spm)] == 5848320

    assert dict_cycles[(key_5_dram, key_1_spm)] == 4806912
    assert dict_cycles[(key_5_dram, key_2_spm)] == 5190912
    assert dict_cycles[(key_5_dram, key_3_spm)] == 6021120
    assert dict_cycles[(key_5_dram, key_4_spm)] == 5898240
    assert dict_cycles[(key_5_dram, key_5_spm)] == 5860608
    assert dict_cycles[(key_5_dram, key_6_spm)] == 5848320

import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters

spm_orderings = ()
dram_orderings = ()

def setup():
    layer = get_sample_layer()
    dicts = [
        {"O": 1, "I": 8, "W": 1},
        {"O": 1, "I": 1, "W": 2},
        {"O": 4, "I": 1, "W": 1}
    ]
    assert len(layer.determine_data_reuse("SPM")) == 3
    orders = [ layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts ]
    global spm_orderings
    spm_orderings = tuple(orders)

    dicts = [
        {"O": 1, "I": 4, "W": 1},
        {"O": 2, "I": 1, "W": 1},
        {"O": 1, "I": 1, "W": 4}
    ]
    assert len(layer.determine_data_reuse("DRAM")) == 3
    orders = [ layer.get_ordering_from_reuse_factor(d, "DRAM") for d in dicts ]
    global dram_orderings
    dram_orderings = tuple(orders)


def get_sample_layer():
    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    args = {
        "name": "test_conv2d12_from_resnet18_v1",
        "channels": 32,
        "kernel_size": "[3, 3]",
        "padding": "[1, 1]",
        "strides": "[1, 1]",
        "output_shape": [1, 64, 3, 3],
        "input_shape": [1, 32, 3, 3],
        "batch_size": 16,
    }
    layer = ConvLayer(env, **args)
    layer.set_tiling("N", [4, 2, 2, 1])
    layer.set_tiling("M", [4, 8, 1, 2])
    layer.set_tiling("C", [2, 4, 2, 2])
    layer.set_tiling("Ox", [1, 1, 3, 1])
    layer.set_tiling("Oy", [1, 1, 3, 1])
    layer.set_tiling("Fx", [1, 1, 1, 3])
    layer.set_tiling("Fy", [1, 1, 1, 3])

    for level in layer._loop_IVs:
        layer.set_ordering(level, ["N", "C", "Fx", "Fy", "M", "Ox", "Oy"])

    return layer


def test_commonFunctions():
    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()
    assert data_RF["I"] == 36
    assert data_RF["W"] == 2
    assert data_RF["O"] == 18

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["I"] == 200
    assert data_PE_array["W"] == 72
    assert data_PE_array["O"] == 36

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["I"] == 1600
    assert data_SPM["W"] == 2304
    assert data_SPM["O"] == 576


def test_TotalEnergy():
    layer = get_sample_layer()
    energy_ops = layer.computeEnergyCost_UsefulOps()
    assert abs(energy_ops - 199065.6) < 1.0e-5

    energy_RF = layer.computeEnergyCost_RF()
    assert abs(energy_RF - 10192158.72) < 1.0e-5

    key_1_spm, key_2_spm, key_3_spm = spm_orderings
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    energy_SPM_pass = layer.get_Energy_One_SPM_Pass()
    for key_dram in dram_orderings:
        assert abs(energy_SPM_pass[(key_dram, key_1_spm)] - 379261.44) < 1
        assert abs(energy_SPM_pass[(key_dram, key_2_spm)] - 639901.44) < 1
        assert abs(energy_SPM_pass[(key_dram, key_3_spm)] - 476487.36) < 1

    total_SPM_pass = layer.get_TripCounts_all_IVs("DRAM")
    assert abs(total_SPM_pass - 32) == 0

    energy_DRAM = layer.get_Energy_DataCommunication_DRAM()
    for key_spm in spm_orderings:
        assert abs(energy_DRAM[(key_1_dram, key_spm)] - 22835200) < 1
        assert abs(energy_DRAM[(key_2_dram, key_spm)] - 26828800) < 1
        assert abs(energy_DRAM[(key_3_dram, key_spm)] - 19456000) < 1

    energy = layer.get_Energy_One_Layer()
    for d in dram_orderings:
        for s in spm_orderings:
            expected_energy = round(energy_ops + energy_RF + (energy_SPM_pass[(d,s)] * total_SPM_pass) + energy_DRAM[(d,s)])
            assert abs(energy[(d,s)] - expected_energy) == 0


def test_Performance():
    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 36

    cycles = layer.calculate_Cycles_DataDistribution_NOC()

    assert cycles[layer._O.name] == 36
    assert cycles[layer._I.name] == 200
    assert cycles[layer._W.name] == 72

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 612

    key_1_spm, key_2_spm, key_3_spm = spm_orderings
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    for key_dram in dram_orderings:
        assert comm_cycles_dict[(key_dram, key_1_spm)][0][1] == 72
        assert comm_cycles_dict[(key_dram, key_2_spm)][0][1] == 200
        assert comm_cycles_dict[(key_dram, key_3_spm)][0][1] == 200

    # Check stall cycles for each RF pass
    for key_dram in dram_orderings:
        assert comm_cycles_dict[(key_dram, key_1_spm)][1][1] == 648
        assert comm_cycles_dict[(key_dram, key_2_spm)][1][1] == 648
        assert comm_cycles_dict[(key_dram, key_3_spm)][1][1] == 0

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    for key_dram in dram_orderings:
        assert dict_cycles_spm_pass[(key_dram, key_1_spm)] == 47104
        assert dict_cycles_spm_pass[(key_dram, key_2_spm)] == 54272
        assert dict_cycles_spm_pass[(key_dram, key_3_spm)] == 23168

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["I"]["dma_invocations"] == 4
    assert dma_accesses["I"]["data_accessed_dma"] == 400
    assert dma_accesses["W"]["dma_invocations"] == 16
    assert dma_accesses["W"]["data_accessed_dma"] == 144
    assert dma_accesses["O"]["dma_invocations"] == 4
    assert dma_accesses["O"]["data_accessed_dma"] == 144

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 304
    assert dma_cycles["W"] == 912
    assert dma_cycles["O"] == 228

    dict_cycles = layer.get_Cycles_One_Layer()
    for key_dram in dram_orderings:
        assert dict_cycles[(key_dram, key_1_spm)] == 1507328
        assert dict_cycles[(key_dram, key_2_spm)] == 1736704
        assert dict_cycles[(key_dram, key_3_spm)] == 741376

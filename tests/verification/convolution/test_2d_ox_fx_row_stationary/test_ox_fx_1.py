import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters

spm_orderings = ()
dram_orderings = ()

def setup():
    layer = get_sample_layer()
    dicts = [
        {"O": 4, "I": 1, "W": 1},
        {"O": 1, "I": 1, "W": 2},
        {"O": 8, "I": 1, "W": 1},
        {"O": 2, "I": 1, "W": 1},
        {"O": 1, "I": 2, "W": 1},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts ]
    global spm_orderings
    spm_orderings = tuple(orders)

    dicts = [
        {"O": 2, "I": 1, "W": 1},
        {"O": 1, "I": 1, "W": 4},
        {"O": 1, "I": 4, "W": 1},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "DRAM") for d in dicts ]
    global dram_orderings
    dram_orderings = tuple(orders)


def get_sample_layer():
    env = expr_parameters.Environment()
    env.pe_pipeline_stages = 1
    args = {
        "name": "test_conv2d12_from_resnet18_v1",
        "channels": 16,
        "kernel_size": "[6, 6]",
        "padding": "[1, 1]",
        "strides": "[1, 1]",
        "output_shape": [1, 32, 5, 5],
        "input_shape": [1, 16, 8, 8],
        "batch_size": 16,
    }
    layer = ConvLayer(env, **args)
    layer.set_tiling("N", [4, 2, 1, 2])
    layer.set_tiling("M", [4, 2, 2, 2])
    layer.set_tiling("C", [2, 4, 2, 1])
    layer.set_tiling("Ox", [1, 1, 1, 5])
    layer.set_tiling("Oy", [1, 1, 5, 1])
    layer.set_tiling("Fx", [1, 1, 2, 3])
    layer.set_tiling("Fy", [1, 2, 3, 1])

    for level in layer._loop_IVs:
        layer.set_ordering(level, ["M", "C", "Fx", "Fy", "N", "Ox", "Oy"])

    return layer


def test_commonFunctions():
    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()
    assert data_RF["I"] == 28
    assert data_RF["W"] == 24
    assert data_RF["O"] == 10

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["I"] == 280
    assert data_PE_array["W"] == 144
    assert data_PE_array["O"] == 200

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["I"] == 3200
    assert data_SPM["W"] == 2304
    assert data_SPM["O"] == 800

def test_Performance():
    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 120

    cycles = layer.calculate_Cycles_DataDistribution_NOC()

    assert cycles[layer._O.name] == 200
    assert cycles[layer._I.name] == 280
    assert cycles[layer._W.name] == 144

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 400

    key_1_spm, key_2_spm, key_3_spm, key_4_spm, key_5_spm = spm_orderings
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    for key_dram in dram_orderings:
        assert comm_cycles_dict[(key_dram, key_1_spm)][0][1] == 280
        assert comm_cycles_dict[(key_dram, key_2_spm)][0][1] == 280
        assert comm_cycles_dict[(key_dram, key_3_spm)][0][1] == 280
        assert comm_cycles_dict[(key_dram, key_4_spm)][0][1] == 280
        assert comm_cycles_dict[(key_dram, key_5_spm)][0][1] == 200

    # Check stall cycles for each RF pass
    for key_dram in dram_orderings:
        assert comm_cycles_dict[(key_dram, key_1_spm)][1][1] == 0
        assert comm_cycles_dict[(key_dram, key_2_spm)][1][1] == 520
        assert comm_cycles_dict[(key_dram, key_3_spm)][1][1] == 0
        assert comm_cycles_dict[(key_dram, key_4_spm)][1][1] == 0
        assert comm_cycles_dict[(key_dram, key_5_spm)][1][1] == 520

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    for key_dram in dram_orderings:
        assert dict_cycles_spm_pass[(key_dram, key_1_spm)] == 13120
        assert dict_cycles_spm_pass[(key_dram, key_2_spm)] == 25600
        assert dict_cycles_spm_pass[(key_dram, key_3_spm)] == 11040
        assert dict_cycles_spm_pass[(key_dram, key_4_spm)] == 17280
        assert dict_cycles_spm_pass[(key_dram, key_5_spm)] == 24320

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["I"]["dma_invocations"] == 4
    assert dma_accesses["I"]["data_accessed_dma"] == 800
    assert dma_accesses["W"]["dma_invocations"] == 8
    assert dma_accesses["W"]["data_accessed_dma"] == 288
    assert dma_accesses["O"]["dma_invocations"] == 4
    assert dma_accesses["O"]["data_accessed_dma"] == 200

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 424
    assert dma_cycles["W"] == 544
    assert dma_cycles["O"] == 244

    dict_cycles = layer.get_Cycles_One_Layer()
    for key_dram in dram_orderings:
        assert dict_cycles[(key_dram, key_1_spm)] == 419840
        assert dict_cycles[(key_dram, key_2_spm)] == 819200
        assert dict_cycles[(key_dram, key_3_spm)] == 353280
        assert dict_cycles[(key_dram, key_4_spm)] == 552960
        assert dict_cycles[(key_dram, key_5_spm)] == 778240

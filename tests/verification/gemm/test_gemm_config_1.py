import tvm
from dMazeRunner.dataflow import GemmLayer
from dMazeRunner.common import expr_parameters

spm_orderings = ()
dram_orderings = ()

def setup():
    layer = get_sample_layer()

    dicts = [
        {'C':1, 'A':1, 'B':2},
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "SPM") for d in dicts ]
    global spm_orderings
    spm_orderings = tuple(orders)
    assert(len(spm_orderings) == 1)

    dicts = [
        {'C':20, 'A':1, 'B':1},
        {'C':1, 'A':2, 'B':1},
        {'C':1, 'A':1, 'B':250}
    ]
    orders = [ layer.get_ordering_from_reuse_factor(d, "DRAM") for d in dicts ]
    global dram_orderings
    dram_orderings = tuple(orders)
    assert(len(dram_orderings) == 3)

def get_sample_layer():
    env = expr_parameters.Environment()
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

def test_commonFunctions():
    layer = get_sample_layer()

    # Set tests to find data allocation in RF and SPM
    data_RF = layer.get_data_allocated_RF()
    assert data_RF["A"] == 50
    assert data_RF["B"] == 200
    assert data_RF["C"] == 4

    data_PE_array = layer.get_data_allocated_PE_array()
    assert data_PE_array["A"] == 100
    assert data_PE_array["B"] == 25000
    assert data_PE_array["C"] == 1000

    data_SPM = layer.get_data_allocated_SPM()
    assert data_SPM["A"] == 200
    assert data_SPM["B"] == 25000
    assert data_SPM["C"] == 2000


def test_TotalEnergy():
    layer = get_sample_layer()

    energy_ops = layer.computeEnergyCost_UsefulOps()
    assert abs(energy_ops - 75e+06) < 1.0e-5

    energy_RF = layer.computeEnergyCost_RF()
    assert abs(energy_RF - 3840e+06) < 1.0e-5

    key_1_spm = spm_orderings[0]
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    energy_SPM_pass = layer.get_Energy_One_SPM_Pass()

    assert abs(energy_SPM_pass[(key_1_dram, key_1_spm)] - 552200) < 1
    assert abs(energy_SPM_pass[(key_2_dram, key_1_spm)] - 552200) < 1
    assert abs(energy_SPM_pass[(key_3_dram, key_1_spm)] - 114700) < 1

    total_SPM_pass = layer.get_TripCounts_all_IVs("DRAM")
    assert abs(total_SPM_pass - 10000) == 0

    energy_DRAM = layer.get_Energy_DataCommunication_DRAM()
    assert abs(energy_DRAM[(key_1_dram, key_1_spm)] - 5.06e+10) < 1
    assert abs(energy_DRAM[(key_2_dram, key_1_spm)] - 5.8e+10) < 1
    assert abs(energy_DRAM[(key_3_dram, key_1_spm)] - (8.4e+09 + 1.75e+07)) < 1

    energy = layer.get_Energy_One_Layer()
    for d in dram_orderings:
        for s in spm_orderings:
            expected_energy = round(energy_ops + energy_RF + (energy_SPM_pass[(d,s)] * total_SPM_pass) + energy_DRAM[(d,s)])
            assert abs(energy[(d,s)] - expected_energy) == 0


def test_Performance():
    layer = get_sample_layer()

    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 200

    cycles = layer.calculate_Cycles_DataDistribution_NOC()
    assert cycles[layer._A.name] == 100
    assert cycles[layer._B.name] == 25000
    assert cycles[layer._C.name] == 1000

    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._A.name] == 0
    assert cycles[layer._B.name] == 0
    assert cycles[layer._C.name] == 0

    key_1_spm = spm_orderings[0]
    key_1_dram, key_2_dram, key_3_dram = dram_orderings

    # Check cycles for 1 RF pass
    comm_cycles_dict = layer.map_operands_to_NOCs()
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][0][1] == 1000
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][0][1] == 1000
    assert comm_cycles_dict[(key_3_dram, key_1_spm)][0][1] == 1000

    # Check stall cycles for each RF pass
    assert comm_cycles_dict[(key_1_dram, key_1_spm)][1][1] == 200
    assert comm_cycles_dict[(key_2_dram, key_1_spm)][1][1] == 200
    assert comm_cycles_dict[(key_3_dram, key_1_spm)][1][1] == 200

    # Check cycles for each SPM pass
    dict_cycles_spm_pass = layer.get_Cycles_One_SPM_Pass()
    assert dict_cycles_spm_pass[(key_1_dram, key_1_spm)] == 26400
    assert dict_cycles_spm_pass[(key_2_dram, key_1_spm)] == 26400
    assert dict_cycles_spm_pass[(key_3_dram, key_1_spm)] == 2400

    # Check DMA communication cycles
    dma_accesses = layer.determine_DMA_Access()
    assert dma_accesses["A"]["dma_invocations"] == 4
    assert dma_accesses["A"]["data_accessed_dma"] == 50
    assert dma_accesses["B"]["dma_invocations"] == 50
    assert dma_accesses["B"]["data_accessed_dma"] == 500
    assert dma_accesses["C"]["dma_invocations"] == 4
    assert dma_accesses["C"]["data_accessed_dma"] == 500

    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["A"] == 200
    assert dma_cycles["B"] == 4150
    assert dma_cycles["C"] == 332

    dict_cycles = layer.get_Cycles_One_Layer()
    assert dict_cycles[(key_1_dram, key_1_spm)] == 26400*20*2*250
    assert dict_cycles[(key_2_dram, key_1_spm)] == 26400*20*2*250
    assert dict_cycles[(key_3_dram, key_1_spm)] == (2400*250 + 26400 + 5014 - 2*2400)*2*20

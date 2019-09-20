from test_common import get_sample_layer

spm_ordering = ()
dram_ordering = ()

def setup():
    layer = get_sample_layer()
    dict_1 = {"O": 16, "I": 1, "W": 1}
    dict_2 = {"O": 8, "I": 1, "W": 1}
    global spm_ordering, dram_ordering
    spm_ordering = layer.get_ordering_from_reuse_factor(dict_1, "SPM")
    dram_ordering = layer.get_ordering_from_reuse_factor(dict_2, "DRAM")

def test_get_energy():
    layer = get_sample_layer()
    assert layer.get_energy(0) == 0.96
    assert layer.get_energy(4) == 0.075
    assert layer.get_energy(0) == layer.get_energy("RF")
    assert layer.get_energy(2) == layer.get_energy("SPM")

def test_computeEnergyCost_RF():
    layer = get_sample_layer()
    assert abs(layer.computeEnergyCost_RF()-1775700541.44) < 1.0e-5

def test_computeEnergyCost_UsefulOps():
    layer = get_sample_layer()
    assert abs(layer.computeEnergyCost_UsefulOps()-34681651.2) < 1.0e-5

def test_calculate_Energy_InterPECommunication():
    layer = get_sample_layer()
    energy_costs = layer.calculate_Energy_InterPECommunication()
    assert energy_costs[layer._I.name] == 0
    assert energy_costs[layer._W.name] == 0
    assert abs(energy_costs[layer._O.name] - 15538.88) < 1.0e-5

def test_computeEnergyCost_DataDistribution_NOC():
    layer = get_sample_layer()
    energy_costs = layer.get_Energy_DataDistribution_NOC()
    assert abs(energy_costs[layer._I.name] - 6272) == 0
    assert abs(energy_costs[layer._W.name] - 28224) == 0
    assert abs(energy_costs[layer._O.name] - 6272) == 0

def test_computeEnergyCost_DataDistribution_SPM():
    layer = get_sample_layer()
    energy_costs = layer.get_Energy_DataCommunication_SPM()
    assert abs(energy_costs[layer._I.name] - 6912) == 0
    assert abs(energy_costs[layer._W.name] - 3888) == 0
    assert abs(energy_costs[layer._O.name] - 42336) == 0

def test_computeEnergyCost_One_SPM_Pass():
    layer = get_sample_layer()
    energy_of_all_orderings = layer.get_Energy_One_SPM_Pass()
    energy_cost = energy_of_all_orderings[(dram_ordering, spm_ordering)]
    assert abs(energy_cost - 724736) == 0

def test_calculate_Energy_DMA_Access():
    layer = get_sample_layer()
    dma_access_energy = layer.calculate_Energy_DMA_Access()
    assert dma_access_energy["I"] == 1638400
    assert dma_access_energy["W"] == 921600
    assert dma_access_energy["O"] == 627200

def test_computeEnergyCost_DataCommunication_DRAM():
    layer = get_sample_layer()
    energy_of_all_orderings = layer.get_Energy_DataCommunication_DRAM()
    energy_cost = energy_of_all_orderings[(dram_ordering, spm_ordering)]
    expected_energy = 1310720000 + 40140800 + 994488.32 + 3110912
    assert abs(energy_cost - expected_energy) < 1.0e-5

def test_TotalEnergy():
    layer = get_sample_layer()
    energy_of_all_orderings = layer.get_Energy_One_Layer()
    energy = energy_of_all_orderings[(dram_ordering, spm_ordering)]
    expected_energy = round(34681651.2 + 1775700541.44 + (724736*512) + 1354966200.32)
    assert abs(energy - expected_energy) == 0

def test_get_min_energy():
    layer = get_sample_layer()
    min_energy, dram_ordering, spm_ordering = layer.get_min_energy()
    energy_of_all_orderings = layer.get_Energy_One_Layer()
    assert min_energy == 3365577523
    assert energy_of_all_orderings[(dram_ordering, spm_ordering)] == min_energy

def test_get_min_edp():
    layer = get_sample_layer()
    min_edp, dram_ordering, spm_ordering = layer.get_min_edp()
    assert min_edp == 1.62016386750976e+16

from test_common import get_sample_layer, get_ops
from nose.tools import assert_raises
import pprint

spm_ordering = ()
dram_ordering = ()

def setup():
    layer = get_sample_layer()
    dict_1 = {"O": 16, "I": 1, "W": 1}
    dict_2 = {"O": 8, "I": 1, "W": 1}
    global spm_ordering, dram_ordering
    spm_ordering = layer.get_ordering_from_reuse_factor(dict_1, "SPM")
    dram_ordering = layer.get_ordering_from_reuse_factor(dict_2, "DRAM")

def test_computeExecTime_UsefulOps_One_RF_Pass():
    layer = get_sample_layer()
    layer.env.pe_pipelined = True
    layer.env.pe_pipeline_stages = 4
    assert layer.computeExecTime_UsefulOps_One_RF_Pass() == 288

def test_calculate_Cycles_Datadistribution_NOC():
    layer = get_sample_layer()
    cycles = layer.calculate_Cycles_DataDistribution_NOC()
    assert len(cycles) == 3
    assert cycles[layer._O.name] == 3136
    assert cycles[layer._I.name] == 512
    assert cycles[layer._W.name] == 288

def test_determine_data_reuse():
    layer = get_sample_layer()

    with assert_raises(Exception):
        layer.determine_data_reuse("Spatial")

    data_reuse_with_all_orderings = layer.determine_data_reuse("SPM")
    loop_ordering = spm_ordering
    data_reuses, data_use_additional_comm = data_reuse_with_all_orderings[loop_ordering]
    assert data_reuses["I"] == 1
    assert data_reuses["W"] == 1
    assert data_reuses["O"] == 16
    assert data_use_additional_comm["I"] == 0
    assert data_use_additional_comm["W"] == 0
    assert data_use_additional_comm["O"] == 0

    data_reuse_with_all_orderings = layer.determine_data_reuse("DRAM")
    loop_ordering = dram_ordering
    data_reuses, data_use_additional_comm = data_reuse_with_all_orderings[loop_ordering]
    assert data_reuses["I"] == 1
    assert data_reuses["W"] == 1
    assert data_reuses["O"] == 8
    assert data_use_additional_comm["I"] == 0
    assert data_use_additional_comm["W"] == 0
    assert data_use_additional_comm["O"] == 0

def test_calculate_Cycles_InterPECommunication():
    layer = get_sample_layer()
    cycles = layer.calculate_Cycles_InterPECommunication()
    assert cycles[layer._I.name] == 0
    assert cycles[layer._W.name] == 0
    assert cycles[layer._O.name] == 3136

def test_map_operands_to_NOCs():
    layer = get_sample_layer()
    cycles_of_all_orderings = layer.map_operands_to_NOCs()
    comm_cycles = cycles_of_all_orderings[(dram_ordering, spm_ordering)][0]
    assert len(comm_cycles) == 1
    assert comm_cycles[1] == 512


def test_get_Cycles_One_SPM_Pass():
    layer = get_sample_layer()
    cycles_of_all_orderings = layer.get_Cycles_One_SPM_Pass()
    cycles = cycles_of_all_orderings[(dram_ordering, spm_ordering)]
    assert cycles  == 8192
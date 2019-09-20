from test_common import get_sample_layer, get_ops
from nose.tools import assert_raises

spm_ordering = ()
dram_ordering = ()

def setup():
    layer = get_sample_layer()
    dict_1 = {"O": 16, "I": 1, "W": 1}
    dict_2 = {"O": 8, "I": 1, "W": 1}
    global spm_ordering, dram_ordering
    spm_ordering = layer.get_ordering_from_reuse_factor(dict_1, "SPM")
    dram_ordering = layer.get_ordering_from_reuse_factor(dict_2, "DRAM")

def test_determine_DMA_Access():
    layer = get_sample_layer()
    #(1)
    result = layer.determine_DMA_Access()
    assert result["I"]["dma_invocations"] == 1
    assert result["I"]["data_accessed_dma"] == 8192
    assert result["W"]["dma_invocations"] == 16
    assert result["W"]["data_accessed_dma"] == 288
    assert result["O"]["dma_invocations"] == 1
    assert result["O"]["data_accessed_dma"] == 3136

    #(2)
    layer.set_tiling("N", [4, 1, 1, 1])
    layer.set_tiling("M", [16, 1, 8, 2])
    layer.set_tiling("C", [8, 16, 1, 2])
    layer.set_tiling("Ox", [1, 1, 2, 7])
    layer.set_tiling("Oy", [1, 1, 2, 7])
    layer.set_tiling("Fx", [3, 1, 1, 1])
    layer.set_tiling("Fy", [3, 1, 1, 1])
    result = layer.determine_DMA_Access()
    assert result["I"]["dma_invocations"] == 448
    assert result["I"]["data_accessed_dma"] == 14
    assert result["W"]["dma_invocations"] == 512
    assert result["W"]["data_accessed_dma"] == 1
    assert result["O"]["dma_invocations"] == 1
    assert result["O"]["data_accessed_dma"] == 3136

    #(3)
    layer.set_tiling("N", [4, 1, 1, 1])
    layer.set_tiling("M", [16, 1, 8, 2])
    layer.set_tiling("C", [8, 16, 1, 2])
    layer.set_tiling("Ox", [1, 1, 2, 7])
    layer.set_tiling("Oy", [1, 1, 2, 7])
    layer.set_tiling("Fx", [1, 1, 3, 1])
    layer.set_tiling("Fy", [3, 1, 1, 1])
    result = layer.determine_DMA_Access()
    assert result["I"]["dma_invocations"] == 512
    assert result["I"]["data_accessed_dma"] == 14
    assert result["W"]["dma_invocations"] == 1536
    assert result["W"]["data_accessed_dma"] == 1
    assert result["O"]["dma_invocations"] == 1
    assert result["O"]["data_accessed_dma"] == 3136

    #(4)
    layer.set_tiling("N", [4, 1, 1, 1])
    layer.set_tiling("M", [16, 1, 8, 2])
    layer.set_tiling("C", [8, 16, 1, 2])
    layer.set_tiling("Ox", [1, 1, 2, 7])
    layer.set_tiling("Oy", [1, 1, 2, 7])
    layer.set_tiling("Fx", [3, 1, 1, 1])
    layer.set_tiling("Fy", [1, 1, 3, 1])
    result = layer.determine_DMA_Access()
    assert result["I"]["dma_invocations"] == 32
    assert result["I"]["data_accessed_dma"] == 224
    assert result["W"]["dma_invocations"] == 512
    assert result["W"]["data_accessed_dma"] == 3
    assert result["O"]["dma_invocations"] == 1
    assert result["O"]["data_accessed_dma"] == 3136

    #(5)
    layer.set_tiling("N", [4, 1, 1, 1])
    layer.set_tiling("M", [16, 1, 8, 2])
    layer.set_tiling("C", [8, 16, 1, 2])
    layer.set_tiling("Ox", [1, 1, 2, 7])
    layer.set_tiling("Oy", [2, 1, 1, 7])
    layer.set_tiling("Fx", [1, 1, 3, 1])
    layer.set_tiling("Fy", [1, 1, 3, 1])
    result = layer.determine_DMA_Access()
    assert result["I"]["dma_invocations"] == 512
    assert result["I"]["data_accessed_dma"] == 9
    assert result["W"]["dma_invocations"] == 16
    assert result["W"]["data_accessed_dma"] == 288
    assert result["O"]["dma_invocations"] == 224
    assert result["O"]["data_accessed_dma"] == 7

    #(6)
    layer.set_tiling("N", [4, 1, 1, 1])
    layer.set_tiling("M", [16, 1, 8, 2])
    layer.set_tiling("C", [8, 16, 1, 2])
    layer.set_tiling("Ox", [2, 1, 1, 7])
    layer.set_tiling("Oy", [1, 1, 2, 7])
    layer.set_tiling("Fx", [1, 1, 3, 1])
    layer.set_tiling("Fy", [1, 1, 3, 1])
    result = layer.determine_DMA_Access()
    assert result["I"]["dma_invocations"] == 32
    assert result["I"]["data_accessed_dma"] == 144
    assert result["W"]["dma_invocations"] == 16
    assert result["W"]["data_accessed_dma"] == 288
    assert result["O"]["dma_invocations"] == 16
    assert result["O"]["data_accessed_dma"] == 98


def test_calculate_DMA_Cycles():
    layer = get_sample_layer()
    assert layer.calculate_DMA_Cycles(16384) == 660
    assert layer.calculate_DMA_Cycles(6272) == 281


def test_calculate_Cycles_DMA_Access():
    layer = get_sample_layer()
    dma_cycles = layer.calculate_Cycles_DMA_Access()
    assert dma_cycles["I"] == 660
    assert dma_cycles["W"] == 1088
    assert dma_cycles["O"] == 281

def test_get_Cycles_One_Layer():
    layer = get_sample_layer()
    cycles_of_all_orderings = layer.get_Cycles_One_Layer()
    cycle = cycles_of_all_orderings[(dram_ordering, spm_ordering)]
    assert cycle == 4581376

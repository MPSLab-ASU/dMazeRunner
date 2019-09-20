import tvm
from dMazeRunner.dataflow import ConvLayer
from dMazeRunner.common import expr_parameters
import collections

spm_ordering = ()
dram_ordering = ()

def setup():
    layer = get_sample_layer()
    dict_1 = {"O": 16, "I": 1, "W": 1}
    dict_2 = {"O": 8, "I": 1, "W": 1}
    global spm_ordering, dram_ordering
    spm_ordering = layer.get_ordering_from_reuse_factor(dict_1, "SPM")
    dram_ordering = layer.get_ordering_from_reuse_factor(dict_2, "DRAM")


def get_sample_layer():
    env = expr_parameters.Environment()
    args = {
        "name": "test_conv2d12_from_resnet18_v1",
        "channels": 256,
        "kernel_size": "[3, 3]",
        "padding": "[1, 1]",
        "strides": "[1, 1]",
        "output_shape": [1, 256, 14, 14],
        "input_shape": [1, 256, 14, 14],
        "batch_size": 4,
    }
    layer = ConvLayer(env, batch=4, **args)
    layer.set_tiling("N", [4, 1, 1, 1])
    layer.set_tiling("M", [16, 1, 8, 2])
    layer.set_tiling("C", [8, 16, 1, 2])
    layer.set_tiling("Ox", [1, 1, 2, 7])
    layer.set_tiling("Oy", [1, 1, 2, 7])
    layer.set_tiling("Fx", [1, 1, 3, 1])
    layer.set_tiling("Fy", [1, 1, 3, 1])

    """
    #the result should be same with the default loop order (N, M, C, Ox, Oy, Fx, Fy)
    for level in layer._loop_IVs:
        #layer.set_ordering(level, ["Oy", "Ox", "M", "N", "C", "Fx", "Fy"])
        layer.set_ordering(level, ["Fx", "Fy", "Oy", "Ox", "M", "N", "C"])
    #layer.set_ordering("DRAM", ["Oy", "Ox", "M", "N", "C", "Fx", "Fy"])
    layer.set_ordering("DRAM", ["Fx", "Fy", "Oy", "Ox", "M", "N", "C"])
    """

    return layer


def get_ops(layer):
    reads, writes = layer._get_reads_writes()
    O_read, I, W = tuple(reads)
    O_write = writes[0]
    return O_write, O_read, I, W

def test_generate_loop_orderings():
    layer = get_sample_layer()
    orderings = layer.generate_loop_orderings()
    assert len(orderings) == 15

    desired_orderings = [['Fy', 'Fx', 'C', 'N', 'M', 'Ox', 'Oy'],
        ['Fy', 'Fx', 'N', 'M', 'Ox', 'Oy', 'C'],
        ['Fy', 'C', 'N', 'M', 'Ox', 'Oy', 'Fx'],
        ['Fy', 'N', 'M', 'Ox', 'Oy', 'Fx', 'C'],
        ['Fx', 'C', 'N', 'M', 'Ox', 'Oy', 'Fy'],
        ['Fx', 'N', 'M', 'Ox', 'Oy', 'C', 'Fy'],
        ['C', 'N', 'M', 'Ox', 'Oy', 'Fx', 'Fy'],
        ['M', 'N', 'C', 'Ox', 'Oy', 'Fx', 'Fy'],
        ['Oy', 'Ox', 'N', 'M', 'C', 'Fx', 'Fy'],
        ['Oy', 'Ox', 'M', 'C', 'Fx', 'Fy', 'N'],
        ['Oy', 'N', 'M', 'C', 'Fx', 'Fy', 'Ox'],
        ['Oy', 'M', 'C', 'Fx', 'Fy', 'Ox', 'N'],
        ['Ox', 'N', 'M', 'C', 'Fx', 'Fy', 'Oy'],
        ['Ox', 'M', 'C', 'Fx', 'Fy', 'Oy', 'N'],
        ['N', 'M', 'C', 'Fx', 'Fy', 'Oy', 'Ox']]

    assert len(desired_orderings) == 15

    for ordering in desired_orderings:
        ordering_tuple = tuple(reversed(ordering))
        #assert ordering_tuple in orderings

    """
    isAllUniqueOrdersGenerated = (sorted(orderings) == sorted(desired_orderings_tuple))
    # TODO: Check that generated orderings belong to desired orderings
    assert(isAllUniqueOrdersGenerated == 1)
    """


def test_get_ordering_from_index():
    layer = get_sample_layer()
    #ordering = ('Fy', 'Oy', 'Ox', 'M', 'N', 'C', 'Fx')
    #assert (layer.get_index_from_ordering(ordering) == 4)
    #assert layer.get_ordering_from_index(4) == ordering

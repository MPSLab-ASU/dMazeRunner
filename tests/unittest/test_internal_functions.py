from test_common import get_sample_layer, get_ops
from nose.tools import assert_raises
import tvm

def test_set_tiling_wrong_inputs():
    layer = get_sample_layer()
    with assert_raises(Exception):
        # wrong iv name
        layer.set_tiling("n", [4, 1, 1, 1])

    with assert_raises(Exception):
        # wrong tiling length
        layer.set_tiling("N", [4, 1, 1, 1, 1])

    with assert_raises(Exception):
        # wrong tiling value
        layer.set_tiling("N", [4, 2, 1, 1])

    # correct case
    layer.set_tiling("N", [4, 1, 1, 1])


def test_set_tiling():
    layer = get_sample_layer()
    assert layer._loop_TCs["N_DRAM"] == 4
    assert layer._loop_TCs["N_SPM"] == 1
    assert layer._loop_TCs["N_RF"] == 1
    assert layer._loop_TCs["N_Spatial"] == 1

    layer.set_tiling("N", [1, 1, 2, 2])
    assert layer._loop_TCs["N_DRAM"] == 1
    assert layer._loop_TCs["N_SPM"] == 1
    assert layer._loop_TCs["N_RF"] == 2
    assert layer._loop_TCs["N_Spatial"] == 2


def test_set_ordering():
    layer = get_sample_layer()
    new_order = ["M", "C", "Ox", "N", "Oy", "Fx", "Fy"]
    layer.set_ordering("DRAM", new_order)
    assert layer._loop_IVs["DRAM"] == [ x+"_DRAM" for x in new_order ]


def test_get_loop():
    layer = get_sample_layer()
    loop = layer._get_loop()


def test_get_stores():
    layer = get_sample_layer()
    stores = layer._get_stores()
    assert len(stores) == 1
    store = stores[0]
    assert isinstance(store, tvm.stmt.Store)

    stores = layer._get_stores(pass_init=False)
    assert len(stores) == 2
    assert store in stores
    for store in stores:
        assert isinstance(store, tvm.stmt.Store)


def test_get_reads_writes():
    layer = get_sample_layer()
    reads, writes = layer._get_reads_writes()
    assert len(reads) == 3
    assert len(writes) == 1
    for read in reads:
        assert isinstance(read, tvm.expr.Load)
    for write in writes:
        assert isinstance(write, tvm.stmt.Store)


def test_get_reads_writes_of_operand():
    layer = get_sample_layer()
    reads, writes = layer._get_reads_writes_of_operand(layer._O.name)
    assert len(reads) == 1 and len(writes) == 1
    reads, writes = layer._get_reads_writes_of_operand(layer._W.name)
    assert len(reads) == 1 and len(writes) == 0
    reads, writes = layer._get_reads_writes_of_operand(layer._I.name)
    assert len(reads) == 1 and len(writes) == 0


def test_get_operands():
    layer = get_sample_layer()
    O_write, O_read, I, W = get_ops(layer)
    operands = layer._get_operands()

    assert O_write in operands[layer._O.name]
    assert O_read in operands[layer._O.name]
    assert I in operands[layer._I.name]
    assert W in operands[layer._W.name]


def test_get_num_different_pixels():
    layer = get_sample_layer()
    reads, writes = layer._get_reads_writes_of_operand(layer._I.name)
    assert layer._get_num_different_pixels(reads[0], [1, 2, 2, 14, 14, 3, 3]) == 512


def test_get_index_vars():
    layer = get_sample_layer()
    O_write, O_read, I, W = get_ops(layer)
    assert layer._get_index_vars(O_write) == ["N", "M", "Ox", "Oy"]
    assert layer._get_index_vars(O_read) == ["N", "M", "Ox", "Oy"]
    assert set(layer._get_index_vars(I)) == set(["N", "C", "Ox", "Oy", "Fx", "Fy"])
    assert layer._get_index_vars(W) == ["M", "C", "Fx", "Fy"]


def test_get_index_exprs():
    layer = get_sample_layer()
    O_write, O_read, I, W = get_ops(layer)
    keys = ["n", "m", "c", "ox", "oy", "fx", "fy"]
    values = [ int(layer.get_TripCounts(key.title(), "loop") / layer.get_TripCounts(key.title(), "DRAM")) for key in layer.base_TCs ]
    local_vars = dict(zip(keys, values))
    assert layer._get_index_expr_evaluated(I, 0, local_vars) == 16
    assert layer._get_index_expr_evaluated(I, 1, local_vars) == 16
    assert layer._get_index_expr_evaluated(I, 2, local_vars) == 32
    assert layer._get_index_expr_evaluated(I, 3, local_vars) == 1


def test_get_tensor_from_name():
    layer = get_sample_layer()
    assert layer._get_tensor_from_name("I") == layer._I
    assert layer._get_tensor_from_name("O") == layer._O
    assert layer._get_tensor_from_name("W") == layer._W
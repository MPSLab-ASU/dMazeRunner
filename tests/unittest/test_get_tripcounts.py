from test_common import get_sample_layer

def test_get_TripCounts():
    layer = get_sample_layer()
    assert layer.get_TripCounts("C", "loop") == 256
    assert layer.get_TripCounts("Ox", "Spatial") == 7

def test_get_TripCounts_all_IVs():
    layer = get_sample_layer()
    assert layer.get_TripCounts_all_IVs("loop") == 462422016
    assert layer.get_TripCounts_all_IVs("Spatial") == 196

"""
if __name__ == "__main__":
    test_get_TripCounts()
    test_get_TripCounts_all_IVs()
"""

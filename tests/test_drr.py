import numpy as np

from scrrpy.drr import DRR


def test_io():
    np.random.seed(1234)
    drr = DRR(0.1, j_grid_size=32)
    d, d_err = drr(2, neval=1e3, threads=1)
    drr.save("test.hdf5")
    drr = DRR.from_file("test.hdf5")
    d_load, d_load_err = drr(drr.l_max, neval=drr.neval, tol=drr.tol, progress_bar=False)
    np.testing.assert_array_max_ulp(d, d_load)
    np.testing.assert_array_max_ulp(d_err, d_load_err)


def test_drr_parallel():
    np.random.seed(1234)
    j_grid_size = 32
    drr = DRR(0.1, j_grid_size=j_grid_size)
    neval = 1e3
    l_max = 2
    d, d_err = drr(l_max, neval=neval, threads=4)
    print(d.mean())
    d_mean = d.mean()
    assert d.size == j_grid_size
    assert (d > 0).all(), d.min()
    assert drr.neval == neval
    assert drr.l_max == l_max
    np.testing.assert_almost_equal(d_mean*1e10, 1.755037479682267, 6)


def test_drr():
    np.random.seed(1234)
    j_grid_size = 32
    drr = DRR(0.1, j_grid_size=j_grid_size)
    neval = 1e3
    l_max = 2
    d, d_err = drr(l_max, neval=neval, threads=1)
    print(d.mean())
    d_mean = d.mean()
    assert d.size == j_grid_size
    assert (d > 0).all(), d.min()
    assert drr.neval == neval
    assert drr.l_max == l_max
    np.testing.assert_almost_equal(d_mean*1e10, 1.7540880485435062, 6)

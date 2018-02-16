import numpy as np

from scrrpy.drr import DRR
from scrrpy.drr import ResInterp


def test_io():
    np.random.seed(1234)
    drr = DRR(0.1, j_grid_size=32)
    d, d_err = drr(2, neval=1e3, threads=1)
    drr.save("test.hdf5")
    drr = DRR.from_file("test.hdf5")
    d_load, d_load_err = drr(drr.l_max, neval=drr.neval, tol=drr.tol, progress_bar=False)
    np.testing.assert_array_almost_equal_nulp(d, d_load)
    np.testing.assert_array_almost_equal_nulp(d_err, d_load_err)


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
    np.testing.assert_almost_equal(d_mean*1e10, 1.7550221461604085, 6)

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
    np.testing.assert_almost_equal(d_mean*1e10, 1.7536283542080702, 6)


def test_res_int(tol=5e-2):
    drr = DRR(0.1)
    jlc = drr.jlc(drr.sma)
    j = np.logspace(np.log10(jlc), 0, 11)[:-1]
    omega = drr.nu_p(drr.sma, j)
    res_int = ResInterp(drr, omega)
    f = res_int(omega[5])
    af = np.logspace(-5,1, 100)
    jf = f(af)
    assert jf.max() < 1.0
    assert jf.min() >= 0
    assert abs(1 - drr.nu_p(af[jf>0], jf[jf>0])/omega[5]).max() < tol

def test_res_int_no_solution():
    drr = DRR(0.1)
    jlc = drr.jlc(drr.sma)
    j = np.logspace(np.log10(jlc), 0, 11)[:-1]
    res_int = ResInterp(drr, [-10])
    f = res_int(-10)
    assert f is None

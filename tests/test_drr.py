import numpy as np

from scrrpy.drr import DRR
from scrrpy.drr import ResInterp

SEED = 51806469

D_VALUES = np.array(
    [1.75752701e-13, 4.48444715e-13, 1.10896626e-12,
     2.69628144e-12, 6.49855385e-12, 1.53897978e-11,
     3.53079070e-11, 7.95060658e-11, 1.74356982e-10,
     3.78702632e-10, 7.62428563e-10, 1.22869756e-08,
     7.80248648e-09, 1.01290011e-08, 9.44420166e-09,
     6.57582445e-09])

D_ERRORS = np.array(
    [8.73116429e-16, 2.26774809e-15, 5.34871809e-15,
     1.24663177e-14, 2.65481855e-14, 5.80360553e-14,
     1.30039735e-13, 2.86358234e-13, 5.78693261e-13,
     1.51284370e-12, 2.23531062e-12, 4.91059215e-11,
     1.30024470e-11, 2.55114407e-11, 3.96599494e-11,
     4.63296323e-11])


def test_io():
    np.random.seed(1234)
    drr = DRR(0.1, j_grid_size=32)
    d, d_err = drr(2, neval=1000, threads=1)
    drr.save("test.hdf5")
    drr = DRR.from_file("test.hdf5")
    d_load, d_load_err = drr(drr.l_max, neval=drr.neval, progress_bar=False)
    np.testing.assert_array_almost_equal_nulp(d, d_load)
    np.testing.assert_array_almost_equal_nulp(d_err, d_load_err)


def test_drr_parallel(tol=0.1):
    j_grid_size = 16
    drr = DRR(0.01, j_grid_size=j_grid_size, seed=SEED)
    neval = 10000
    l_max = 3
    d, d_err = drr(l_max, neval=neval, threads=4)
    assert d.size == j_grid_size
    assert (d > 0).all(), d.min()
    assert abs(1 - d / D_VALUES).max() < tol
    assert abs(1 - d_err / D_ERRORS).max() < tol


def test_drr(tol=0.1):
    j_grid_size = 16
    drr = DRR(0.01, j_grid_size=j_grid_size, seed=SEED)
    neval = 10000
    l_max = 3
    d, d_err = drr(l_max, neval=neval, threads=1)
    assert d.size == j_grid_size
    assert (d > 0).all(), d.min()
    assert abs(1 - d / D_VALUES).max() < tol
    assert abs(1 - d_err / D_ERRORS).max() < tol


def test_res_int(tol=5e-2):
    drr = DRR(0.1)
    jlc = drr.jlc(drr.sma)
    j = np.logspace(np.log10(jlc), 0, 11)[:-1]
    omega = drr.nu_p(drr.sma, j)
    res_int = ResInterp(drr, omega)
    f = res_int(omega[5])
    af = np.logspace(-5, 1, 100)
    jf = f(af)
    assert jf.max() < 1.0
    assert jf.min() >= 0
    assert abs(1 - drr.nu_p(af[jf > 0], jf[jf > 0]) / omega[5]).max() < tol


def test_res_int_no_solution():
    drr = DRR(0.1)
    res_int = ResInterp(drr, [-10])
    f = res_int(-10)
    assert f is None


def test_seeds():
    drr1 = DRR(0.1, j_grid_size=32)
    d1, d1_err = drr1(3)
    drr2 = DRR(0.1, j_grid_size=32, seed=drr1.seed)
    d2, d2_err = drr2(3, threads=4)
    np.testing.assert_array_almost_equal_nulp(d1, d2)
    np.testing.assert_array_almost_equal_nulp(d1_err, d2_err)

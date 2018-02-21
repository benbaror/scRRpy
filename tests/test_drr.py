import numpy as np

from scrrpy.drr import DRR
from scrrpy.drr import ResInterp

SEED = 51806469
D_VALUES = np.array(
    [1.50897683e-13, 2.57756608e-13, 3.81189736e-13, 5.67561719e-13,
     1.10040103e-12, 1.65432712e-12, 2.65663167e-12, 4.01980115e-12,
     6.42012326e-12, 9.64329795e-12, 1.52952290e-11, 2.17079524e-11,
     3.49644532e-11, 5.18927891e-11, 7.89822410e-11, 1.19442397e-10,
     1.75480178e-10, 2.46134140e-10, 3.89342714e-10, 5.35015375e-10,
     7.32471368e-10, 9.10132017e-09, 1.19507800e-08, 1.16408858e-08,
     7.64564747e-09, 7.91802921e-09, 1.00172340e-08, 8.95923264e-09,
     9.51971588e-09, 8.16075956e-09, 6.63161761e-09, 3.45813831e-09])

D_ERRORS = np.array(
    [4.08517254e-15, 5.59792158e-15, 7.67274951e-15, 9.26829134e-15,
     2.16663603e-14, 3.26153629e-14, 4.94142257e-14, 5.93086192e-14,
     1.07907139e-13, 1.34376631e-13, 2.30649389e-13, 2.87999699e-13,
     4.46081901e-13, 7.39000537e-13, 1.06535335e-12, 1.50364829e-12,
     1.94782491e-12, 2.79802558e-12, 5.37676407e-12, 6.90495986e-12,
     7.81492049e-12, 2.66302478e-10, 1.75609960e-10, 1.08093145e-10,
     4.57050360e-11, 5.65354444e-11, 8.90268064e-11, 9.86369762e-11,
     1.51850226e-10, 1.73300996e-10, 1.89504446e-10, 1.58843262e-10])


def test_io():
    np.random.seed(1234)
    drr = DRR(0.1, j_grid_size=32)
    d, d_err = drr(2, neval=1000, threads=1)
    drr.save("test.hdf5")
    drr = DRR.from_file("test.hdf5")
    d_load, d_load_err = drr(drr.l_max, neval=drr.neval, progress_bar=False)
    np.testing.assert_array_almost_equal_nulp(d, d_load)
    np.testing.assert_array_almost_equal_nulp(d_err, d_load_err)


def test_drr_parallel(tol=0.01):
    j_grid_size = 32
    drr = DRR(0.01, j_grid_size=j_grid_size, seed=SEED)
    neval = 1000
    l_max = 3
    d, d_err = drr(l_max, neval=neval, threads=4)
    assert d.size == j_grid_size
    assert (d > 0).all(), d.min()
    assert abs(1 - d / D_VALUES).max() < tol
    assert abs(1 - d_err / D_ERRORS).max() < tol


def test_drr(tol=0.01):
    j_grid_size = 32
    drr = DRR(0.01, j_grid_size=j_grid_size, seed=SEED)
    neval = 1000
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

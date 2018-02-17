import numpy as np

from scrrpy.drr import DRR
from scrrpy.drr import ResInterp

SEED = 51806469
D_VALUES = np.array([1.39265032e-13, 2.49013916e-13, 3.77645313e-13,
                     6.20497017e-13, 1.07377960e-12, 1.64955579e-12,
                     2.60571728e-12, 4.16908003e-12, 6.30972762e-12,
                     9.82491605e-12, 1.50096702e-11, 2.29475431e-11,
                     3.56462558e-11, 5.24008475e-11, 7.90938647e-11,
                     1.18020313e-10, 1.73932742e-10, 2.51332689e-10,
                     3.84368693e-10, 5.46209741e-10, 7.42423844e-10,
                     9.23045750e-09, 1.22223252e-08, 1.16254839e-08,
                     7.66519619e-09, 7.96051597e-09, 1.00282001e-08,
                     8.88728624e-09, 9.49657505e-09, 8.59321545e-09,
                     6.53196814e-09, 3.40134726e-09])

D_ERRORS = np.array([4.73975074e-15, 4.67986537e-15, 1.07368120e-14,
                     1.31026299e-14, 2.26012355e-14, 3.41891000e-14,
                     4.47063123e-14, 7.50575648e-14, 9.81317087e-14,
                     1.48547571e-13, 2.20694427e-13, 3.63450815e-13,
                     4.84654622e-13, 7.43737866e-13, 1.04351678e-12,
                     1.48741171e-12, 2.24575718e-12, 2.97986359e-12,
                     5.06191633e-12, 7.34072802e-12, 7.57853760e-12,
                     2.69056996e-10, 1.92206614e-10, 1.03251830e-10,
                     4.39041735e-11, 5.38682737e-11, 8.83398708e-11,
                     1.08804604e-10, 1.42786314e-10, 1.70700964e-10,
                     1.72207725e-10, 1.62980171e-10])


def test_io():
    np.random.seed(1234)
    drr = DRR(0.1, j_grid_size=32)
    d, d_err = drr(2, neval=1e3, threads=1)
    drr.save("test.hdf5")
    drr = DRR.from_file("test.hdf5")
    d_load, d_load_err = drr(drr.l_max, neval=drr.neval, progress_bar=False)
    np.testing.assert_array_almost_equal_nulp(d, d_load)
    np.testing.assert_array_almost_equal_nulp(d_err, d_load_err)


def test_drr_parallel(tol=1e-8):
    j_grid_size = 32
    drr = DRR(0.01, j_grid_size=j_grid_size, seed=SEED)
    neval = 1e3
    l_max = 3
    d, d_err = drr(l_max, neval=neval, threads=4)
    assert d.size == j_grid_size
    assert (d > 0).all(), d.min()
    assert abs(1-d / D_VALUES).max() < tol
    assert abs(1-d_err / D_ERRORS).max() < tol


def test_drr(tol=1e-8):
    j_grid_size = 32
    drr = DRR(0.01, j_grid_size=j_grid_size, seed=SEED)
    neval = 1e3
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
    jlc = drr.jlc(drr.sma)
    j = np.logspace(np.log10(jlc), 0, 11)[:-1]
    res_int = ResInterp(drr, [-10])
    f = res_int(-10)
    assert f is None


def test_seeds():
    drr1 = DRR(0.1, j_grid_size=32)
    d1, d1_err = drr1(3)
    drr2 = DRR(0.1, j_grid_size=32, seed=drr1.seed)
    d2, d2_err = drr1(3, threads=4)
    np.testing.assert_array_almost_equal_nulp(d1, d2)
    np.testing.assert_array_almost_equal_nulp(d1_err, d2_err)

import numpy as np

from scrrpy.drr import DRR


def test_drr_parallel():
    np.random.seed(1234)
    drr = DRR(0.1)
    neval = 1e3
    l_max = 2
    d, d_err = drr(l_max, neval=neval, threads=4)
    print(d.mean())
    d_mean = d.mean()
    assert (d > 0).all(), d.min()
    assert drr.neval == neval
    assert drr.l_max == l_max
    np.testing.assert_almost_equal(d_mean*1e10, 1.757449818387992, 6)


def test_drr():
    np.random.seed(1234)
    drr = DRR(0.1)
    neval = 1e3
    l_max = 2
    d, d_err = drr(l_max, neval=neval, threads=1)
    print(d.mean())
    d_mean = d.mean()
    assert (d > 0).all(), d.min()
    assert drr.neval == neval
    assert drr.l_max == l_max
    np.testing.assert_almost_equal(d_mean*1e10, 1.7607861026986751, 6)

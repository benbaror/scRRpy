import numpy as np

from scrrpy.drr import DRR

# def test_main():
#     runner = CliRunner()
#     result = runner.invoke(main, [])

#    assert result.output == '()\n'
#    assert result.exit_code == 0


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


def test_g(tol=1e-3):
    j = np.logspace(-2,0,1000)[:-1]

    #gamma = 2
    drr = DRR(0.1, gamma=2)
    g = drr._g(j)
    g2 = j/(1+j)
    assert abs(1 - g/g2).max() < tol

    #gamma = 1
    drr = DRR(0.1, gamma=1)
    g = drr._g(j)
    g2 = j
    assert abs(1 - g/g2).max() < tol

    #gamma = 1.75
    drr = DRR(0.1, gamma=1.75)
    g = drr._g(0.999999)
    g2 = (3 - drr.gamma)/2
    assert abs(1 - g/g2) < tol


def test_gp(tol=1e-3):
    j = np.logspace(-2,0,1000)[:-1]

    #gamma = 2
    drr = DRR(0.1, gamma=2)
    gp = drr._gp(j)
    gp2 = 1/(1+j)**2
    assert abs(1 - gp/gp2).max() < tol

    #gamma = 1
    drr = DRR(0.1, gamma=1)
    gp = drr._gp(j)
    gp2 = 1.0
    assert abs(1 - gp/gp2).max() < tol

import numpy as np

from scrrpy import Cusp


def test_g(tol=1e-3):
    j = np.logspace(-2, 0, 1000)[:-1]

    # gamma = 2
    cusp = Cusp(gamma=2)
    g = cusp._g(j)
    g2 = j / (1 + j)
    assert abs(1 - g / g2).max() < tol

    # gamma = 1
    cusp = Cusp(gamma=1)
    g = cusp._g(j)
    g2 = j
    assert abs(1 - g / g2).max() < tol

    # gamma = 1.75
    cusp = Cusp(gamma=1.75)
    g = cusp._g(0.999999)
    g2 = (3 - cusp.gamma) / 2
    assert abs(1 - g / g2) < tol


def test_gp(tol=1e-3):
    j = np.logspace(-2, 0, 1000)[:-1]

    # gamma = 2
    cusp = Cusp(gamma=2)
    gp = cusp._gp(j)
    gp2 = 1 / (1 + j) ** 2
    assert abs(1 - gp / gp2).max() < tol

    # gamma = 1
    cusp = Cusp(gamma=1)
    gp = cusp._gp(j)
    gp2 = 1.0
    assert abs(1 - gp / gp2).max() < tol


def test_period(tol=0.4):
    # S2 period
    cusp = Cusp(mbh_mass=4.35e6)
    arcsec_kpc_pc = 0.0048481368
    r0 = 8.33
    sma_s2 = 0.1255 * arcsec_kpc_pc * r0  # pc
    p_s2 = 16.00  # yr
    period = cusp.period(sma_s2)
    nu_r = cusp.nu_r(sma_s2)
    np.testing.assert_almost_equal(nu_r * period, 2 * np.pi)
    assert abs(1 - period / p_s2) < tol

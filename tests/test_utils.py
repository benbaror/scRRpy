import numpy as np

from scrrpy.utils import interp_reg
from scrrpy.utils import interp_reg_loglog
from scrrpy.utils import interp_reg_semilogx
from scrrpy.utils import interp_reg_semilogx_vec


def test_interp_reg(tol=1e-4):
    x = np.linspace(0, 1, 100)
    x_intp = np.random.rand(1000)
    y = x**2
    f_intp = interp_reg(x, y)(x_intp)
    err = max(abs(x_intp**2 - f_intp))
    assert err < tol, err

def test_interp_reg_semilogx(tol=1e-3):
    x = np.logspace(-5, 5, 1000)
    x_intp = 10**((1 - 2*np.random.rand(1000))*5)
    y = np.log(x)**2
    f_intp = interp_reg_semilogx(x_intp, y, x[0], x[1])
    err = max(abs(np.log(x_intp)**2 - f_intp))
    assert err < tol, err

def test_interp_reg_semilogx_vec(tol=1e-3):
    x = np.logspace(-5, 5, 1000)
    x_intp = 10**((1 - 2*np.random.rand(1000))*5)
    y = np.log(x)**2
    f_intp = interp_reg_semilogx_vec(x, y)(x_intp)
    err = max(abs(np.log(x_intp)**2 - f_intp))
    assert err < tol, err


def test_interp_reg_loglog(tol=1e-3):
    x = np.logspace(-5, 5, 1000)
    x_intp = 10**((1 - 2*np.random.rand(1000))*5)
    y = x**2
    f_intp = interp_reg_loglog(x, y)(x_intp)
    err = max(abs(x_intp**2 - f_intp))
    assert err < tol, err

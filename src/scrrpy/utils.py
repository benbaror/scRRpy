import numpy as np
from numba import float64
from numba import guvectorize
from numba import vectorize


def interp_reg(x, f):
    """
    Linear interpolation over a regular grid

    Parameters
    ----------
    x : array,
      assumed to be a linear regular grid
    f : array, func

    :return: an interpolation function
    """

    x0 = x[0]
    dx_inv = 1 / (x[1] - x0)

    @vectorize([float64(float64)], nopython=True)
    def _interp_reg(x_intp):
        x_dx = (x_intp - x0) * dx_inv
        ind = int(x_dx)
        w = x_dx - ind
        return f[ind] * (1 - w) + f[ind + 1] * w

    return _interp_reg


def interp_reg_loglog(x, f):
    """
    Linear interpolation over a logarithmic grid

    Parameters
    ----------
    x : array,
      assumed to be a linear regular grid
    f : array, func

    :return: an interpolation function
    """

    x0 = x[0]
    logx0 = np.log(x[0])
    dx_inv = 1 / np.log(x[1] / x0)
    logf = np.log(f)

    @vectorize([float64(float64)], nopython=True)
    def _interp_reg(x_intp):
        x_dx = (np.log(x_intp) - logx0) * dx_inv
        ind = int(x_dx)
        w = x_dx - ind
        return np.exp(logf[ind] * (1 - w) + logf[ind + 1] * w)

    return _interp_reg


def interp_reg_semilogx_vec(x, f, log=np.log):
    """
    Linear interpolation over a semi-logarithmic regular grid

    Parameters
    ----------
    x : array
      assumed to be a logarithmic regular grid
    f : array

    :return: an interpolation function
    """

    x0 = x[0]
    dx_inv = 1 / log(x[1] / x0)

    @vectorize([float64(float64)])
    def _interp_reg(x_intp):
        x_dx = log(x_intp / x0) * dx_inv
        ind = int(x_dx)
        w = x_dx - ind
        return f[ind] * (1 - w) + f[ind + 1] * w

    return _interp_reg


@guvectorize([(float64[:], float64[:], float64, float64, float64[:])],
             '(n),(m),(),()->(n)')
def interp_reg_semilogx(x_intp, f, x0, x1, res):
    """
    Linear interpolation over a semi-logarithmic regular grid

    """
    dx_inv = 1 / np.log(x1 / x0)
    for i in range(x_intp.size):
        x_dx = np.log(x_intp[i] / x0) * dx_inv
        ind = int(x_dx)
        w = x_dx - ind
        res[i] = f[ind] * (1 - w) + f[ind + 1] * w

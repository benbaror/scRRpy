"""
A module for calculating Resonant Relaxation diffusion coefficients
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
from ast import literal_eval as make_tuple
from builtins import range
from builtins import super
from builtins import zip
from functools import lru_cache

import h5py
import numpy as np
# noinspection PyPackageRequirements
import progressbar
import vegas
from numba import jit
from numpy import mod
from numpy import sqrt
from scipy import special

from .cusp import Cusp


class DRR(Cusp):
    """
    Resonant relaxation diffusion coefficient (DRR)
    Assuming a power law stellar cusp around a massive black hole (MBH).
    The cusp is assumed to have an isotropic distribution function
    :math:`f(E) \propto |E|^p` corresponding ro a stellar density
    :math:`n(r) \propto r^{-\gamma}` where :math:`\gamma = 3/2 + p`

    Parameters
    ----------
    sma : float
       The semi-mahor axis along which DRR will be computed

    gamma : float, int, optional
        The slope of the density profile.
        default: 7/4 (Bahcall wolf cusp)
    mbh_mass : float, int, optional
        Mass of the MBH [solar mass].
        default: 4.3 10^6 (Milky Way MBH)
    star_mass : float, int, optional
        Mass of individual stars [solar mass].
        default: 1.0
    rh : float, int, optional
        Radius of influence [pc].
        Define as the radius in which the velocity
        dispersion of the stellar cusp :math:`\sigma` is equal to the
        Keplerian velocity due to the MBH
        :math:`\sigma(r_h)^2 = G M_\bullet / r_h`.
        default: 2.0
    """

    def __init__(self, sma, gamma=1.75, mbh_mass=4e6, star_mass=1.0, j_grid_size=128, rh=2.0):

        super().__init__(gamma=gamma,
                         mbh_mass=mbh_mass,
                         star_mass=star_mass,
                         rh=rh)
        self.sma = sma
        self.gr_factor = 1.0
        self.j = np.logspace(np.log10(self.jlc(self.sma)), 0, j_grid_size + 1)[:-1]
        self.omega = abs(self.nu_p(self.sma, self.j))

    @lru_cache()
    def _res_intrp(self, ratio):
        return ResInterp(self, self.omega * ratio, self.gr_factor)

    def _integrand(self, j, sma_p, j_p, lnnp, true_anomaly):
        d_nu_p = abs(self.d_nu_p(sma_p, j_p))
        a2_int = a2_integrand(self.sma, j, sma_p, j_p, lnnp, true_anomaly)
        return 2 * j_p * a2_int / d_nu_p / lnnp[-1]

    def __call__(self, l_max, neval=1e3, threads=1, tol=0.0, progress_bar=True):
        """
        Returns the RR diffusion coefficient over Jc^2 in 1/yr.
        """

        # Get all non-vanishing resonances up to l=l_max
        lnnp = [(l, n, n_p)
                for l in range(1, l_max + 1)
                for n in range(1, l + 1)
                for n_p in range(1, l + 1)
                if not mod(l + n, 2) + mod(l + n_p, 2)
                ]

        if progress_bar:
            pbar = progressbar.ProgressBar()(range(len(lnnp)))
        else:
            pbar = range(len(lnnp))

        for i in pbar:
            self._drr_lnnp(*lnnp[i], neval=neval, threads=threads, tol=tol)

        drr = np.vstack([self._drr_lnnp(*lnnp, neval=neval, threads=threads,
                                        tol=tol)[0]
                         for lnnp in lnnp])

        # Remove non-physical values
        drr[drr < 0] = 0

        # sum all resonances
        drr = drr.sum(axis=0)

        drr_err = sqrt(sum(self._drr_lnnp(*lnnp, neval=neval,
                                          threads=threads,
                                          tol=tol)[-1] ** 2
                           for lnnp in lnnp))
        return drr, drr_err

    def _drr_lnnp(self, l, n, n_p, neval=1e3, threads=1,
                  tol=0.0):
        """
        Calculates the l,n,n_p term of the diffusion coefficient
        """
        neval = int(neval)
        try:
            drr = (self._drr_lnnp_cache[str((l, n, n_p, neval, tol))][0] +
                   self._drr_lnnp_cache[str((l, n, -n_p, neval, tol))][0])

            drr_err = sqrt(self._drr_lnnp_cache[str((l, n, n_p, neval,
                                                     tol))][-1] ** 2 +
                           self._drr_lnnp_cache[str((l, n, -n_p, neval,
                                                     tol))][-1] ** 2)
            return drr, drr_err
        except AttributeError:
            self._drr_lnnp_cache = {}
        except KeyError:
            pass

        if threads > 1:
            queue = mp.Queue()

            def parallel_drr(pos, seed, j, omega):
                np.random.seed(seed)
                _results = [self._drr(j, omega, (l, n, n_p), neval=neval,
                                      tol=tol)
                            for j, omega in zip(j, omega)]
                _drr = [result[0] for result in _results]
                _drr_err = [result[-1] for result in _results]
                queue.put((pos, (_drr, _drr_err)))

            js = [self.j[i::threads] for i in range(threads)]
            omegas = [self.omega[i::threads] for i in range(threads)]

            seeds = np.random.randint(100000, size=threads)
            processes = [mp.Process(target=parallel_drr, args=(i, seed, j, omega))
                         for i, (seed, j, omega) in enumerate(zip(seeds, js, omegas))]
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            _, results = zip(*sorted(map(queue.get, processes)))
            drr, drr_err = np.hstack(results)

        else:
            results = [self._drr(j, omega, (l, n, n_p), neval=neval, tol=tol)
                       for j, omega, i in zip(self.j, self.omega,
                                              range(self.j.size))]

            drr = np.array([result[0] for result in results])
            drr_err = np.array([result[-1] for result in results])

        self._drr_lnnp_cache[str((l, n, -n_p,
                                  neval, tol))] = (drr[:, 0], drr_err[:, 0])
        self._drr_lnnp_cache[str((l, n, n_p,
                                  neval, tol))] = (drr[:, -1], drr_err[:, -1])

        drr = (self._drr_lnnp_cache[str((l, n, n_p, neval, tol))][0] +
               self._drr_lnnp_cache[str((l, n, -n_p, neval, tol))][0])
        drr_err = sqrt(self._drr_lnnp_cache[str((l, n, n_p, neval,
                                                 tol))][-1] ** 2 +
                       self._drr_lnnp_cache[str((l, n, -n_p, neval,
                                                 tol))][-1] ** 2)
        return drr, drr_err

    def _drr(self, j, omega, lnnp, neval=1e3, tol=0.0):
        ratio = lnnp[1] / lnnp[-1]
        get_jf1 = self._res_intrp(ratio).get_jf1(omega * ratio)
        get_jf2 = self._res_intrp(ratio).get_jf2(omega * ratio)

        @vegas.batchintegrand
        def c_lnnp1(x):
            true_anomaly = x[:, :-1].T * np.pi
            sma_f = self.inverse_cumulative_a(x[:, -1])
            jf1 = get_jf1(sma_f)
            res = np.zeros(x.shape[0], float)
            ix1 = np.where(jf1 > 0)[0]
            if len(ix1) > 0:
                res[ix1] = self._integrand(j, sma_f[ix1], jf1[ix1], lnnp,
                                           true_anomaly[:, ix1])
            return res

        @vegas.batchintegrand
        def c_lnnp2(x):
            true_anomaly = x[:, :-1].T * np.pi
            sma_f = self.inverse_cumulative_a(x[:, -1])
            jf2 = get_jf2(sma_f)
            res = np.zeros(x.shape[0], float)
            ix2 = np.where(jf2 > 0)[0]
            if len(ix2) > 0:
                res[ix2] = self._integrand(j, sma_f[ix2], jf2[ix2], lnnp,
                                           true_anomaly[:, ix2])
            return res

        integ = vegas.Integrator(5 * [[0, 1]])

        if get_jf1 is None:
            int1, err1 = 0.0, 0.0
        else:
            int1, err1 = np.array(integrate(c_lnnp1, integ, neval, tol))

        if get_jf2 is None:
            int2, err2 = 0.0, 0.0
        else:
            int2, err2 = np.array(integrate(c_lnnp2, integ, neval, tol))

        return 4 * np.pi * (np.array([[int1, int2], [err1, err2]]) *
                            _a2_norm_factor(*lnnp) * lnnp[1] ** 2 *
                            self.nu_r(self.sma) ** 2 / self.mass_ratio ** 2 * self.total_number_of_stars)

    @property
    def l_max(self):
        try:
            return max([make_tuple(key)[0]
                        for key in self._drr_lnnp_cache.keys()])
        except AttributeError:
            pass

    @property
    def neval(self):
        try:
            return max([make_tuple(key)[-2]
                        for key in self._drr_lnnp_cache.keys()])
        except AttributeError:
            pass

    @property
    def tol(self):
        try:
            return min([make_tuple(key)[-1]
                        for key in self._drr_lnnp_cache.keys()])
        except AttributeError:
            pass

    def save(self, file_name):
        """
        Save the cached data to an hdf5 file so it can be read later.
        """
        with h5py.File(file_name, 'w') as h5:
            drr_lnnp_cache = h5.create_group("_drr_lnnp_cache")
            for key, value in self._drr_lnnp_cache.items():
                drr_lnnp_cache[key] = value
            for key, value in self.__dict__.items():
                try:
                    h5[key] = value
                except TypeError:
                    pass

    def _read(self, file_name):
        """
        Read the cached data from an hdf5 file
        """
        with h5py.File(file_name, 'r') as h5:

            for key, value in h5['_drr_lnnp_cache'].items():
                try:
                    self._drr_lnnp_cache[key] = value.value
                except AttributeError:
                    self._drr_lnnp_cache = {key: value.value}

            for key, value in h5.items():
                if key != '_drr_lnnp_cache':
                    setattr(self, key, value.value)

    @classmethod
    def from_file(cls, file_name):
        """Load from file and return an instance

        example:
        drr = DRR.from_file(file_name)
        """
        return cls(1.0)._read(file_name)


def integrate(func, integ, neval=1e4, tol=0.0):
    integ.set(rtol=tol)
    integ(func, nitn=10, neval=neval)
    result = integ(func, nitn=10, neval=neval)
    try:
        res, err = np.array([[r.val, np.sqrt(r.var)] for r in result]).T
    except TypeError:
        res, err = result.val, np.sqrt(result.var)
    return res, err


@jit(nopython=True)
def a2_integrand(sma, j, sma_p, j_p, lnnp, true_anomaly):
    """
    returns the |alnnp|^2 integrand to use in the MC integration
    """
    l, n, n_p = lnnp
    cnnp = np.cos(true_anomaly.T * np.array([n, n, n_p, n_p])).T
    cnnp = cnnp[0] * cnnp[1] * cnnp[2] * cnnp[3]
    j2 = j ** 2
    j_p2 = j_p ** 2
    ecc, eccp = np.sqrt(1 - j2), np.sqrt(1 - j_p2)
    r12 = (sma * j2 / (1 - ecc * np.cos(true_anomaly[:2])))
    rp12 = (sma_p * j_p2 / (1 - eccp * np.cos(true_anomaly[2:])))
    r_1, r_2 = r12[0], r12[-1]
    rp1, rp2 = rp12[0], rp12[-1]
    return (cnnp / j2 / j_p2 / sma ** 2 / sma_p ** 4 *
            (np.minimum(r_1, rp1) * np.minimum(r_2, rp2)) ** (2 * l + 1) /
            (r_1 * r_2 * rp1 * rp2) ** (l - 1))


@lru_cache()
def _a2_norm_factor(l, n, n_p):
    """
    Normalization factor for |alnnp|^2
    """

    return (abs(special.sph_harm(n, l, 0, np.pi / 2)) ** 2 *
            abs(special.sph_harm(n_p, l, 0, np.pi / 2)) ** 2 *
            (4 * np.pi / (2 * l + 1)) ** 2) / (2 * l + 1)


class ResInterp(object):
    """
    Interpolation function for the resonant condition
    """

    def __init__(self, cusp, omega, gr_factor=1.0):
        """
        """
        self._cusp = cusp
        self._cusp.gr_factor = gr_factor
        self.omega = omega
        self._af = np.logspace(np.log10(self._cusp.rg),
                               np.log10(self._cusp.rh),
                               1000)

        # self._jf = np.logspace(np.log10(self._cusp.jlc(self._cusp.rh)),
        #                                 0, 1001)[:-1]

        def get_j(_nup):
            jf = self._jf[_nup > 0]
            _nup = _nup[_nup > 0]
            s = np.argsort(_nup)
            j = np.interp(self.omega, _nup[s], jf[s], left=0, right=0)

            # j[self.omega < nup.min()] = 0
            # j[self.omega > nup.max()] = 0
            return j

        # The minimal a at which omega changes sign.
        a_gr1 = self._cusp.a_gr1
        # The minimal at which omega intersects nu_p
        self._af = np.logspace(np.log10(self._cusp.rg),
                               np.log10(self._cusp.rh),
                               1000)

        a_min = self._af[(self._af < a_gr1) *
                         (omega.max() < self._cusp.nu_p1(self._af))].max()
        self._af = np.logspace(np.log10(a_min),
                               np.log10(self._cusp.rh),
                               1000)

        self._j1 = np.zeros([self._af.size, self.omega.size])
        self._j2 = np.zeros([self._af.size, self.omega.size])

        last = 0
        for i, a in enumerate(self._af[self._af < a_gr1]):
            self._jf = np.logspace(np.log10(self._cusp.jlc(a)),
                                   0, 1001)[:-1]
            nup = self._cusp.nu_p(a, self._jf)
            self._j1[i, :] = get_j(nup)
            last += 1
        # last += 1

        for i, a in enumerate(self._af[self._af > a_gr1]):
            self._jf = np.logspace(np.log10(self._cusp.jlc(a)),
                                   0, 1001)[:-1]
            nup = self._cusp.nu_p(a, self._jf)
            self._j1[i + last, :] = get_j(nup)
            if any(nup < 0):
                self._j2[i + last, :] = get_j(-nup)

    def get_jf1(self, omega):
        i = np.argmin(abs(self.omega - omega))
        if abs(self.omega[i] - omega) > 1e-8:
            raise ValueError
        j = self._j1[:, i]
        ix = np.where(j > 0)[0]
        if len(ix) > 0:
            return lambda af: np.interp(af,
                                        self._af[ix], j[ix], left=0, right=0)

    def get_jf2(self, omega):
        i = np.argmin(abs(self.omega - omega))
        if abs(self.omega[i] - omega) > 1e-8:
            raise ValueError
        j = self._j2[:, i]
        ix = np.where(j > 0)[0]
        if len(ix) > 0:
            return lambda af: np.interp(af,
                                        self._af[ix], j[ix], left=0, right=0)

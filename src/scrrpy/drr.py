r"""
A module for calculating Resonant Relaxation diffusion coefficients
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
from ast import literal_eval as make_dict
from builtins import range
from builtins import super
from builtins import zip
from collections import namedtuple
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

from . import Cusp

Res = namedtuple('Res', ('l', 'n', 'np', 'neval'))


class DRR(Cusp):
    r"""
    Resonant relaxation diffusion coefficient (DRR).
    Assuming a power law stellar cusp around a massive black hole (MBH).
    The cusp is assumed to have an isotropic distribution function
    :math:`f(E) \propto |E|^p` corresponding ro a stellar density
    :math:`n(r) \propto r^{-\gamma}` where :math:`\gamma = \tfrac{3}{2} + p`

    Parameters
    ----------
    sma : float
       The semi-mahor axis along which DRR will be computed

    gamma : float, int, optional
        The slope of the density profile.
        Default: 7/4 (Bahcall wolf cusp)
    mbh_mass : float, int, optional
        Mass of the MBH [solar mass].
        Default: :math:`4.3 \times 10^6` (Milky Way MBH)
    star_mass : float, int, optional
        Mass of individual stars [solar mass].
        Default: 1.0
    rh : float, int, optional
        Radius of influence [pc].
        Define as the radius in which the velocity
        dispersion of the stellar cusp :math:`\sigma` is equal to the
        Keplerian velocity due to the MBH
        :math:`\sigma(r_\mathrm{h})^2 = G M_{\bullet} / r_\mathrm{h}`.
        Default: 2.0
    """

    def __init__(self, sma, gamma=1.75, mbh_mass=4e6, star_mass=1.0,
                 j_grid_size=128, rh=2.0, seed=None):

        super().__init__(gamma=gamma,
                         mbh_mass=mbh_mass,
                         star_mass=star_mass,
                         rh=rh)
        self.sma = sma
        self.gr_factor = 1.0
        self.j = np.logspace(np.log10(self.jlc(self.sma)), 0,
                             j_grid_size + 1)[:-1]
        self.omega = self.nu_p(self.sma, self.j)
        if seed is None:
            self.seed = np.random.randint(int(1e8))
        else:
            self.seed = seed

        np.random.seed(self.seed)

        self.seeds = np.random.randint(int(1e8), size=self.j.size)

    @lru_cache()
    def _res_intrp(self, ratio):
        return ResInterp(self, self.omega * ratio, self.gr_factor)

    def _integrand(self, j, sma_p, j_p, lnnp, true_anomaly):
        d_nu_p = abs(self.d_nu_p(sma_p, j_p))
        a2_int = a2_integrand(self.sma, j, sma_p, j_p, lnnp, true_anomaly)
        return j_p * a2_int / d_nu_p

    def __call__(self, l_max, neval=1e3, threads=1,
                 progress_bar=True, seed=None):
        r"""
        Returns the RR diffusion coefficient :math:`D_{JJ}/J_{\mathrm{c}}^2` [1/yr].

        Parameters
        ----------
        l_max : int
            Maximal order of spherical harmonics to compute
        neval : int
            The maximum number of integrand evaluations
            in each iteration of the `vegas` algorithm.
            Default: 1000
        threads : int
            Number of parallel threads to use.
            Default: 1 (no parallelization)
        progress_bar : bool
            Show progress bar.
            Default: ``True``
        """

        # Get all non-vanishing resonances up to l=l_max
        lnnp = [(l, n, n_p)
                for l in range(1, l_max + 1)
                for n in range(1, l + 1)
                for n_p in range(-l - 1, l + 1)
                if (not mod(l + n, 2) + mod(l + n_p, 2)) and n_p != 0
                ]

        if progress_bar:
            pbar = progressbar.ProgressBar()(range(len(lnnp)))
        else:
            pbar = range(len(lnnp))

        for i in pbar:
            self._drr_lnnp(*lnnp[i], neval=neval, threads=threads)

        drr = np.vstack([self._drr_lnnp(*lnnp, neval=neval, threads=threads)[0]
                         for lnnp in lnnp])

        # Remove non-physical values
        drr[drr < 0] = 0

        # sum all resonances
        drr = drr.sum(axis=0)

        drr_err = sqrt(sum(self._drr_lnnp(*lnnp, neval=neval,
                                          threads=threads)[-1] ** 2
                           for lnnp in lnnp))
        return drr, drr_err

    def _drr_lnnp(self, l, n, n_p, neval=1e3, threads=1):
        r"""
        Calculates the :math:`(l,n,n_p)` term of the diffusion coefficient
        """
        neval = int(neval)

        # Pre compute the resonance interpolation grid
        ratio = n / n_p
        self._res_intrp(ratio)
        try:
            drr, drr_err = self._drr_lnnp_cache[Res(l, n, n_p, neval)]

            return drr, drr_err
        except AttributeError:
            self._drr_lnnp_cache = {}
        except KeyError:
            pass

        if threads > 1:
            queue = mp.Queue()

            def parallel_drr(pos, j_s, omega_s, seed_s):
                _results = [
                    self._drr(j, omega, (l, n, n_p), neval=neval, seed=seed)
                    for j, omega, seed in zip(j_s, omega_s, seed_s)
                ]
                _drr = [result[0] for result in _results]
                _drr_err = [result[-1] for result in _results]
                queue.put((pos, (_drr, _drr_err)))

            js = [self.j[i::threads] for i in range(threads)]
            omegas = [self.omega[i::threads] for i in range(threads)]
            seeds = [self.seeds[i::threads] for i in range(threads)]

            processes = [
                mp.Process(target=parallel_drr, args=(i, j_s, omega_s, seed_s))
                for i, (j_s, omega_s, seed_s) in
                enumerate(zip(js, omegas, seeds))]
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            drr, drr_err = zip(*[(drr, drr_err)
                                 for (i, (drr, drr_err)) in
                                 sorted(map(queue.get, processes))])
            drr, drr_err = (np.concatenate(list(zip(*drr))),
                            np.concatenate(list(zip(*drr_err))))

        else:
            results = [self._drr(j, omega, (l, n, n_p), neval=neval, seed=seed)
                       for j, omega, seed in
                       zip(self.j, self.omega, self.seeds)]

            drr, drr_err = np.array(list(zip(*results)))

        self._drr_lnnp_cache[Res(l, n, n_p, neval)] = (drr, drr_err)

        return drr, drr_err

    def _drr(self, j, omega, lnnp, neval=1e3, seed=None):
        l, n, n_p = lnnp
        ratio = n / n_p
        get_jf = self._res_intrp(ratio)(omega * ratio)
        if seed is not None:
            # make a unique seed
            np.random.seed([seed, l, n, 2 * l + n_p])

        pi = np.pi

        @vegas.batchintegrand
        def c_lnnp(x):
            true_anomaly = x[:, :-1].T * pi
            sma_f = self.inverse_cumulative_a(x[:, -1])
            jf = get_jf(sma_f)
            res = np.zeros(x.shape[0], float)
            ix = np.where(jf > 0)[0]
            if len(ix) > 0:
                res[ix] = self._integrand(j, sma_f[ix], jf[ix], lnnp,
                                          true_anomaly[:, ix])
            return res

        self.c_lnnp = c_lnnp
        integ = vegas.Integrator(5 * [[0, 1]])

        if get_jf is None:
            result = np.zeros(2)
        else:
            result = np.array(integrate(c_lnnp, integ, neval))

        return result * (8 * pi * n ** 2 / abs(n_p) *
                         _a2_norm_factor(*lnnp) *
                         self.nu_r(self.sma) ** 2 / self.mass_ratio ** 2 *
                         self.total_number_of_stars)

    @property
    def l_max(self):
        try:
            return max([key.l
                        for key in self._drr_lnnp_cache.keys()])
        except AttributeError:
            pass

    @property
    def neval(self):
        try:
            return max([key.neval
                        for key in self._drr_lnnp_cache.keys()])
        except AttributeError:
            pass

    def save(self, file_name):
        r"""
        Save the current instance to an hdf5 file.

        Example
        -------
        >>> drr = DRR(0.1, j_grid_size=32)
        >>> d, d_err = drr(l_max=3)
        >>> drr.save('example.hdf5')
        >>> drr = DRR.from_file('example.hdf5')
        >>> d, d_err = drr(l_max=drr.l_max, neval=drr.neval)
        """
        with h5py.File(file_name, 'w') as h5:
            drr_lnnp_cache = h5.create_group("_drr_lnnp_cache")
            for key, value in self._drr_lnnp_cache.items():
                drr_lnnp_cache[str(dict(key._asdict()))] = value
            for key, value in self.__dict__.items():
                try:
                    h5[key] = value
                except TypeError:
                    pass

    def _read(self, file_name):
        r"""
        Read the cached data from an hdf5 file
        """
        with h5py.File(file_name, 'r') as h5:

            for key, value in h5['_drr_lnnp_cache'].items():
                try:
                    self._drr_lnnp_cache[Res(**make_dict(key))] = value.value
                except AttributeError:
                    self._drr_lnnp_cache = {Res(**make_dict(key)): value.value}

            for key, value in h5.items():
                if key != '_drr_lnnp_cache':
                    setattr(self, key, value.value)

    @classmethod
    def from_file(cls, file_name):
        r"""
        Load from file and return an instance

        Example
        -------
        >>> drr = DRR(0.1, j_grid_size=32)
        >>> d, d_err = drr(l_max=3)
        >>> drr.save('example.hdf5')
        >>> drr = DRR.from_file('example.hdf5')
        >>> d, d_err = drr(l_max=drr.l_max, neval=drr.neval)
        """
        drr = cls(1.0)
        drr._read(file_name)
        return drr


def integrate(func, integ, neval=1e4):
    integ(func, nitn=10, neval=neval)
    result = integ(func, nitn=10, neval=neval)
    try:
        res, err = np.array([[r.val, np.sqrt(r.var)] for r in result]).T
    except TypeError:
        res, err = result.val, np.sqrt(result.var)
    return res, err


@jit(nopython=True)
def a2_integrand(sma, j, sma_p, j_p, lnnp, true_anomaly):
    r"""
    Returns the :math:`|a_{\ell n n^{\prime}}|^{2}` integrand to use in the MC integration
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
    r"""
    Normalization factor for :math:`|a_{\ell n n^{prime}} |^{2}`
    """

    return (abs(special.sph_harm(n, l, 0, np.pi / 2)) ** 2 *
            abs(special.sph_harm(n_p, l, 0, np.pi / 2)) ** 2 *
            (4 * np.pi / (2 * l + 1)) ** 2) / (2 * l + 1)


class ResInterp(object):
    r"""
    Generates an interpolation function :math:`j^{\prime} (a^{\prime})` where :math:`j^{\prime}`
    satisfies the resonant condition: :math:`\nu^{\prime} (a^{\prime}, j^{\prime}) == \omega`

    Parameters
    ----------
    cusp : a `Cusp` instance
    omega : ndarray
          The resonance frequencies

    Example
    -------
    >>> from scrrpy import Cusp
    >>> from scrrpy.drr import ResInterp
    >>> import numpy as np
    >>> cusp = Cusp(gamma=1.75, mbh_mass=4e6, rh=2.0)
    >>> j = np.logspace(-2, 0, 11)[:-1]
    >>> omega = cusp.nu_p(a=0.1, j=j)
    >>> resint = ResInterp(cusp, omega)
    >>> jf = np.array([resint(o)(0.1) for o in omega])
    >>> abs(1 - jf/j).max() < 1e-4
    True
    """

    def __init__(self, cusp, omega, gr_factor=1.0):
        r"""
        """
        self._cusp = cusp
        self.size = 1000
        self._cusp.gr_factor = gr_factor
        self.omega = omega
        self._af = np.logspace(np.log10(16 * self._cusp.rg),
                               np.log10(self._cusp.rh),
                               1001)[1:]
        self.x = np.logspace(-5, 0, self.size + 1)[:-1][::-1]
        self.j_grid = self.make_grid()

    def __call__(self, omega):
        log_ap, jp = self.j_grid[omega]
        if jp.size == 0:
            return
        return lambda af: np.interp(np.log(af), log_ap, jp, left=0, right=0)

    def make_grid(self):
        j_grid = []
        for af in self._af:
            jlc = self._cusp.jlc(af)
            j = (1 - jlc) * self.x + jlc
            nup = self._cusp.nu_p(af, j)
            j_grid.append(np.exp(np.interp(self.omega, nup, np.log(j),
                                           left=-np.inf, right=-np.inf)))
        j_grid = np.array(list(zip(*j_grid)))
        return dict((omega, (np.log(self._af[j > 0]), j[j > 0]))
                    for omega, j in zip(self.omega, j_grid))

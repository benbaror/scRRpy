"""
A module for calculating Resonant Relaxation diffusion coefficients
"""

from functools import lru_cache

import numpy as np
import progressbar
import vegas
from numpy import mod
from numpy import sqrt
from scipy import special

import functools
import multiprocessing as mp

from .cusp import Cusp


class DRR(Cusp):
    """
    Resonant relaxation diffusion coefficient
    """

    def __init__(self, sma, **kwargs):
        """

        Arguments:
        - `sma`: semi-major axis [pc]
        - `l_sphr`: multipole order
        - `gamma`: slope of the density profile
        - `mbh`: Black hole mass [solar mass]
        - `mstar`: mass of individual stars  [solar mass]
        - `rh`: radius of influence [pc]
        """

        # Default arguments
        kwargs = {**dict(l_max=1,
                         gamma=1.75,
                         mbh=4e6,
                         mstar=1,
                         Njs=201,
                         rh=2),
                  **kwargs}

        super().__init__(gamma=kwargs['gamma'],
                         mbh=kwargs['mbh'],
                         mstar=kwargs['mstar'],
                         rh=kwargs['rh'])

        self.sma = sma
        self.j = np.logspace(np.log10(self.jlc(self.sma)), 0,
                             kwargs['Njs'])[:-1]
        self.omega = abs(self.nu_p(self.sma, self.j))
        self.l_max = kwargs['l_max']

    @lru_cache()
    def _res_intrp(self, ratio):
        return ResInterp(self, self.omega*ratio)

    def _integrand(self, j, sma_p, j_p, lnnp, true_anomaly):
        return (2*j_p/abs(self.d_nu_p(sma_p, j_p))/lnnp[-1] *
                A2_integrand(self.sma, j, sma_p, j_p,
                             lnnp[0], lnnp[1], lnnp[-1],
                             true_anomaly))

    def drr(self, l_max, neval=1e3, threads=1):
        lnnp = [(l, n, n_p)
               for l in range(1, l_max + 1)
               for n in range(1, l+1)
               for n_p in range(1, l+1)
               if not mod(l+n, 2)+mod(l+n_p, 2)]

        pbar = progressbar.ProgressBar()(range(len(lnnp)-1))


        drr = sum((self._drr_lnnp(*lnnp, neval=neval, threads=threads)[0]
                   for lnnp, _ in  zip(lnnp, pbar)))
        drr_err = sqrt(sum(self._drr_lnnp(*lnnp, neval=neval,
                                          threads=threads)[-1]**2
                           for lnnp in lnnp))
        return drr, drr_err

    def _drr_lnnp(self, l, n, n_p, neval=1e3, threads=1):
        """
        Calculates the l,n,n_p term of the diffusion coefficient
        """
        neval = int(neval)
        try:
            drr =  (self._drr_lnnp_cache[(l, n, n_p, neval)][0] +
                    self._drr_lnnp_cache[(l, n, -n_p, neval)][0])
            drr_err = sqrt(self._drr_lnnp_cache[(l, n, n_p, neval)][-1]**2 +
                           self._drr_lnnp_cache[(l, n, -n_p, neval)][-1]**2)
            return drr, drr_err
        except AttributeError:
            self._drr_lnnp_cache = {}
        except KeyError:
            pass

        if threads > 1:
            queue = mp.Queue()
            def parallel_drr(pos, seed, j, omega):
                np.random.seed(seed)
                results = [self._drr(j, omega, [l, n, n_p], neval=neval)
                           for j, omega in zip(j, omega)]
                drr = [result[0] for result in results]
                drr_err = [result[-1] for result in results]
                queue.put((pos, (drr, drr_err)))

            nchunks = self.j.shape[0] // threads + 1
            js = [self.j[i*nchunks: (i+1)*nchunks] for i in range(threads)]
            omegas = [self.omega[i*nchunks: (i+1)*nchunks]
                      for i in range(threads)]
            seeds = np.random.randint(1e6, size=threads)
            processes = [mp.Process(target=parallel_drr, args=(i, *x))
                         for i, x in enumerate(zip(seeds, js, omegas))]
            for processe in processes:
                processe.start()
            for processe in processes:
                processe.join()

            results = [queue.get() for p in processes]
            results.sort()
            drr = np.concatenate([result[-1][0] for result in results])
            drr_err = np.concatenate([result[-1][1] for result in results])

        else:
            widgets = ['{}, {}, {}'.format(l, n, n_p), ' ',
                       progressbar.Percentage(),
                       ' (', progressbar.SimpleProgress(), ')', ' ',
                       progressbar.Bar(), ' ',
                       progressbar.Timer(), ' ',
                       progressbar.AdaptiveETA()]

            pbar = progressbar.ProgressBar(widgets=widgets)

            drr, drr_err = np.zeros([2, 2, self.j.size])


            for j_i, omegai, i in zip(self.j, self.omega,
                                      pbar(range(self.j.size))):
                drr[i], drr_err[i] = self._drr(j_i, omegai, [l, n, n_p],
                                               neval=neval)

        self._drr_lnnp_cache[l, n, -n_p, neval] = (drr[:, 0], drr_err[:, 0])
        self._drr_lnnp_cache[l, n, n_p, neval] = (drr[:, -1], drr_err[:, -1])

        drr =  (self._drr_lnnp_cache[(l, n, n_p, neval)][0] +
                self._drr_lnnp_cache[(l, n, -n_p, neval)][0])
        drr_err = sqrt(self._drr_lnnp_cache[(l, n, n_p, neval)][-1]**2 +
                       self._drr_lnnp_cache[(l, n, -n_p, neval)][-1]**2)
        return drr, drr_err



    def tau2(self, l_max, neval=1e3, include_zero=True):
        n_p_min = 0 if include_zero else 1
        tau2 = sum([self._tau2_lnnp(l, n, n_p, neval=neval)[0]
                    for l in range(1, l_max + 1)
                    for n in range(1, l+1)
                    for n_p in range(n_p_min, l+1)
                    if not mod(l+n, 2)+mod(l+n_p, 2)])

        tau2_err = sqrt(sum([self._tau2_lnnp(l, n, n, neval=neval)[-1]**2
                             for l in range(1, l_max + 1)
                             for n in range(1, l+1)
                             for n_p in range(n_p_min, l+1)
                             if not mod(l+n, 2)+mod(l+n_p, 2)]))
        return tau2, tau2_err

    @lru_cache()
    def _tau2_lnnp(self, l, n, n_p, neval=1e3):
        """
        Calculates the l,n,n_p term of the diffusion coefficient
        """
        tau2, tau2_err = np.zeros([2, self.j.size])

        widgets = ['{}, {}, {}'.format(l, n, n_p), ' ',
                   progressbar.Percentage(),
                   ' (', progressbar.SimpleProgress(), ')', ' ',
                   progressbar.Bar(), ' ',
                   progressbar.Timer(), ' ',
                   progressbar.AdaptiveETA()]

        pbar = progressbar.ProgressBar(widgets=widgets)

        for j_i, omegai, i in zip(self.j, self.omega,
                                  pbar(range(self.j.size))):
            tau2[i], tau2_err[i] = self._tau2(j_i, [l, n, n_p], neval=neval)
        return tau2, tau2_err

    def _drr(self, j, omega, lnnp, neval=1e3):

        integ = vegas.Integrator(5 * [[0, 1]])
        ratio = lnnp[1]/lnnp[-1]

        @vegas.batchintegrand
        def Clnnp(x):
            true_anomaly = x[:, :-1].T*np.pi
            sma_f = self.inverse_cumulative_a(x[:, -1])
            jf1 = self._res_intrp(ratio).get_jf1(omega*ratio, sma_f)
            jf2 = self._res_intrp(ratio).get_jf2(omega*ratio, sma_f)
            res = np.zeros((x.shape[0], 2), float)
            ix1 = jf1 > 0
            ix2 = jf2 > 0
            res[ix1, 0] = self._integrand(j, sma_f[ix1], jf1[ix1], lnnp,
                                          true_anomaly[:, ix1])
            res[ix2, 1] = self._integrand(j, sma_f[ix2], jf2[ix2], lnnp,
                                          true_anomaly[:, ix2])
            return res
        return 4*np.pi*(np.array(integrate(Clnnp, integ, neval)) *
                        _A2_norm_factor(*lnnp)*lnnp[1]**2)

    def _tau2(self, j, lnnp, neval=1e3):
        if np.mod(lnnp[0] + lnnp[1], 2) or np.mod(lnnp[0] + lnnp[-1], 2):
            return 0, 0

        tau2, tau2_err = np.zeros([2, self.j.size])

        integ = vegas.Integrator(6 * [[0, 1]])
        sma = self.sma
        gamma = self.gamma
        rh = self.rh
        l, n, n_p = lnnp
        j2 = j**2
        ecc = sqrt(1-j**2)

        @vegas.batchintegrand
        def Clnnp(x):
            true_anomaly = x[:, :-2].T*np.pi
            sma_p = (x[:, -2])**(1/(3-gamma))*rh
            j2_p = x[:, -1]
            cnnp = np.cos(true_anomaly.T*np.array([n, n, n_p, n_p])).T
            cnnp = cnnp[0]*cnnp[1]*cnnp[2]*cnnp[3]
            eccp = sqrt(1-j2_p)
            r12 = sma*j2/(1-ecc*np.cos(true_anomaly[:2]))
            rp12 = sma_p*j2_p/(1-eccp*np.cos(true_anomaly[2:]))
            r_1, r_2 = r12[0], r12[-1]
            rp1, rp2 = rp12[0], rp12[-1]
            return (cnnp/j2_p/sma_p**4 *
                    (np.minimum(r_1, rp1)*np.minimum(r_2, rp2))**(2*l+1) /
                    (r_1*r_2*rp1*rp2)**(l-1))
        # Symmetry factor for using n>0 and n_p >= 0
        symmetry_factor = 2 if n_p == 0 else 4
        return symmetry_factor*(np.array(integrate(Clnnp, integ, neval)) *
                                _A2_norm_factor(*lnnp)*lnnp[1]**2)/j2/sma**2


def integrate(func, integ, neval):
    result = integ(func, nitn=10, neval=neval/5)
    result = integ(func, nitn=10, neval=neval)
    try:
        return np.array([[r.val, np.sqrt(r.var)] for r in result]).T
    except TypeError:
        return result.val, np.sqrt(result.var)


def A2_integrand(sma, j, sma_p, j_p, l, n, n_p, true_anomaly):
    """
    returns the |alnnp|^2 integrand to use the the MC integration
    """
    cnnp = np.cos(true_anomaly.T*np.array([n, n, n_p, n_p])).T
    cnnp = cnnp[0]*cnnp[1]*cnnp[2]*cnnp[3]
    ecc, eccp = np.sqrt(1-j**2), np.sqrt(1-j_p**2)
    r12 = (sma*(1-ecc**2)/(1-ecc*np.cos(true_anomaly[:2])))
    rp12 = (sma_p*(1-eccp**2)/(1-eccp*np.cos(true_anomaly[2:])))
    r_1, r_2 = r12[0], r12[-1]
    rp1, rp2 = rp12[0], rp12[-1]
    return (cnnp/j**2/j_p**2/sma**2/sma_p**4 *
            (np.minimum(r_1, rp1)*np.minimum(r_2, rp2))**(2*l+1) /
            (r_1*r_2*rp1*rp2)**(l-1))


@lru_cache()
def _A2_norm_factor(l, n, n_p):
    """
    Normalization factor for |alnnp|^2
    ! To be implemented
    """

    return (abs(special.sph_harm(n, l, 0, np.pi/2))**2 *
            abs(special.sph_harm(n_p, l, 0, np.pi/2))**2 *
            (4*np.pi/(2*l + 1))**2)/(2*l + 1)


class ResInterp(object):
    """
    Interpolation function for the resonant condition
    """

    def __init__(self, cusp, omega):
        """
        """
        self._cusp = cusp
        self.omega = omega
        self._af = np.logspace(np.log10(self._cusp.rg),
                               np.log10(self._cusp.rh),
                               1000)
        # self._jf = np.logspace(np.log10(self._cusp.jlc(self._cusp.rh)),
        #                                 0, 1001)[:-1]

        def get_j(nup):
            jf = self._jf[nup > 0]
            nup = nup[nup > 0]
            s = np.argsort(nup)
            j = np.interp(self.omega, nup[s], jf[s], left=0, right=0)
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

        for i, a in enumerate(self._af[self._af < a_gr1]):
            self._jf = np.logspace(np.log10(self._cusp.jlc(a)),
                                   0, 1001)[:-1]
            nup = self._cusp.nu_p(a, self._jf)
            self._j1[i, :] = get_j(nup)

        last = i + 1
        for i, a in enumerate(self._af[self._af > a_gr1]):
            self._jf = np.logspace(np.log10(self._cusp.jlc(a)),
                                   0, 1001)[:-1]
            nup = self._cusp.nu_p(a, self._jf)
            self._j1[i+last, :] = get_j(nup)
            if any(nup < 0):
                self._j2[i+last, :] = get_j(-nup)

    def get_jf1(self, omega, af):
        i = np.argmin(abs(self.omega-omega))
        if abs(self.omega[i]-omega) > 1e-8:
            raise ValueError
        j = self._j1[:, i]
        if sum(j > 0):
            return np.interp(af, self._af[j > 0], j[j > 0], left=0, right=0)
        else:
            return af*0.0

    def get_jf2(self, omega, af):
        i = np.argmin(abs(self.omega-omega))
        if abs(self.omega[i]-omega) > 1e-8:
            raise ValueError
        j = self._j2[:, i]
        if sum(j > 0):
            return np.interp(af, self._af[j > 0], j[j > 0], left=0, right=0)
        else:
            return af*0.0


def get_drr(drr, neval, lnnp):
    return drr._drr_lnnp(*lnnp, neval=neval)

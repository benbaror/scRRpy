from cusp import Cusp
import numpy as np
import vegas
import progressbar
from scipy import special


class DRR(Cusp):
    """
    """

    def __init__(self, a, l_sphr=1, gamma=1.75, mbh=4e6, mstar=1, rh=2):
        """

        Arguments:
        - `a`: semi-major axis
        - `l_sphr`: multipole order
        - `gamma`: slope of the density profile
        - `mbh`: Black hole mass [solar mass]
        - `mstar`: mass of individual stars  [solar mass]
        - `rh`: radius of influence [pc]
        """
        super().__init__(gamma=gamma, mbh=mbh, mstar=mstar, rh=rh)
        self.a = a
        self.j = np.logspace(np.log10(self.jlc(self.a)), 0, 101)[:-1]
        self.omega = abs(self.nu_p(self.a, self.j))
        self.l_sphr = l_sphr

    def res_int(self, ratio):
        try:
            return self._res_int[ratio]
        except AttributeError:
            self._res_int = {}
        except KeyError:
            pass
        self._res_int[ratio] = Res_interp(self, self.omega*ratio)
        return self._res_int[ratio]

    def _integrand(self, a, j, ap, jp, l, n, n_p, true_anomaly):
        return (2*jp/abs(self.d_nu_p(ap, jp))/n_p *
                A2_integrand(a, j, ap, jp, l, n, n_p, true_anomaly))

    def drr(self, l_sphr, n, n_p, neval=1e4):
        key = '{},{},{}'.format(l_sphr, n, n_p)
        try:
            return self._drr_dict[key]
        except AttributeError:
            self._drr_dict = {}
        except KeyError:
            pass
        self._drr_dict[key] = np.zeros([self.j.size, 2])

        bar = progressbar.ProgressBar()
        for ji, omegai, i in zip(self.j, self.omega, bar(range(self.j.size))):
            self._drr_dict[key][i, :] = self._drr(self.a, ji, omegai,
                                                  l_sphr, n, n_p, neval=neval)

        return self._drr_dict[key]

    def _drr(self, a, j, omega, l, n, n_p, neval=1e3):
        integ = vegas.Integrator(5 * [[0, 1]])
        ratio = n/n_p

        @vegas.batchintegrand
        def C(x):
            true_anomaly = x[:, :-1].T*np.pi
            af = self.inverse_cumulative_a(x[:, -1])
            jf1 = self.res_int(ratio).get_jf1(omega*ratio, af)
            jf2 = self.res_int(ratio).get_jf2(omega*ratio, af)
            x = np.zeros_like(af)
            ix1 = jf1 > 0
            ix2 = jf2 > 0
            x[ix1] = self._integrand(a, j, af[ix1], jf1[ix1], l, n, n_p,
                                     true_anomaly[:, ix1])
            x[ix2] = self._integrand(a, j, af[ix2], jf2[ix2], l, n, n_p,
                                     true_anomaly[:, ix2])
            return x
        return (np.array(integrate(C, integ, neval)) *
                self._A2_norm_factor(l, n, n_p)*n**2)

    def _A2_norm_factor(self, l, n, n_p):
        """
        Normalization factor for |alnnp|^2
        ! To be implemented
        """
        key = '{},{},{}'.format(l, n, n_p)
        try:
            return self._A2_norm[key]
        except AttributeError:
            self._A2_norm = {}
        except KeyError:
            pass
        self._A2_norm[key] = _A2_norm_factor(l, n, n_p)
        return self._A2_norm[key]


def integrate(func, integ, neval):
    result = integ(func, nitn=10, neval=neval)
    result = integ(func, nitn=10, neval=neval)
    try:
        return np.array([[r.val, np.sqrt(r.var)] for r in result]).T
    except TypeError:
        return result.val, np.sqrt(result.var)


def A2_integrand(a, j, ap, jp, l, n, n_p, true_anomaly):
    """
    returns the |alnnp|^2 integrand to use the the MC integration
    """
    c = np.prod(np.cos(true_anomaly.T*np.array([n, n, n_p, n_p])), 1)
    ecc, eccp = np.sqrt(1-j**2), np.sqrt(1-jp**2)
    r1, r2 = (a*(1-ecc**2)/(1-ecc*np.cos(true_anomaly[:2])))
    rp1, rp2 = (ap*(1-eccp**2)/(1-eccp*np.cos(true_anomaly[2:])))
    return 16*(c/j**2/jp**2/a**2/ap**4 *
               (np.minimum(r1, rp1)*np.minimum(r2, rp2))**(2*l+1) /
               (r1*r2*rp1*rp2)**(l-1))


def _A2_norm_factor(l, n, n_p):
    """
    Normalization factor for |alnnp|^2
    ! To be implemented
    """
    return (abs(special.sph_harm(n, l, 0, np.pi/2))**2 *
            abs(special.sph_harm(n_p, l, 0, np.pi/2))**2 *
            (4*np.pi/(2*l + 1)))**2


class Res_interp(object):
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

"""Properties of the stellar cusp"""

import numpy as np
# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
from astropy.constants import G
from astropy.constants import M_sun
from astropy.constants import c
# from functools import partial
from numba import jit
from numpy import pi
from scipy.special import eval_legendre

# from numba.errors import TypingError

M0 = M_sun
# Define physical constants
# Gravitational radius of a solar mass black hole
RG0 = (G*M_sun/c**2).to('pc').value
# Light crossing time of solar mass black hole
TG0 = (G*M_sun/c**3).to('year').value


# noinspection PyCompatibility
class Cusp(object):
    """
    A power law stellar cusp around a massive black hole (MBH).
    The cusp is assumed to have an isotropic distribution function
    :math:`f(E) \propto |E|^p` corresponding ro a stellar density
    :math:`n(r) \propto r^{-\gamma}` where :math:`\gamma = 3/2 + p`

    Parameters
    ----------
    gamma : float, int, optional
        The slope of the density profile.
        default: 7/4 (Bahcall wolf cusp)
    mbh_mass : float, int
        Mass of the MBH [solar mass].
        default: 4.3 10^6 (Milky Way MBH)
    star_mass : float, int
        Mass of individual stars [solar mass].
        default: 1.0
    rh : float, int
        Radius of influence [pc].
        Define as the radius in which the velocity
        dispersion of the stellar cusp :math:`\sigma` is equal to the
        Keplerian velocity due to the MBH
        :math:`\sigma(r_h)^2 = G M_\bullet / r_h`.
        default: 2.0

    TODO: Implement normalization Total mass at r_H
    TODO: Implement normalization N(a) vs N(r)
    """

    def __init__(self, gamma=1.75, mbh_mass=4e6,
                 star_mass=1.0, rh=2.0):
        """
        """
        self.gamma = float(gamma)
        self.mbh_mass = float(mbh_mass)
        self.star_mass = float(star_mass)
        self.rh = float(rh)
        self.gr_factor = 1.0

    @property
    def rg(self):
        """
        Gravitational radius of the MBH [pc]
        """
        return RG0*self.mbh_mass

    @property
    def tg(self):
        """
        Light crossing time of the MBH [sec]
        """
        return TG0*self.mbh_mass

    def jlc(self, a):
        """
        Relativistic loss cone:
          Minimal normalized angular momentum on which orbits are stable.

          :math:`j_{lc} = J_{lc} / J_c`,
          where :math:`J_{lc} = 4GM_\bullat/c` is the last stable orbit in the
          parabolic limit and :math:`J_c = \sqrt{GM_\bullet a}` is the
          maximal (circular) stable orbit.

          This is an approximation which works when the orbital binding energy
          `E` is much smaller than rest energy of the MBH `Mc^2`.

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return 4 * np.sqrt(self.rg / a)

    @property
    def mass_ratio(self):
        """
        MBH to star mass ratio
        """
        return self.mbh_mass / self.star_mass

    @property
    def total_number_of_stars(self):
        """
        Number of stars within the radius of influence `rh`
        """
        return self.total_stellar_mass / self.star_mass

    @property
    def total_stellar_mass(self):
        """
        Total mass within the radius of influence `rh` [solar mass]

        TODO: Implement normalization
        """
        return self.mass_ratio

    def number_of_stars(self, a):
        """
        Number of stars with semi-major axis smaller than a[pc]

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return self.total_number_of_stars * (a / self.rh) ** (3 - self.gamma)

    def stellar_mass(self, a):
        """
        Enclosed mass within r = a[pc].
        TODO: check M(r) vs M(a)

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return self.total_stellar_mass * (a / self.rh) ** (3 - self.gamma)

    def period(self, a):
        """
        The orbital period in years at a[pc]

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return 2*pi/self.nu_r(a)

    def nu_r(self, a):
        """
        The orbital frequency in rad/year at a[pc]

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return (self.rg/a)**1.5/self.tg

    def nu_mass(self, a, j):
        """
        Precession frequency [rad/year] due to stellar mass.

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        j: float, array
            Normalized angular momentum :math:`j = J/J_c = \sqrt{1-e^2}`.
        """
        return self._nu_mass0(a)*self._g(j)

    def nu_gr(self, a, j):
        """
        Precession frequency [rad/year] due to general relativity
        (first PN term)

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        j: float, array
            Normalized angular momentum :math:`j = J/J_c = \sqrt{1-e^2}`.
        """
        return self.gr_factor*self.nu_r(a)*3*(self.rg/a)/j**2

    def nu_p(self, a, j):
        """
        Precession frequency [rad/year].
           `nu_p(a, j) = nu_gr(a, j) + nu_mass(a, j)`

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        j: float, array
            Normalized angular momentum :math:`j = J/J_c = \sqrt{1-e^2}`.
        """
        return self.nu_gr(a, j) + self.nu_mass(a, j)

    def nu_p1(self, a):
        """
        Precession frequency at j=1
        """
        return self.nu_gr(a, 1.0) + self._nu_mass0(a)*(3-self.gamma)/2

    def d_nu_p(self, a, j):
        """
        The derivative of nu_p with respect to j, defined to be positive
        """
        d_nu_gr = 6*self.gr_factor*(self.rg/a)/(j*j*j)
        d_nu_mass = -self.stellar_mass(a) / self.mbh_mass * self._gp(j)
        return (d_nu_gr - d_nu_mass)*self.nu_r(a)

    def inverse_cumulative_a(self, x):
        """
        The inverse of N(a). Useful to generate a random
        sample of semi-major axis.

        Parameters
        ----------
        x: float, array
          x in [0, 1]

        Example
        -------
        >>> cusp = Cusp(gamma=1.75)
        >>> np.random.seed(1234)
        >>> sma = cusp.inverse_cumulative_a(np.random.rand(100))
        >>> print("{:0.10}, {:0.10}, {:0.10}".format(sma.min(), sma.mean(), sma.max()))
        0.03430996478, 1.147418232, 1.987320281
        """
        return x**(1/(3-self.gamma))*self.rh

    def _nu_mass0(self, a):
        """
        The frequency of the mass precession divided by g(j).
        """
        return -self.nu_r(a) * self.stellar_mass(a) / self.mbh_mass

    def _g(self, j):
        """
        g(j) = -j^(6-2\gamma)/pi/sqrt(1-j^2)*\int_0^\pi cos(s) /
               (1+sqrt(1+j^2)^(3-\gamma)
        For \gamma = 2:
            g(j) = j/(1+j)
        For \gamma = 1:
            g(j) = j
        For any \gamma
        g(1) = (3-\gamma)/2
        """
        p1 = self._eval_legendre_inv(1 - self.gamma, j)
        p2 = self._eval_legendre_inv(2 - self.gamma, j)
        return - j**(4 - self.gamma)/(1-j**2)*(p1 - p2/j)

    def _gp(self, j):
        """
        dg(j)/dj
        """
        n2 = 2 - self.gamma
        p1 = self._eval_legendre_inv(1 - self.gamma, j)
        p2 = self._eval_legendre_inv(2 - self.gamma, j)
        return gp(j, p1, p2, n2)

    @property
    def a_gr1(self):
        """
        The sma below which nup is only positive,
        that is nup(a,j=1) = 0
        """
        return ((self.gr_factor *
                 6 / (3-self.gamma) * self.mass_ratio /
                 self.total_number_of_stars * self.rg / self.rh) **
                (1/(4-self.gamma)) *
                self.rh)

    @staticmethod
    def _eval_legendre_inv(n, j):
        return eval_legendre(n, 1/j)

    # def _eval_legendre_inv_int(self, n, j):
    #     try:
    #         return self._eval_legendre_inv[n](j)
    #     except (AttributeError, KeyError, TypingError) as err:
    #         if type(err) is TypingError:
    #             return self._eval_legendre_inv[n](np.atleast_1d(j))[0]
    #         if type(err) is AttributeError:
    #             self._eval_legendre_inv = {}
    #         j_samp = np.logspace(np.log10(self.jlc(self.rh)), 0, 1000)
    #         pn = eval_legendre(n, 1/j_samp)
    #         x0 = np.log(j_samp[0])
    #         dx_inv = 1/(np.log(j_samp[1]) - x0)
    #         self._eval_legendre_inv[n] = partial(interp_loglog, x0=x0,
    #                                              dx_inv=dx_inv,
    #                                              logf=np.log(pn))
    #         return self._eval_legendre_inv[n](j)


@jit(nopython=True)
def interp_semilogx(x_int, x0, dx_inv, f):
    f_int = np.empty_like(x_int)
    for i in range(x_int.size):
        x_dx = (np.log(x_int[i])-x0)*dx_inv
        ind = int(x_dx)
        w = x_dx - ind
        f_int[i] = f[ind]*(1-w) + f[ind+1]*w
    return f_int


@jit(nopython=True)
def interp_loglog(x_int, x0, dx_inv, logf):
    f_int = np.empty_like(x_int)
    for i, x in enumerate(x_int):
        x_dx = (np.log(x)-x0)*dx_inv
        ind = int(x_dx)
        w = x_dx - ind
        f_int[i] = logf[ind]*(1-w) + logf[ind+1]*w
    return np.exp(f_int)


@jit(nopython=True)
def gp(j, p1, p2, n2):
    return (j**n2 / (1 - j**2)**2 *
            ((j**2 + 1)*p2 - j*(2 + n2*(1 - j**2))*p1))

r"""
Properties of the stellar cusp
"""

import numpy as np
from astropy.constants import G
from astropy.constants import M_sun
from astropy.constants import c
from numba import jit
from numpy import pi
from scipy.special import eval_legendre

from .utils import interp_reg_loglog

M0 = M_sun
# Define physical constants
# Gravitational radius of a solar mass black hole
RG0 = (G * M_sun / c ** 2).to('pc').value
# Light crossing time of solar mass black hole
TG0 = (G * M_sun / c ** 3).to('year').value


# noinspection PyCompatibility
class Cusp(object):
    r"""
    A power law stellar cusp around a massive black hole (MBH).
    The cusp is assumed to have an isotropic distribution function
    :math:`f(E) \propto |E|^p` corresponding ro a stellar density
    :math:`n(r) \propto r^{-\gamma}` where :math:`\gamma = \tfrac{3}{2} + p`

    TODO - Implement normalization Total mass at :math:`r_{\mathrm{h}}`

    TODO - Implement normalization :math:`N(a)` vs :math:`N(r)`

    Parameters
    ----------
    gamma : float, int, optional
        The slope of the density profile.
        Default: 7/4 (Bahcall-Wolf cusp)
    mbh_mass : float, int
        Mass of the MBH [solar mass].
        Default: :math:`4.3 \times 10^6` (Milky Way MBH)
    star_mass : float, int
        Mass of individual stars [solar mass].
        Default: 1.0
    rh : float, int
        Radius of influence [pc].
        Define as the radius in which the velocity
        dispersion of the stellar cusp :math:`\sigma` is equal to the
        Keplerian velocity due to the MBH
        :math:`\sigma(r_{\mathrm{h}})^2 = G M_{\bullet} / r_{\mathrm{h}}`.
        Default: 2.0
    """

    def __init__(self, gamma=1.75, mbh_mass=4e6,
                 star_mass=1.0, rh=2.0):
        r"""
        """
        self.gamma = float(gamma)
        self.mbh_mass = float(mbh_mass)
        self.star_mass = float(star_mass)
        self.rh = float(rh)
        self.gr_factor = 1.0
        self.mu = 1.0

    @property
    def rg(self):
        """
        Gravitational radius of the MBH [pc]
        """
        return RG0 * self.mbh_mass

    @property
    def tg(self):
        r"""
        Light crossing time of the MBH [sec]
        """
        return TG0 * self.mbh_mass

    def jlc(self, a):
        r"""
        Relativistic loss cone

        Minimal normalized angular momentum on which orbits are stable.

        :math:`j_{\mathrm{lc}} = J_{\mathrm{lc}} / J_{\mathrm{c}}`,
        where :math:`J_{\mathrm{lc}} = 4GM_{\bullet}/c` is the last stable orbit in the
        parabolic limit and :math:`J_{\mathrm{c}} = \sqrt{GM_{\bullet} a}` is the
        maximal (circular) stable orbit.

        This is an approximation which works when the orbital binding energy
        :math:`E` is much smaller than rest energy of the MBH :math:`Mc^2`.

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return 4 * np.sqrt(self.rg / a)

    @property
    def mass_ratio(self):
        r"""
        MBH to star mass ratio
        """
        return self.mbh_mass / self.star_mass

    @property
    def total_number_of_stars(self):
        r"""
        Number of stars within the radius of influence :math:`r_{\mathrm{h}}`
        """
        return self.total_stellar_mass / self.star_mass

    @property
    def total_stellar_mass(self):
        r"""
        Total mass within the radius of influence :math:`r_\mathrm{h}` [solar mass]

        TODO - Implement normalization
        """
        return self.mbh_mass*self.mu

    def number_of_stars(self, a):
        r"""
        Number of stars with semi-major axis smaller than :math:`a` [pc]

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return self.total_number_of_stars * (a / self.rh) ** (3 - self.gamma)

    def stellar_mass(self, a):
        r"""
        Enclosed mass within :math:`r = a` [pc].

        TODO - check :math:`M(r)` vs :math:`M(a)`

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return self.total_stellar_mass * (a / self.rh) ** (3 - self.gamma)

    def period(self, a):
        r"""
        The orbital period in years at :math:`a` [pc]

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return 2 * pi / self.nu_r(a)

    def nu_r(self, a):
        r"""
        The orbital frequency in rad/year at :math:`a` [pc]

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        """
        return (self.rg / a) ** 1.5 / self.tg

    def nu_mass(self, a, j):
        r"""
        Precession frequency [rad/year] due to stellar mass.

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        j: float, array
            Normalized angular momentum :math:`j = J/J_{\mathrm{c}} = \sqrt{1-e^2}`.
        """
        return self._nu_mass0(a) * self._g(j)

    def nu_gr(self, a, j):
        r"""
        Precession frequency [rad/year] due to general relativity
        (first PN term)

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        j: float, array
            Normalized angular momentum :math:`j = J/J_\mathrm{c} = \sqrt{1-e^2}`.
        """
        return self.gr_factor * self.nu_r(a) * 3 * (self.rg / a) / j ** 2

    def nu_p(self, a, j):
        r"""
        Precession frequency [rad/year]

        :math:`\nu_{\mathrm{p}} (a, j) = \nu_{\mathrm{gr}} (a, j) + \nu_{\mathrm{mass}} (a, j)`

        Parameters
        ----------
        a: float, array
            Semi-major axis [pc].
        j: float, array
            Normalized angular momentum :math:`j = J/J_\mathrm{c} = \sqrt{1-e^2}`.
        """
        return self.nu_gr(a, j) + self.nu_mass(a, j)

    def nu_p1(self, a):
        r"""
        Precession frequency at :math:`j=1`
        """
        return self.nu_gr(a, 1.0) + self._nu_mass0(a) * (3 - self.gamma) / 2

    def d_nu_p(self, a, j):
        r"""
        The derivative of :math:`\nu_\mathrm{p}` with respect to :math:`j`, defined to be positive
        """
        d_nu_gr = 6 * self.gr_factor * (self.rg / a) / (j * j * j)
        d_nu_mass = - self.stellar_mass(a) / self.mbh_mass * self._gp(j)
        return (d_nu_gr - d_nu_mass) * self.nu_r(a)

    def inverse_cumulative_a(self, x):
        r"""
        The inverse of :math:`N(a)`. Useful to generate a random
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
        return x ** (1 / (3 - self.gamma)) * self.rh

    def _nu_mass0(self, a):
        r"""
        The frequency of the mass precession divided by :math:`g(j)`.
        """
        return -self.nu_r(a) * self.stellar_mass(a) / self.mbh_mass

    def _g(self, j):
        r"""
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
        return - j ** (4 - self.gamma) / (1 - j ** 2) * (p1 - p2 / j)

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
        r"""
        The sma below which :math:`\nu_\mathrm{p}` is only positive,
        that is :math:`\nu_\mathrm{p} (a,j=1) = 0`
        """
        return ((self.gr_factor *
                 6 / (3 - self.gamma) * self.mass_ratio /
                 self.total_number_of_stars * self.rg / self.rh) **
                (1 / (4 - self.gamma)) *
                self.rh)

    def _eval_legendre_inv(self, n, j):
        # return eval_legendre(n, 1/j)
        try:
            return self._eval_legendre_inv_cache[n](j)
        except (AttributeError, KeyError) as err:
            if type(err) is AttributeError:
                self._eval_legendre_inv_cache = {}
            j_samp = np.logspace(np.log10(self.jlc(self.rh)), 0, 1000)
            pn = eval_legendre(n, 1 / j_samp)
            self._eval_legendre_inv_cache[n] = interp_reg_loglog(j_samp, pn)
        return self._eval_legendre_inv(n, j)


@jit(nopython=True)
def gp(j, p1, p2, n2):
    return (j ** n2 / (1 - j ** 2) ** 2 *
            ((j ** 2 + 1) * p2 - j * (2 + n2 * (1 - j ** 2)) * p1))

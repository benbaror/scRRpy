"""Properties of the stellar cusp"""
import numpy as np
from astropy import constants
from numpy import pi
from scipy.special import eval_legendre
from functools import partial
from numba import jit
from numba.errors import TypingError
G = constants.G
M0 = constants.M_sun
C = constants.c
RG0 = (G*M0/C**2).to('pc').value
TG0 = (G*M0/C**3).to('year').value


class Cusp(object):
    """
    """

    def __init__(self, gamma=1.75, mbh=4e6, mstar=1, rh=2):
        """
        """
        self.gamma = gamma
        self.mbh = mbh
        self.mstar = mstar
        self.rh = rh
        self.gr_factor = 1.0

    @property
    def rg(self):
        """
        Gravitational rads in pc
        """
        return RG0*self.mbh

    @property
    def tg(self, ):
        """
        Gravitational time
        """
        return TG0*self.mbh

    @property
    def Q(self):
        """
        """
        return self.mbh/self.mstar

    @property
    def Nh(self):
        """
        """
        return self.Q

    def Nstars(self, a):
        """
        The number of stars within a[pc]
        """
        return self.Nh*(a/self.rh)**(3-self.gamma)

    def Mstars(self, a):
        """
        The number of stars within a[pc]
        """
        return self.mbh*(a/self.rh)**(3-self.gamma)

    def dMda(self, a):
        """
        dM/da at a[pc]
        """
        return (3-self.gamma)*self.mbh*(a/self.rh)**(3-self.gamma)/a

    def period(self, a):
        """
        The orbital period in years at a[pc]
        """
        return 2*pi/self.nu_r(a)

    def nu_r(self, a):
        """
        The orbital frequency in rad/year at a[pc]
        """
        return (self.rg/a)**(1.5)/self.tg

    def nu_mass(self, a, j):
        """
        The frequency of the mass precession
        """
        return self._nu_mass0(a)*self._g(j)

    def _nu_mass0(self, a):
        """
        The frequency of the mass precession divided by g(j)
        """
        return -self.nu_r(a)*self.Mstars(a)/self.mbh

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
        j = np.asarray(j, dtype=np.float64)
        P1 = self.eval_legendre_inv(1 - self.gamma, j)
        P2 = self.eval_legendre_inv(2 - self.gamma, j)
        return  - j**(4 - self.gamma)/(1-j**2)*(P1 - P2/j)

    def _gp(self, j):
        """
        dg(j)/dj
        """
        n2 = 2 - self.gamma
        P1 = self.eval_legendre_inv(1 - self.gamma, j)
        P2 = self.eval_legendre_inv(2 - self.gamma, j)
        return gp(j, P1, P2, n2)

    def nu_gr(self, a, j):
        """
        The frequency of the GR precession
        """
        return self.gr_factor*self.nu_r(a)*3*(self.rg/a)/j**2

    def nu_p(self, a, j):
        """
        The frequency of the precession
        """
        return self.nu_gr(a, j) + self.nu_mass(a, j)

    def nu_p1(self, a):
        """
        The frequency of the precession at j=1
        """
        return self.nu_gr(a, 1.0) + self._nu_mass0(a)*(3-self.gamma)/2

    def d_nu_p(self, a, j):
        """
        The derivative of \nu_p with respect to j defined to be positive
        """
        d_nu_gr = 6*self.gr_factor*(self.rg/a)/(j*j*j)
        gp = self._gp(j)
        d_nu_mass = -self.Mstars(a)/self.mbh*gp
        return (d_nu_gr - d_nu_mass)*self.nu_r(a)

    def nu_mass_inv(self, j, omega):
        """
        Return a such that nu_mass(a,j) = omega
        """
        nu_mass_rh = abs(self.nu_mass(self.rh, j))
        return (omega/nu_mass_rh)**(1/(3/2-self.gamma))*self.rh

    def inverse_cumulative_a(self, r):
        return r**(1/(3-self.gamma))*self.rh

    @property
    def a_gr1(self):
        """
        The below which nup is only positive,
        that is nup(a,j=1) = 0
        """
        return ((self.gr_factor *
                 6/(3-self.gamma)*self.Q/self.Nh*self.rg/self.rh) **
                (1/(4-self.gamma)) *
                self.rh)

    def jlc(self, a):
        """
        Relativistic loss cone
        """
        return 4*np.sqrt(self.rg/a)

    def eval_legendre_inv(self, n, j):
        try:
            return self._eval_legendre_inv[n](j)
        except (AttributeError, KeyError, TypingError) as err:
            if type(err) is TypingError:
                return self._eval_legendre_inv[n](np.atleast_1d(j))[0]
            if type(err) is AttributeError:
                self._eval_legendre_inv = {}
            j_samp = np.logspace(np.log10(self.jlc(self.rh)), 0, 1000)
            Pn = eval_legendre(n, 1/j_samp)
            x0 = np.log(j_samp[0])
            dx_inv =  1/(np.log(j_samp[1]) - x0)
            self._eval_legendre_inv[n] = partial(interp_loglog, x0=x0,
                                                 dx_inv=dx_inv,
                                                 logf=np.log(Pn))
            return self._eval_legendre_inv[n](j)


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
    for i in range(x_int.size):
        x_dx = (np.log(x_int[i])-x0)*dx_inv
        ind = int(x_dx)
        w = x_dx - ind
        f_int[i] = logf[ind]*(1-w) + logf[ind+1]*w
    return np.exp(f_int)


@jit(nopython=True)
def gp(j, P1, P2, n2):
    return (j**n2 / (1 - j**2)**2 *
            ((j**2 + 1)*P2 - j*(2 + n2*(1 - j**2))*P1))

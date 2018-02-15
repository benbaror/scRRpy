from scrrpy import DRR
import matplotlib.pyplot as plt


drr = DRR(0.1, gamma=1.75, mbh_mass=4.3e6, rh=2.0, star_mass=1.0, j_grid_size=32)

djj, djj_err = drr(l_max=5)

plt.loglog(drr.j, djj)
plt.xlabel(r'$J/J_\mathrm{c}$')
plt.ylabel(r'$D^{\mathrm{RR}}_{JJ}/J_\mathrm{c}^2$ [1/Myr]')

plt.show()





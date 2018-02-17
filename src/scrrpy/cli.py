"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mscrrpy` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``scrrpy.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``scrrpy.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import click

from .drr import DRR


@click.command()
@click.argument('name')
@click.argument('sma', type=click.FLOAT)
@click.argument('l_max', type=click.INT)
# @click.argument('sma', type=click.FLOAT, help='Semi-major axis [pc].')
# @click.argument('l_max', type=click.INT, help='Maximum degree of spherical harmonics to compute')
@click.option('--gamma', default=1.75, help="Slope of the cusp's density profile", show_default=True)
@click.option('--mstar', default=1.0, help="Mass of individual stars [solar mass]", show_default=True)
@click.option('--mbh', default=4.3e6, help="Massive black Hole mass [solar mass]", show_default=True)
@click.option('--rh', default=2.0, help="Radius of influence [pc]", show_default=True)
@click.option('--threads', default=1, help="Number of threads", show_default=True)
@click.option('--neval', default=1e4, help="Maximum number of evaluation pass to the Vegas integrator", show_default=True)
@click.option('--plot', is_flag=True, help="Plot the results into NAME.eps", show_default=True)
@click.option('--no_pbar', is_flag=True, help="Suppress progress bar")
@click.option('--j_grid', default=128, help="Size of the gird in j=J/J_c space. The grid is evenly spaced on log "
                                            "scale between J_lc/J_c and 1", show_default=True)
def main(name, sma, l_max, gamma, mstar, mbh, rh, threads, neval, plot,
         no_pbar, j_grid):
    drr = DRR(sma, gamma=gamma, mbh_mass=mbh, star_mass=mstar, rh=rh, j_grid_size=j_grid)

    d_rr, d_err = drr(l_max, threads=threads, neval=neval, progress_bar=not no_pbar)
    drr.save(name + '.hdf5')
    if plot:
        import matplotlib.pyplot as plt
        plt.errorbar(drr.j, d_rr, d_err, fmt='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$J/J_\mathrm{c}$')
        plt.ylabel(r'$D^{\mathrm{RR}}_{JJ}/J_\mathrm{c}^2$ [1/Myr]')
        plt.savefig(name + '.eps')

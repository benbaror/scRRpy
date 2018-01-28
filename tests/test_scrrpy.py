
from click.testing import CliRunner

from scrrpy.cli import main
from scrrpy.drr import DRR
import numpy as np


# def test_main():
#     runner = CliRunner()
#     result = runner.invoke(main, [])

#    assert result.output == '()\n'
#    assert result.exit_code == 0


def test_drr():
    drr = DRR(0.1)
    d = drr._drr_lnnp(1, 1, 1, 1e3)[0]
    print(d.min())
    assert (d > 0).all(), d.min()


def test_g(tol=1e-3):
    j = np.logspace(-2,0,1000)[:-1]

    #gamma = 2
    drr = DRR(0.1, gamma=2)
    g = drr._g(j)
    g2 = j/(1+j)
    assert abs(1 - g/g2).max() < tol

    #gamma = 1
    drr = DRR(0.1, gamma=1)
    g = drr._g(j)
    g2 = j
    assert abs(1 - g/g2).max() < tol

    #gamma = 1.75
    drr = DRR(0.1, gamma=1.75)
    g = drr._g(0.999999)
    g2 = (3 - drr.gamma)/2
    assert abs(1 - g/g2) < tol


def test_gp(tol=1e-3):
    j = np.logspace(-2,0,1000)[:-1]

    #gamma = 2
    drr = DRR(0.1, gamma=2)
    gp = drr._gp(j)
    gp2 = 1/(1+j)**2
    assert abs(1 - gp/gp2).max() < tol

    #gamma = 1
    drr = DRR(0.1, gamma=1)
    gp = drr._gp(j)
    gp2 = 1.0
    assert abs(1 - gp/gp2).max() < tol


from click.testing import CliRunner

from scrrpy.cli import main
from scrrpy.drr import DRR


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])

    assert result.output == '()\n'
    assert result.exit_code == 0


def test_drr():
    drr = DRR(0.1)
    d = drr._drr_lnnp(1, 1, 1, 1e3)[0]
    print(d.min())
    assert (d > 0).all(), d.min()

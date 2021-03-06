from click.testing import CliRunner

from scrrpy.cli import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, ["test", "0.01", "2",
                                  "--mbh=4.3e6",
                                  "--gamma=1.75",
                                  "--rh=2.0",
                                  "--mstar=1.0",
                                  "--neval=1e3",
                                  "--j_grid=32",
                                  "--no_pbar"])
    assert result.output == '', result.output
    assert result.exit_code == 0

def test_main_plot():
    runner = CliRunner()
    result = runner.invoke(main, ["test", "0.01", "2",
                                  "--mbh=4.3e6",
                                  "--gamma=1.75",
                                  "--rh=2.0",
                                  "--mstar=1.0",
                                  "--neval=1e3",
                                  "--plot",
                                  "--j_grid=32",
                                  "--no_pbar"])
    assert result.output == '', result.output
    assert result.exit_code == 0

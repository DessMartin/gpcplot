import numpy as np
import matplotlib
from gpcplot.pipeline import GPCPlotPipeline, Options

def test_minor_ticks_are_2_to_9(tmp_path):
    # Build a trivial plot with defaults
    f = tmp_path / "a.txt"
    f.write_text("100 0.0\n1000 1.0\n10000 0.0\n")
    pipe = GPCPlotPipeline()
    fig, ax = pipe.plot_from_files([str(f)], Options())
    loc = ax.xaxis.get_minor_locator()
    # LogLocator with integer subs
    assert isinstance(loc, matplotlib.ticker.LogLocator)
    subs = loc.subs
    # subs can be a list-like
    assert all(int(s) in range(2,10) for s in subs)
    matplotlib.pyplot.close(fig)

def test_xmax_respects_data_when_no_cap(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("1e3 0.0\n5e6 1.0\n")
    pipe = GPCPlotPipeline()
    opts = Options(xmax="auto", xmax_cap=None)  # remove cap
    fig, ax = pipe.plot_from_files([str(f)], opts)
    assert ax.get_xlim()[1] >= 5e6
    matplotlib.pyplot.close(fig)

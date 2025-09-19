import numpy as np
from gpcplot.core import parse_up_to_three_numeric_cols, resolve_output, smooth_log10M

def test_resolve_output_windows_hint(tmp_path):
    out = tmp_path / "out\\"
    d, base = resolve_output(str(out))
    assert base == "gpc_overlay"
    assert d.exists()

def test_parse_three_cols(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("1000 0.1 1\n1000 0.1 1\n2000 0.2 2\n")  # duplicate x
    x, y, c = parse_up_to_three_numeric_cols(str(f))
    assert np.all(np.diff(x) > 0)
    assert c is not None

def test_log_smoothing_identity_when_zero():
    M = np.geomspace(1e2, 1e5, 200)
    y = np.sin(np.log10(M))
    y0 = smooth_log10M(M, y, 0.0)
    assert np.allclose(y0, y)

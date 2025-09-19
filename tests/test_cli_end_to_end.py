import sys, shutil
from pathlib import Path
from gpcplot.cli import main

def test_cli_saves_outputs(tmp_path, monkeypatch):
    a = tmp_path / "a.txt"
    a.write_text("1000 0.0\n2000 1.0\n")
    outdir = tmp_path / "plots/"
    argv = [str(a), "-o", str(outdir), "--out-format", "png"]
    monkeypatch.setenv("MPLBACKEND", "Agg")
    main(argv)
    files = list(outdir.glob("gpc_overlay.png"))
    assert len(files) == 1 and files[0].exists()

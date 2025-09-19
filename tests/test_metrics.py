import numpy as np
from gpcplot.metrics import to_mass_fractions_log10M, moments_from_mass_fractions

def test_moments_basic():
    M = np.logspace(2,4,50); y = np.exp(-((np.log10(M)-3.0)**2)/(2*0.2**2))
    Mi, wi = to_mass_fractions_log10M(M, y)
    stats = moments_from_mass_fractions(Mi, wi)
    assert np.isfinite(stats["Mw"]) and stats["Mw"] > 0

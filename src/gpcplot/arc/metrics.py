import numpy as np

def to_mass_fractions_log10M(M, Wlog10M):
    M = np.asarray(M, float); y = np.asarray(Wlog10M, float)
    L10 = np.log10(M); dL10 = np.diff(L10)
    if np.any(dL10 <= 0):
        idx = np.argsort(M); M, y, L10 = M[idx], y[idx], L10[idx]
        dL10 = np.diff(L10); keep = np.r_[True, dL10 > 0]
        M, y, L10 = M[keep], y[keep], L10[keep]; dL10 = np.diff(L10)
    if len(M) < 2: return np.array([]), np.array([])
    wi = 0.5*(y[:-1] + y[1:]) * dL10
    Mi = np.sqrt(M[:-1]*M[1:])
    W = wi.sum()
    if not np.isfinite(W) or W <= 0: return Mi, wi
    return Mi, wi / W

def moments_from_mass_fractions(Mi, wi):
    Mi = np.asarray(Mi, float); wi = np.asarray(wi, float)
    s_w = wi.sum()
    if s_w > 0 and not np.isclose(s_w, 1.0): wi = wi / s_w
    Mi_safe = np.clip(Mi, 1e-300, None)
    Mn_inv = np.sum(wi / Mi_safe); Mn = 1.0 / Mn_inv if Mn_inv > 0 else np.nan
    Mw = np.sum(wi * Mi); Mz = np.sum(wi * Mi**2) / Mw if Mw > 0 else np.nan
    PDI = Mw / Mn if (Mn > 0 and np.isfinite(Mn)) else np.nan
    return {"Mn": Mn, "Mw": Mw, "Mz": Mz, "PDI": PDI}

def mp_mode_from_curve(M, y_smooth):
    if len(M) == 0: return np.nan
    return float(M[int(np.nanargmax(y_smooth))])

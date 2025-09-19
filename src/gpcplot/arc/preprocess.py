import numpy as np

def baseline_min(y):
    y2 = y - np.nanmin(y); y2[y2 < 0] = 0.0; return y2

def baseline_poly_logM(M, y, deg=2):
    lnM = np.log(M); c = np.polyfit(lnM, y, deg); b = np.polyval(c, lnM)
    z = y - b; z[z < 0] = 0.0; return z

def baseline_asls(y, lam=1e5, p=1e-3, niter=10):
    y = np.asarray(y, float); L = len(y)
    if L < 3: return baseline_min(y)
    D = np.diff(np.eye(L), 2); w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = np.linalg.solve(W + lam * (D.T @ D), w * y)
        w = p * (y > Z) + (1 - p) * (y <= Z)
    z = y - Z; z[z < 0] = 0.0; return z

def _gaussian_kernel(sigma_pts: float):
    if sigma_pts <= 0: k = np.zeros(1); k[0]=1.0; return k
    rad = max(1, int(3 * sigma_pts))
    xs = np.arange(-rad, rad + 1, dtype=float)
    k = np.exp(-(xs**2) / (2 * sigma_pts * sigma_pts)); k /= k.sum(); return k

def smooth_index(y, sigma_pts: float):
    return np.convolve(y, _gaussian_kernel(sigma_pts), mode="same")

def resample_log10M(M, y, points=None):
    if points is None: points = len(M)
    L10 = np.log10(M); L10u = np.linspace(L10.min(), L10.max(), points)
    yu = np.interp(L10u, L10, y); Mu = 10**L10u; return Mu, yu, L10u

def smooth_log10M(M, y, sigma_pts: float):
    if sigma_pts <= 0: return y
    M_u, y_u, L10u = resample_log10M(M, y, points=len(y))
    y_s = np.convolve(y_u, _gaussian_kernel(sigma_pts), mode="same")
    return np.interp(np.log10(M), L10u, y_s)

def normalize_curve(M, y, mode):
    if mode == "none": return y
    if mode == "log10M": A = np.trapezoid(y, x=np.log10(M))
    elif mode == "linearM": A = np.trapezoid(y, x=M)
    else: A = None
    if A is None or not np.isfinite(A) or A <= 0:
        print("[WARN] normalization area <= 0 or non-finite; leaving curve unnormalized.")
        return y
    return y / A

def parse_threshold_to_fraction(s: str) -> float:
    s = str(s).strip()
    if s.endswith("%"): v = float(s[:-1].strip()) / 100.0
    else:
        v = float(s); v = v/100.0 if v > 1.0 else v
    return max(0.0, min(1.0, v))

def crop_with_cumulative(x, y, cum, thr_frac: float):
    if cum is None or not np.any(np.isfinite(cum)): return x, y
    c = cum.copy()
    if np.nanmax(c) > 1.5: c = c / 100.0
    m = c >= thr_frac
    if not np.any(m): return x, y
    i0 = int(np.argmax(m)); return x[i0:], y[i0:]

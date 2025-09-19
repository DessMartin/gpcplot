from __future__ import annotations
import argparse, pathlib, re, os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator, Formatter
from cycler import cycler

# ---------- Tick formatter ----------
class OneTimesLogFormatter(Formatter):
    def __call__(self, x, pos=None):
        if x <= 0: return ""
        exp = int(np.round(np.log10(x)))
        return r"$1\times10^{%d}$" % exp if np.isclose(x, 10**exp, rtol=0, atol=1e-12*10**exp) else ""

# ---------- Regex ----------
_num_pat = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")

# ---------- I/O helpers ----------
def expand_inputs(inputs, dir_glob: str):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, dir_glob))))
        else:
            files.extend(sorted(glob.glob(item)))
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    return unique

def resolve_output(output_arg: str):
    """
    Interpret --output as either a directory or a basename path.
    Robust to Windows: treat trailing '/' or '\\' as dir hint.
    """
    p = pathlib.Path(output_arg)
    dir_trailer = str(output_arg).endswith(("/", "\\"))  # <-- Windows backslash supported
    is_dir_hint = dir_trailer or (p.exists() and p.is_dir())
    if is_dir_hint:
        outdir = p
        basename = "gpc_overlay"
    else:
        if p.suffix.lower() in {".png", ".pdf", ".svg"}:
            basename = p.stem
            outdir = p.parent if str(p.parent) != "" else pathlib.Path.cwd()
        else:
            basename = p.name
            outdir = p.parent if str(p.parent) != "" else pathlib.Path.cwd()
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Warning: no permission to write to '{outdir}'. Falling back to CWD.")
        outdir = pathlib.Path.cwd()
    return outdir, basename

def safe_save(fig, path: pathlib.Path, dpi: int, tight: bool, transparent: bool, bg: str):
    bbox = "tight" if tight else None
    if not transparent:
        if bg == "white":
            fig.patch.set_alpha(1.0)
            fig.patch.set_facecolor("white")
            for ax in fig.axes:
                ax.set_facecolor("white")
    try:
        fig.savefig(str(path), dpi=dpi, bbox_inches=bbox, transparent=transparent)
        print(f"Saved: {path.resolve()}")
    except PermissionError:
        alt = path.with_name(path.stem + "_copy" + path.suffix)
        fig.savefig(str(alt), dpi=dpi, bbox_inches=bbox, transparent=transparent)
        print(f"Warning: Permission denied for '{path}'. Saved to '{alt}'.")

# ---------- Parsing ----------
def parse_up_to_three_numeric_cols(path: str):
    xs, ys, cs = [], [], []
    bad = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip().replace(",", " ").replace(";", " ")
            nums = _num_pat.findall(s)
            if len(nums) >= 2:
                try:
                    x = float(nums[0]); y = float(nums[1])
                    c = float(nums[2]) if len(nums) >= 3 else np.nan
                    if np.isfinite(x) and np.isfinite(y):
                        xs.append(x); ys.append(y); cs.append(c)
                except Exception:
                    bad += 1
            elif s:
                bad += 1
    if bad:
        print(f"[WARN] {path}: ignored {bad} non-numeric/ill-formed lines.")
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    c = np.asarray(cs, float) if len(cs) == len(xs) else None
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    x, y = x[m], y[m]; c = c[m] if c is not None else None
    if x.size == 0:
        raise ValueError(f"No numeric data read from {path}")
    idx = np.argsort(x); x, y = x[idx], y[idx]; c = c[idx] if c is not None else None
    dx = np.diff(x)
    if np.any(dx <= 0):
        dup = int(np.sum(dx == 0))
        print(f"[WARN] {path}: x contained non-increasing points (duplicates: {dup}).")
        keep = np.r_[True, dx > 0]
        x, y = x[keep], y[keep]; c = c[keep] if c is not None else None
    return x, y, c

# ---------- Baselines ----------
def baseline_min(y):
    y2 = y - np.nanmin(y); y2[y2 < 0] = 0.0; return y2

def baseline_poly_logM(M, y, deg=2):
    lnM = np.log(M)
    c = np.polyfit(lnM, y, deg)
    b = np.polyval(c, lnM)
    z = y - b; z[z < 0] = 0.0
    return z

def baseline_asls(y, lam=1e5, p=1e-3, niter=10):
    y = np.asarray(y, float)
    L = len(y)
    if L < 3: return baseline_min(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = np.linalg.solve(W + lam * (D.T @ D), w * y)
        w = p * (y > Z) + (1 - p) * (y <= Z)
    z = y - Z
    z[z < 0] = 0.0
    return z

# ---------- Smoothing ----------
def _gaussian_kernel(sigma_pts: float):
    if sigma_pts <= 0:
        k = np.zeros(1); k[0] = 1.0
        return k
    rad = max(1, int(3 * sigma_pts))
    xs = np.arange(-rad, rad + 1, dtype=float)
    k = np.exp(-(xs**2) / (2 * sigma_pts * sigma_pts))
    k /= k.sum()
    return k

def smooth_index(y, sigma_pts: float):
    return np.convolve(y, _gaussian_kernel(sigma_pts), mode="same")

def resample_log10M(M, y, points=None):
    if points is None: points = len(M)
    L10 = np.log10(M)
    L10u = np.linspace(L10.min(), L10.max(), points)
    yu = np.interp(L10u, L10, y)
    Mu = 10**L10u
    return Mu, yu, L10u

def smooth_log10M(M, y, sigma_pts: float):
    if sigma_pts <= 0: return y
    M_u, y_u, L10u = resample_log10M(M, y, points=len(y))
    y_s = np.convolve(y_u, _gaussian_kernel(sigma_pts), mode="same")
    return np.interp(np.log10(M), L10u, y_s)

# ---------- Normalization ----------
def normalize_curve(M, y, mode):
    if mode == "none": return y
    if mode == "log10M":
        A = np.trapezoid(y, x=np.log10(M))
    elif mode == "linearM":
        A = np.trapezoid(y, x=M)
    else:
        A = None
    if A is None or not np.isfinite(A) or A <= 0:
        print("[WARN] normalization area <= 0 or non-finite; leaving curve unnormalized.")
        return y
    return y / A

# ---------- Cumulative cropping ----------
def parse_threshold_to_fraction(s: str) -> float:
    s = str(s).strip()
    if s.endswith("%"):
        v = float(s[:-1].strip()) / 100.0
    else:
        v = float(s)
        v = v/100.0 if v > 1.0 else v
    return max(0.0, min(1.0, v))

def crop_with_cumulative(x, y, cum, thr_frac: float):
    if cum is None or not np.any(np.isfinite(cum)): return x, y
    c = cum.copy()
    if np.nanmax(c) > 1.5:
        c = c / 100.0
    m = c >= thr_frac
    if not np.any(m): return x, y
    i0 = int(np.argmax(m))
    return x[i0:], y[i0:]

# ---------- Colors & linestyles ----------
DEFAULT_COLORS6 = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00", "#CC79A7"]

def _repeat_to_length(seq, n):
    if not seq: return []
    reps = (n + len(seq) - 1) // len(seq)
    return list((seq * reps)[:n])

def _parse_color_override(colors_arg, n):
    if colors_arg is None: return None
    cols = [c.strip() for c in str(colors_arg).split(",") if c.strip()]
    if not cols: return None
    return _repeat_to_length(cols, n)

def _parse_linestyle_override(linestyles_arg, n):
    if linestyles_arg is None: return None
    styles = [s.strip() for s in str(linestyles_arg).split(",") if s.strip()]
    if not styles: return None
    return _repeat_to_length(styles, n)

def _palette_colors(palette, n):
    if n <= 0: return []
    p = (palette or "").lower()
    if p in ("auto6","default6"):
        return _repeat_to_length(DEFAULT_COLORS6, n)
    if p == "mono":
        return _repeat_to_length(["#000000"], n)
    if p in ("tab10","tableau"):
        cmap = plt.get_cmap("tab10");  return [cmap(i % cmap.N) for i in range(n)]
    if p == "set2":
        cmap = plt.get_cmap("Set2");   return [cmap(i % cmap.N) for i in range(n)]
    if p == "dark2":
        cmap = plt.get_cmap("Dark2");  return [cmap(i % cmap.N) for i in range(n)]
    if p == "viridis":
        cmap = plt.get_cmap("viridis");return [cmap(i / max(1, n-1)) for i in range(n)]
    if p == "colorblind":
        cols = ["#0072B2","#E69F00","#009E73","#56B4E9","#D55E00","#CC79A7","#F0E442","#999999"]
        return _repeat_to_length(cols, n)
    return _repeat_to_length(DEFAULT_COLORS6, n)

def build_line_cycler(n, palette=None, colors_arg=None, linestyles_arg=None):
    user_colors     = _parse_color_override(colors_arg, n)
    user_linestyles = _parse_linestyle_override(linestyles_arg, n)
    if user_colors is not None:
        colors = user_colors
    else:
        colors = _palette_colors(palette, n)
        if not colors:
            colors = _repeat_to_length(DEFAULT_COLORS6, n)
    cyc = cycler(color=colors)
    if user_linestyles:
        cyc = cyc + cycler(linestyle=user_linestyles)
    return cyc

# ---------- Figure sizing ----------
def parse_figsize_ratio(width, height, figsize_str, ratio_str):
    if figsize_str:
        s = figsize_str.lower().replace("×","x").replace(" ","")
        if "x" in s:
            w, h = s.split("x", 1); return float(w), float(h)
    if ratio_str:
        r = ratio_str.replace("×",":").replace(" ","")
        if ":" in r:
            a, b = r.split(":",1); R = float(a)/float(b)
        else:
            R = float(r)
        return float(width), float(width)/R
    return float(width), float(height)

# ---------- Moments & mass fractions ----------
def to_mass_fractions_log10M(M, Wlog10M):
    M = np.asarray(M, float); y = np.asarray(Wlog10M, float)
    L10 = np.log10(M)
    dL10 = np.diff(L10)
    if np.any(dL10 <= 0):
        idx = np.argsort(M)
        M, y, L10 = M[idx], y[idx], L10[idx]
        dL10 = np.diff(L10)
        keep = np.r_[True, dL10 > 0]
        M, y, L10 = M[keep], y[keep], L10[keep]
        dL10 = np.diff(L10)
    if len(M) < 2:
        return np.array([]), np.array([])
    wi = 0.5*(y[:-1] + y[1:]) * dL10
    Mi = np.sqrt(M[:-1]*M[1:])
    W = wi.sum()
    if not np.isfinite(W) or W <= 0:
        return Mi, wi
    return Mi, wi / W

def moments_from_mass_fractions(Mi, wi):
    Mi = np.asarray(Mi, float); wi = np.asarray(wi, float)
    s_w = wi.sum()
    if s_w > 0 and not np.isclose(s_w, 1.0):
        wi = wi / s_w
    Mi_safe = np.clip(Mi, 1e-300, None)
    Mn_inv = np.sum(wi / Mi_safe)
    Mn = 1.0 / Mn_inv if Mn_inv > 0 else np.nan
    Mw = np.sum(wi * Mi)
    Mz = np.sum(wi * Mi**2) / Mw if Mw > 0 else np.nan
    PDI = Mw / Mn if (Mn > 0 and np.isfinite(Mn)) else np.nan
    return {"Mn": Mn, "Mw": Mw, "Mz": Mz, "PDI": PDI}

def mp_mode_from_curve(M, y_smooth):
    if len(M) == 0: return np.nan
    i = int(np.nanargmax(y_smooth))
    return float(M[i])

# ---------- Themes ----------
def apply_theme(theme, base_fontsize, font_family=None):
    rc = {
        "savefig.dpi": 600,
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "legend.fontsize": max(8.0, base_fontsize-1),
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "axes.linewidth": 0.8,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.default": "regular",
        "mathtext.fontset": "dejavusans",
    }
    if font_family:
        rc["font.family"] = font_family
    if theme in ("nature","science","acs","plain"):
        rc.update({
            "axes.grid": False,
            "xtick.direction": "in", "ytick.direction": "in",
        })
    plt.rcParams.update(rc)

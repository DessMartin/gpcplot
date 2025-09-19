from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator, Formatter
from cycler import cycler

# --- colors/linestyles (Okabe–Ito defaults, solid-only unless user asks) ---
DEFAULT_COLORS6 = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00", "#CC79A7"]

def _repeat_to_n(seq, n):
    reps = (n + len(seq) - 1) // len(seq); return (seq * reps)[:n]

def _palette_colors(n, name):
    name = (name or "").lower()
    if name in ("auto6","default6"): return _repeat_to_n(DEFAULT_COLORS6, n)
    if name == "colorblind":
        base = ["#0072B2","#E69F00","#009E73","#56B4E9","#D55E00","#CC79A7","#F0E442","#999999"]
        return _repeat_to_n(base, n)
    if name in ("tab10","tableau"):
        cmap = plt.get_cmap("tab10"); return [cmap(i % 10) for i in range(n)]
    if name == "set2":
        cmap = plt.get_cmap("Set2"); return [cmap(i % cmap.N) for i in range(n)]
    if name == "dark2":
        cmap = plt.get_cmap("Dark2"); return [cmap(i % cmap.N) for i in range(n)]
    if name == "viridis":
        cmap = plt.get_cmap("viridis"); return [cmap(i / max(1, n-1)) for i in range(n)]
    if name == "mono":
        return _repeat_to_n(["#000000"], n)
    return _repeat_to_n(DEFAULT_COLORS6, n)

def build_line_cycler(n, palette=None, colors_arg=None, linestyles_arg=None):
    # solid-first policy: only add linestyles if user asked
    if colors_arg:
        cols = [c.strip() for c in str(colors_arg).split(",") if c.strip()]
    else:
        cols = _palette_colors(n, palette or "auto6")
    if linestyles_arg:
        ls = [s.strip() for s in str(linestyles_arg).split(",") if s.strip()]
        return cycler(color=_repeat_to_n(cols, n)) + cycler(linestyle=_repeat_to_n(ls, n))
    return cycler(color=_repeat_to_n(cols, n))

# --- tick formatter ---
class OneTimesLogFormatter(Formatter):
    def __call__(self, x, pos=None):
        if x <= 0: return ""
        exp = int(np.round(np.log10(x)))
        return r"$1\times10^{%d}$" % exp if np.isclose(x, 10**exp, rtol=0, atol=1e-12*10**exp) else ""

# --- theme + axes config + saving ---
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
        "lines.linestyle": "solid",
        "lines.dash_capstyle": "round",
    }
    if font_family: rc["font.family"] = font_family
    if theme in ("nature","science","acs","plain"):
        rc.update({"axes.grid": False, "xtick.direction": "in", "ytick.direction": "in"})
    plt.rcParams.update(rc)

def configure_axes(ax, args):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_major_formatter(OneTimesLogFormatter())
    if args.minor_ticks == "2-9":
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10), numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=False, right=False, length=4, width=0.8)
    ax.tick_params(which="minor", length=2, width=0.8, color="0.5")
    for s in ax.spines.values(): s.set_linewidth(0.8)
    if args.grid: ax.grid(True, which="both", alpha=0.25, linestyle="-", linewidth=0.6)
    ax.set_xlabel(r"Molecular weight, $M$ (g mol$^{-1}$)")
    ax.set_ylabel(args.ylabel)

def _plim(s):
    try: return float(s)
    except: return None

def apply_limits_legend_and_dp(ax, fig, args, all_xmins, all_xmaxs):
    xmin_in = _plim(args.xmin); xmax_in = _plim(args.xmax)
    xmin_auto = max(1.0, min(all_xmins)) if all_xmins else 1.0
    xmax_auto = max(all_xmaxs) if all_xmaxs else 1e7
    xmin = max(xmin_in if (xmin_in and xmin_in > 0) else xmin_auto, args.xmin_floor)
    xmax = min(xmax_in if (xmax_in and xmax_in > 0) else xmax_auto, args.xmax_cap or xmax_auto)
    xmin = 10**np.floor(np.log10(xmin))
    ax.set_xlim(xmin, xmax)

    # DP axis
    if args.monomer_mw and args.monomer_mw > 0:
        def M_to_DP(M): return np.asarray(M)/args.monomer_mw
        def DP_to_M(DP): return np.asarray(DP)*args.monomer_mw
        sec = ax.secondary_xaxis("top", functions=(M_to_DP, DP_to_M))
        sec.set_xlabel("Degree of polymerization, $\\mathrm{DP}=M/M_0$")
        sec.tick_params(labelsize=args.fontsize)

    # legend
    legend_kwargs = dict(frameon=args.legend_box, ncol=args.legend_ncols,
                         handlelength=1.8, handletextpad=0.6, columnspacing=0.8)
    if args.legend_box:
        legend_kwargs.update(dict(framealpha=args.legend_alpha,
                                  facecolor=args.legend_facecolor,
                                  edgecolor=args.legend_edgecolor,
                                  fancybox=True))
    if args.legend == "outside-right":
        leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), **legend_kwargs)
    else:
        if args.legend_xy is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(args.legend_xy)
        leg = ax.legend(loc=args.legend_loc, **legend_kwargs)
    if args.legend_box and leg is not None:
        fr = leg.get_frame(); fr.set_linewidth(0.8)
        try: fr.set_boxstyle("round,pad=0.25,rounding_size=0.15")
        except Exception: pass

def save_figure(fig, path, dpi: int, tight: bool, transparent: bool, bg: str):
    bbox = "tight" if tight else None
    if not transparent and bg == "white":
        fig.patch.set_alpha(1.0); fig.patch.set_facecolor("white")
        for ax in fig.axes: ax.set_facecolor("white")
    fig.savefig(str(path), dpi=dpi, bbox_inches=bbox, transparent=transparent)
    print(f"Saved: {path}")

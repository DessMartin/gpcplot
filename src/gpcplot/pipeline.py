from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import pathlib, contextlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator
try:
    from importlib.resources import files, as_file
except ImportError:  # py<3.9
    from importlib_resources import files, as_file

from .core import (
    OneTimesLogFormatter, expand_inputs, resolve_output, safe_save,
    parse_up_to_three_numeric_cols, baseline_min, baseline_poly_logM, baseline_asls,
    smooth_log10M, smooth_index, normalize_curve, parse_threshold_to_fraction,
    crop_with_cumulative, build_line_cycler, parse_figsize_ratio,
    to_mass_fractions_log10M, moments_from_mass_fractions, mp_mode_from_curve,
    apply_theme
)

@dataclass
class Options:
    # I/O
    output: str = "gpc_overlay"
    out_format: str = "pdf,svg,png"
    dpi: int = 600
    bbox_tight: bool = False
    transparent: bool = False
    bg: str = "auto"
    glob: str = "*.txt"
    # Sizing
    width: float = 8.0
    height: float = 5.0
    figsize: Optional[str] = None
    ratio: Optional[str] = None
    # Processing
    smooth: float = 2.0
    smooth_domain: str = "log10M"
    baseline_mode: str = "min"               # ["min","asls","poly"]
    baseline_poly_deg: int = 2
    asls_lam: float = 1e5
    asls_p: float = 1e-3
    normalize: str = "log10M"                # ["log10M","linearM","none"]
    xmin: str = "auto"
    xmin_floor: float = 1e2
    xmax: str = "auto"
    xmax_cap: Optional[float] = 2e6          # None => truly no cap
    cum_threshold: Optional[str] = None
    # Labels / fonts / axes
    ylabel: str = r"W(log$_{10}$M)"
    font: Optional[str] = None
    fontsize: float = 11.0
    theme: str = "plain"
    grid: bool = False
    minor_ticks: str = "2-9"                 # ensure 2..9 (not 0.2..0.9)
    # Colors / styles
    palette: str = "auto6"
    colors: Optional[str] = None
    linestyles: Optional[str] = None
    linewidth: float = 2.0
    labels: Optional[Iterable[str]] = None   # fileStem=Label ...
    order: Optional[str] = None              # comma list of basenames
    # Legend
    legend: str = "inside"                   # ["inside","outside-right"]
    legend_loc: str = "upper left"
    legend_xy: Optional[Iterable[float]] = None
    legend_ncols: int = 1
    legend_box: bool = False
    legend_facecolor: str = "white"
    legend_edgecolor: str = "grey"
    legend_alpha: float = 1.0
    legend_stats: bool = False
    # Markers / DP axis
    markers: bool = False
    monomer_mw: Optional[float] = None
    # Reporting & diagnostics
    report: Optional[str] = None
    diagnostic: bool = False
    diagnostic_shadow: bool = False

class GPCPlotPipeline:
    """Notebook-friendly orchestrator; used by the CLI as well."""

    def __init__(self, use_builtin_style: bool = True):
        self._style_ctx = self._style_context() if use_builtin_style else contextlib.nullcontext()

    @contextlib.contextmanager
    def _style_context(self):
        # Apply bundled mplstyle early
        res = files("gpcplot").joinpath("_styles/gpcplot.mplstyle")
        with as_file(res) as p:
            with plt.style.context(str(p)):
                yield

    def plot_from_files(self, inputs: List[str], opts: Options):
        with self._style_ctx:
            # Theme/rc
            W, H = parse_figsize_ratio(opts.width, opts.height, opts.figsize, opts.ratio)
            apply_theme(opts.theme, opts.fontsize, opts.font)

            fig, ax = plt.subplots(figsize=(W, H), layout='constrained')

            files = expand_inputs(inputs, opts.glob)
            if not files:
                raise SystemExit("No input files found.")

            # line color/style cycle
            ax.set_prop_cycle(build_line_cycler(
                n=len(files),
                palette=opts.palette,
                colors_arg=opts.colors,
                linestyles_arg=opts.linestyles
            ))
            plt.rcParams["lines.linestyle"] = "solid"
            plt.rcParams["lines.dash_capstyle"] = "round"

            thr = parse_threshold_to_fraction(opts.cum_threshold) if opts.cum_threshold is not None else None

            label_map = {}
            if opts.labels:
                for kv in opts.labels:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        label_map[k.strip()] = v.strip()

            if opts.order:
                desired = [s.strip() for s in opts.order.split(",") if s.strip()]
                files = sorted(files, key=lambda f: (desired.index(pathlib.Path(f).stem)
                           if pathlib.Path(f).stem in desired else 1e9, pathlib.Path(f).stem))

            # Axes setup (log x)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.xaxis.set_major_formatter(OneTimesLogFormatter())
            if opts.minor_ticks == "2-9":
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10), numticks=100))  # <-- 2..9
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_minor_formatter(NullFormatter())

            ax.tick_params(which="both", direction="in", top=False, right=False, length=4, width=0.8)
            ax.tick_params(which="minor", length=2, width=0.8, color="0.5")
            for s in ax.spines.values(): s.set_linewidth(0.8)

            ax.grid(opts.grid, which="both", alpha=0.25 if opts.grid else 1.0, linestyle="-", linewidth=0.6 if opts.grid else 0.6)

            ax.set_xlabel(r"Molecular weight, $M$ (g mol$^{-1}$)")
            ax.set_ylabel(opts.ylabel)

            all_xmins, all_xmaxs = [], []
            for f in files:
                try:
                    x, y, cum = parse_up_to_three_numeric_cols(f)
                    if thr is not None:
                        x, y = crop_with_cumulative(x, y, cum, thr)

                    y_raw = y.copy()
                    if opts.baseline_mode == "min":
                        yb = baseline_min(y)
                    elif opts.baseline_mode == "poly":
                        yb = baseline_poly_logM(x, y, deg=opts.baseline_poly_deg)
                    else:
                        yb = baseline_asls(y, lam=opts.asls_lam, p=opts.asls_p)

                    if opts.smooth > 0:
                        y_s = smooth_log10M(x, yb, opts.smooth) if opts.smooth_domain == "log10M" else smooth_index(yb, opts.smooth)
                    else:
                        y_s = yb

                    y_f = normalize_curve(x, y_s, opts.normalize)

                    if opts.diagnostic:
                        A_log10 = np.trapezoid(y_s, x=np.log10(x))
                        A_lin   = np.trapezoid(y_s, x=x)
                        print(f"[DIAG] {pathlib.Path(f).name}: area(log10M)={A_log10:.6g}, area(M)={A_lin:.6g}, "
                              f"min(y)={np.nanmin(y_raw):.6g}, max(y)={np.nanmax(y_raw):.6g}")

                    Mp = mp_mode_from_curve(x, y_f)
                    Mi, wi = to_mass_fractions_log10M(x, y_f)
                    stats = moments_from_mass_fractions(Mi, wi) if len(Mi) else {"Mn":np.nan,"Mw":np.nan,"Mz":np.nan,"PDI":np.nan}

                    stem = pathlib.Path(f).stem
                    lab = label_map.get(stem, stem)
                    if opts.legend_stats:
                        def _abbr_k(v): return f"{v/1e3:.0f}k" if (v and np.isfinite(v)) else "NA"
                        lab = f"{lab} — Mw={_abbr_k(stats['Mw'])}, PDI={stats['PDI']:.2f}" if np.isfinite(stats["PDI"]) else f"{lab}"

                    ax.plot(x, y_f, lw=opts.linewidth, label=lab)

                    if opts.diagnostic_shadow:
                        y_shadow = normalize_curve(x, baseline_min(y_raw), opts.normalize)
                        ax.plot(x, y_shadow, lw=max(0.6, 0.6*opts.linewidth), alpha=0.35, label=None)

                    if opts.markers:
                        for name, val in (("Mp", Mp), ("Mn", stats["Mn"]), ("Mw", stats["Mw"]), ("Mz", stats["Mz"])):
                            if val and np.isfinite(val):
                                ax.axvline(val, ymin=0.0, ymax=0.98, alpha=0.3, linewidth=1.0)
                                ymax = ax.get_ylim()[1]
                                ax.text(val, ymax, name, rotation=90, va="top", ha="right")

                    all_xmins.append(np.nanmin(x)); all_xmaxs.append(np.nanmax(x))
                except Exception as e:
                    print(f"[WARN] Skipped {f}: {e}")

            def _plim(s):
                try: return float(s)
                except: return None

            xmin_in  = _plim(opts.xmin)
            xmax_in  = _plim(opts.xmax)

            xmin_auto = max(1.0, min(all_xmins)) if all_xmins else 1.0
            xmax_auto = max(all_xmaxs) if all_xmaxs else 1e7

            xmin_candidate = xmin_in if (xmin_in is not None and xmin_in > 0) else xmin_auto
            xmax_candidate = xmax_in if (xmax_in is not None and xmax_in > 0) else xmax_auto

            xmin = max(xmin_candidate, opts.xmin_floor)
            xmax = min(xmax_candidate, opts.xmax_cap) if (opts.xmax_cap and opts.xmax_cap > 0) else xmax_candidate

            xmin = 10**np.floor(np.log10(xmin))
            ax.set_xlim(xmin, xmax)

            if opts.monomer_mw and opts.monomer_mw > 0:
                def M_to_DP(M): return np.asarray(M)/opts.monomer_mw
                def DP_to_M(DP): return np.asarray(DP)*opts.monomer_mw
                sec = ax.secondary_xaxis("top", functions=(M_to_DP, DP_to_M))
                sec.set_xlabel("Degree of polymerization, $\\mathrm{DP}=M/M_0$")

            legend_kwargs = dict(
                frameon=opts.legend_box,
                ncol=opts.legend_ncols,
                handlelength=1.8,
                handletextpad=0.6,
                columnspacing=0.8,
            )
            if opts.legend == "outside-right":
                leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), **legend_kwargs)
            else:
                inner_kwargs = dict(legend_kwargs)
                if opts.legend_xy is not None:
                    inner_kwargs["bbox_to_anchor"] = tuple(opts.legend_xy)
                leg = ax.legend(loc=opts.legend_loc, **inner_kwargs)

            if opts.legend_box and leg is not None:
                fr = leg.get_frame()
                fr.set_facecolor(opts.legend_facecolor)
                fr.set_edgecolor(opts.legend_edgecolor)
                fr.set_alpha(opts.legend_alpha)
                fr.set_linewidth(0.8)

            return fig, ax

    def plot_and_save(self, inputs: List[str], opts: Options):
        fig, ax = self.plot_from_files(inputs, opts)
        outdir, basename = resolve_output(opts.output)
        exts = [e.strip().lower() for e in opts.out_format.split(",") if e.strip()]
        for ext in exts:
            target = outdir / f"{basename}.{ext}"
            safe_save(fig, target, dpi=opts.dpi, tight=opts.bbox_tight,
                      transparent=opts.transparent, bg=opts.bg)
        plt.close(fig)

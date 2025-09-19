from __future__ import annotations
import argparse, sys
from .pipeline import GPCPlotPipeline, Options

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GPC overlay plotter (v0.6)")
    ap.add_argument("inputs", nargs="+", help="Files, GLOB patterns, or directories.")
    ap.add_argument("-o","--output", default="gpc_overlay",
                    help="Basename or directory. If a directory (or ends with '/' or '\\'), files are saved inside.")
    ap.add_argument("--out-format", default="pdf,svg,png",
                    help="Comma-separated formats to save (default 'pdf,svg,png').")
    ap.add_argument("--dpi", type=int, default=600, help="Export DPI (default 600).")
    ap.add_argument("--bbox-tight", action="store_true", help="Use bbox_inches='tight' on save (off by default).")
    ap.add_argument("--transparent", action="store_true", help="Transparent background for raster outputs.")
    ap.add_argument("--bg", choices=["auto","white","none"], default="auto",
                help="Background override when NOT using --transparent. 'white' forces a white figure/axes background; 'none' leaves rcParams as-is (default: auto).")
    # Sizing
    ap.add_argument("--width", type=float, default=8.0, help="Figure width in inches (default 8).")
    ap.add_argument("--height", type=float, default=5.0, help="Figure height in inches (default 5).")
    ap.add_argument("--figsize", default=None, help="Override width/height with 'W x H', e.g., '8x5'.")
    ap.add_argument("--ratio", default=None, help="Aspect ratio 'W:H' or float (W/H). Height computed from width.")
    # Processing
    ap.add_argument("--smooth", type=float, default=2.0, help="Gaussian sigma in sample points (0 disables).")
    ap.add_argument("--smooth-domain", choices=["log10M","index"], default="log10M", help="Domain for smoothing (default log10M).")
    ap.add_argument("--baseline-mode", choices=["min","asls","poly"], default="min", help="Baseline removal mode.")
    ap.add_argument("--baseline-poly-deg", type=int, default=2)
    ap.add_argument("--asls-lam", type=float, default=1e5)
    ap.add_argument("--asls-p", type=float, default=1e-3)
    ap.add_argument("--normalize", choices=["log10M","linearM","none"], default="log10M", help="Normalization mode (default log10M).")
    ap.add_argument("--xmin", default="auto", help="Lower x-limit or 'auto'.")
    ap.add_argument("--xmin-floor", type=float, default=1e2, help="Lower x-limit floor applied after auto/manual limits (default 1e2).")
    ap.add_argument("--xmax", default="auto", help="Upper x-limit or 'auto'.")
    ap.add_argument("--xmax-cap", type=float, default=2e6, help="Upper cap enforced after auto/manual limits (default 2e6 to respect calibration).")
    ap.add_argument("--cum-threshold", default=None, help="Trim from first point where cumulative >= threshold (percent or fraction).")
    ap.add_argument("--glob", default="*.txt", help="If a directory is provided, include files matching this pattern.")
    # Labels / fonts / axes
    ap.add_argument("--ylabel", default=r"W(log$_{10}$M)", help="Y-axis label (default 'W(log$_{10}$M)').")
    ap.add_argument("--font", default=None, help="Font family (e.g., 'Arial').")
    ap.add_argument("--fontsize", type=float, default=11.0, help="Base font size (default 11).")
    ap.add_argument("--theme", choices=["plain","nature","science","acs"], default="plain", help="Preset styling.")
    ap.add_argument("--grid", action="store_true", help="Show light grid.")
    ap.add_argument("--minor-ticks", choices=["off","2-9"], default="2-9", help="Minor ticks on log axis (default 2-9).")
    # Colors / styles
    ap.add_argument("--palette", default="auto6",
                    choices=["auto6","tab10","tableau","Set2","Dark2","viridis","colorblind","mono"],
                    help="Color palette (default 'auto6' = Okabe–Ito solids; use 'mono' for all black).")
    ap.add_argument("--colors", default=None, help="Comma-separated colors; overrides palette.")
    ap.add_argument("--linestyles", default=None, help="Comma-separated linestyles: 'solid,dashed,dashdot,dotted'.")
    ap.add_argument("--linewidth", type=float, default=2.0, help="Line width (default 2.0).")
    ap.add_argument("--labels", nargs="*", default=None, help='Label remap: fileStem=Label ...')
    ap.add_argument("--order", default=None, help='Comma-separated order of basenames.')
    # Legend
    ap.add_argument("--legend", choices=["inside","outside-right"], default="inside", help="Legend placement.")
    ap.add_argument("--legend-loc", default="upper left", help="Inside legend location key.")
    ap.add_argument("--legend-xy", nargs=2, type=float, default=None, help="Axes-fraction anchor for inside legend.")
    ap.add_argument("--legend-ncols", type=int, default=1, help="Legend columns (default 1).")
    ap.add_argument("--legend-box", action="store_true", help="Draw an opaque legend box.")
    ap.add_argument("--legend-facecolor", default="white", help="Legend box facecolor (default white).")
    ap.add_argument("--legend-edgecolor", default="grey", help="Legend box edgecolor (default grey).")
    ap.add_argument("--legend-alpha", type=float, default=1.0, help="Legend box alpha (default 1.0).")
    ap.add_argument("--legend-stats", action="store_true", help="Append Mw and PDI to legend labels.")
    # Markers / DP axis
    ap.add_argument("--markers", action="store_true", help="Draw vertical lines at Mp/Mn/Mw/Mz.")
    ap.add_argument("--monomer-mw", type=float, default=None, help="For DP axis & annotations (DP=M/M0).")
    # Reporting & diagnostics
    ap.add_argument("--report", default=None, help="CSV path to append per-trace stats.")
    ap.add_argument("--diagnostic", action="store_true", help="Print normalization area and basic stats.")
    ap.add_argument("--diagnostic-shadow", action="store_true", help="Overlay faint unsmoothed curve for comparison.")
    return ap

def main(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)

    opts = Options(
        output=args.output, out_format=args.out_format, dpi=args.dpi,
        bbox_tight=args.bbox_tight, transparent=args.transparent, bg=args.bg,
        glob=args.glob, width=args.width, height=args.height, figsize=args.figsize, ratio=args.ratio,
        smooth=args.smooth, smooth_domain=args.smooth_domain, baseline_mode=args.baseline_mode,
        baseline_poly_deg=args.baseline_poly_deg, asls_lam=args.asls_lam, asls_p=args.asls_p,
        normalize=args.normalize, xmin=args.xmin, xmin_floor=args.xmin_floor, xmax=args.xmax,
        xmax_cap=args.xmax_cap, cum_threshold=args.cum_threshold, ylabel=args.ylabel, font=args.font,
        fontsize=args.fontsize, theme=args.theme, grid=args.grid, minor_ticks=args.minor_ticks,
        palette=args.palette, colors=args.colors, linestyles=args.linestyles, linewidth=args.linewidth,
        labels=args.labels, order=args.order, legend=args.legend, legend_loc=args.legend_loc,
        legend_xy=args.legend_xy, legend_ncols=args.legend_ncols, legend_box=args.legend_box,
        legend_facecolor=args.legend_facecolor, legend_edgecolor=args.legend_edgecolor,
        legend_alpha=args.legend_alpha, legend_stats=args.legend_stats, markers=args.markers,
        monomer_mw=args.monomer_mw, report=args.report, diagnostic=args.diagnostic,
        diagnostic_shadow=args.diagnostic_shadow
    )

    pipe = GPCPlotPipeline(use_builtin_style=True)
    pipe.plot_and_save(args.inputs, opts)

if __name__ == "__main__":
    main()

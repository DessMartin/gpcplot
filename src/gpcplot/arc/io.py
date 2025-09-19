import os, glob, pathlib, re, numpy as np

_num_pat = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")

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
    p = pathlib.Path(output_arg)
    dir_trailer = str(output_arg).endswith(("/", "\\"))   # Windows-friendly
    is_dir_hint = dir_trailer or (p.exists() and p.is_dir())
    if is_dir_hint:
        outdir = p; basename = "gpc_overlay"
    else:
        if p.suffix.lower() in {".png", ".pdf", ".svg"}:
            basename, outdir = p.stem, (p.parent if str(p.parent)!="" else pathlib.Path.cwd())
        else:
            basename, outdir = p.name, (p.parent if str(p.parent)!="" else pathlib.Path.cwd())
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Warning: no permission to write to '{outdir}'. Falling back to CWD.")
        outdir = pathlib.Path.cwd()
    return outdir, basename

def parse_up_to_three_numeric_cols(path: str):
    xs, ys, cs, bad = [], [], [], 0
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
    if bad: print(f"[WARN] {path}: ignored {bad} non-numeric/ill-formed lines.")
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    c = np.asarray(cs, float) if len(cs) == len(xs) else None
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    x, y = x[m], y[m]; c = c[m] if c is not None else None
    if x.size == 0: raise ValueError(f"No numeric data read from {path}")
    idx = np.argsort(x); x, y = x[idx], y[idx]; c = c[idx] if c is not None else None
    dx = np.diff(x)
    if np.any(dx <= 0):
        dup = int(np.sum(dx == 0))
        print(f"[WARN] {path}: x contained non-increasing points (duplicates: {dup}).")
        keep = np.r_[True, dx > 0]
        x, y = x[keep], y[keep]; c = c[keep] if c is not None else None
    return x, y, c

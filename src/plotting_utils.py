import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_COUNT = 10


# ---------- extraction from an Explanation ------------------------------

def _feature_names(expl):
    names = getattr(expl, "feature_names", None)
    n = expl.values.shape[1]
    if names is None:
        return [f"Feature {i}" for i in range(n)]
    return [str(v) for v in names]


# Dark blue: ~10:1 contrast on white. Grid is decorative, kept light.
POINT_COLOUR = "#1F3A68"
CI_COLOUR = "#1F3A68"
GRID_COLOUR = "#BBBBBB"
TEXT_COLOUR = "#000000"
ARROW = "\u2193"      # direction marker drawn after each x-axis label
ARROW_SIZE = 18       # font size of the arrow, independent of the label size



def setup_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        # Arial; Liberation Sans is a metric-identical fallback where Arial is absent
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "normal",
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.color": TEXT_COLOUR,
        "axes.labelcolor": TEXT_COLOUR,
        "xtick.color": TEXT_COLOUR,
        "ytick.color": TEXT_COLOUR,
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _resolve_feature(expl, feature):
    """Accept an int index or a feature name; return (index, name)."""
    names = _feature_names(expl)
    if isinstance(feature, (int, np.integer)):
        return int(feature), names[int(feature)]
    matches = [i for i, nm in enumerate(names) if nm == str(feature)]
    if not matches:
        raise KeyError(f"{feature!r} not in feature_names")
    if len(matches) > 1:
        raise KeyError(f"{feature!r} is ambiguous: appears at {matches}")
    return matches[0], names[matches[0]]


def explanation_to_xy(expl, feature, output=None):
    """Pull (feature values, SHAP values, name) out of an Explanation."""
    idx, name = _resolve_feature(expl, feature)
    vals = np.asarray(expl.values)

    if vals.ndim == 3:
        n_feat = vals.shape[1]
        if vals.shape[2] == n_feat:
            raise ValueError(
                "3D values with shape[2] == n_features: this is either an "
                "interaction matrix or a multiclass output with as many "
                "classes as features. Slice it yourself before calling.")
        if output is None:
            raise ValueError(
                f"Multi-output explanation ({vals.shape[2]} outputs). "
                f"Pass output=<index>.")
        y = vals[:, idx, int(output)]
    elif vals.ndim == 2:
        if output is not None:
            raise ValueError("output= given but values are single-output.")
        y = vals[:, idx]
    else:
        raise ValueError(f"Unexpected values.ndim = {vals.ndim}")

    data = getattr(expl, "data", None)
    if data is None:
        raise ValueError(
            f"Explanation has no .data, so there are no feature values to "
            f"plot against. Rebuild it with the feature matrix passed in.")
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D .data, got ndim = {data.ndim}")
    x = data[:, idx]

    if x.dtype.kind in "OUS":          # strings / objects
        raise TypeError(
            f"{name!r} is non-numeric; hexbin needs a continuous x axis. "
            f"Use a suppressed group summary for categoricals instead.")

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), name


# ---------- plotting ----------------------------------------------------

def _rounded_range(v, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    step = 10.0 ** np.floor(np.log10(max(hi - lo, 1e-12) / 10))
    return np.floor(lo / step) * step, np.ceil(hi / step) * step


def shap_hexbin(expl, feature, outfile, output=None, units=None,
                gridsize=30, min_count=MIN_COUNT, dpi=150):
    x, y, name = explanation_to_xy(expl, feature, output=output)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < min_count:
        raise RuntimeError(f"{name}: only {x.size} usable records.")

    extent = (*_rounded_range(x), *_rounded_range(y))

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    hb = ax.hexbin(x, y, gridsize=gridsize, extent=extent,
                   mincnt=min_count, cmap="viridis", linewidths=0)

    counts = hb.get_array()
    counts = (counts.compressed() if np.ma.isMaskedArray(counts)
              else np.asarray(counts)[np.isfinite(counts)])
    if counts.size == 0:
        raise RuntimeError(f"{name}: every cell suppressed; lower gridsize.")
    if counts.min() < min_count:
        raise RuntimeError(f"{name}: cell with {int(counts.min())} records "
                           f"survived (min_count={min_count}).")
    shown, total = int(counts.sum()), int(x.size)

    xlab = f"{name} ({units})" if units else name
    ylab = f"SHAP value for {name}\n(impact on model output)"
    if output is not None:
        ylab = ylab.replace("model output", f"output {output}")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(f"records per hexagon (cells with <{min_count} suppressed)")

    fig.savefig(outfile, dpi=dpi, format="png")
    plt.close(fig)

    return {"feature": name, "output": output, "file": outfile,
            "shown": shown, "total": total,
            "suppressed_frac": 1 - shown / total,
            "min_cell": int(counts.min())}


def shap_hexbin_all(expl, outdir, output=None, units=None, **kw):
    """One figure per numeric feature. Returns a release log."""
    os.makedirs(outdir, exist_ok=True)
    units = units or {}
    log, skipped = [], []
    for i, name in enumerate(_feature_names(expl)):
        safe = "".join(c if c.isalnum() else "_" for c in name)
        path = os.path.join(outdir, f"shap_{i:03d}_{safe}.png")
        try:
            log.append(shap_hexbin(expl, i, path, output=output,
                                   units=units.get(name), **kw))
        except (TypeError, RuntimeError) as e:
            skipped.append((name, str(e)))
    return log, skipped


from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

MIN_COUNT = 10


def _values_2d(expl, output=None):
    v = np.asarray(expl.values)
    if v.ndim == 3:
        if output is None:
            raise ValueError(f"Multi-output ({v.shape[2]}); pass output=<i>.")
        return v[:, :, int(output)]
    if v.ndim != 2:
        raise ValueError(f"Unexpected values.ndim = {v.ndim}")
    if output is not None:
        raise ValueError("output= given but values are single-output.")
    return v


def _pct_rank(col):
    """Feature values -> [0,1] rank. Monotone, so it keeps the colour
    ordering without ever putting the raw feature scale on the figure."""
    col = np.asarray(col)
    if col.dtype.kind in "OUS":
        return None
    col = col.astype(float)
    ok = np.isfinite(col)
    out = np.full(col.shape, np.nan)
    if ok.sum() < 2:
        return out
    order = np.argsort(col[ok], kind="mergesort")
    r = np.empty(ok.sum())
    r[order] = np.arange(ok.sum())
    out[ok] = r / (ok.sum() - 1)
    return out


def shap_beeswarm_binned(expl, outfile, output=None, max_display=12,
                         nbins=40, min_count=MIN_COUNT, clip_pct=(1, 99),
                         cmap="coolwarm", dpi=150):
    vals = _values_2d(expl, output)
    n_rec, n_feat = vals.shape
    names = ([str(v) for v in expl.feature_names]
             if getattr(expl, "feature_names", None) is not None
             else [f"Feature {i}" for i in range(n_feat)])

    data = getattr(expl, "data", None)
    data = None if data is None else np.asarray(data)

    order = np.argsort(np.nanmean(np.abs(vals), axis=0))[::-1][:max_display]

    sel = vals[:, order]
    lo, hi = np.nanpercentile(sel, clip_pct)
    span = max(hi - lo, 1e-12)
    step = 10.0 ** np.floor(np.log10(span / 10))
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    edges = np.linspace(lo, hi, nbins + 1)

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.8),
                           constrained_layout=True)
    cm = plt.get_cmap(cmap)
    log, dropped = [], 0

    for row, fi in enumerate(order):
        y = vals[:, fi].astype(float)
        keep = np.isfinite(y)
        y = y[keep]
        if y.size == 0:
            continue

        # Top/bottom-code into the outer bins rather than discarding the
        # tails: pooling the outliers is itself disclosure control.
        yc = np.clip(y, edges[0], edges[-1] - 1e-12)
        idx = np.clip(np.digitize(yc, edges) - 1, 0, nbins - 1)

        colour = None
        if data is not None and data.ndim == 2:
            pr = _pct_rank(data[:, fi])
            if pr is not None:
                colour = pr[keep]

        counts = np.bincount(idx, minlength=nbins)
        vis = counts >= min_count
        dropped += int(counts[~vis].sum())
        if not vis.any():
            continue
        scale = counts[vis].max()

        for b in np.flatnonzero(vis):
            h = 0.78 * counts[b] / scale
            if colour is not None:
                cb = np.nanmean(colour[idx == b])
                fc = cm(cb) if np.isfinite(cb) else "0.6"
            else:
                fc = "0.6"
            ax.add_patch(Rectangle((edges[b], row - h / 2),
                                   edges[b + 1] - edges[b], h,
                                   facecolor=fc, edgecolor="none"))

        log.append({"feature": names[fi], "bins_shown": int(vis.sum()),
                    "records_shown": int(counts[vis].sum()),
                    "records_total": int(y.size),
                    "min_bin": int(counts[vis].min())})

    ax.axvline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(edges[0], edges[-1])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on model output)"
                  + (f", output {output}" if output is not None else "")
                  + f"\nouter bins include values beyond "
                    f"[{clip_pct[0]}th, {clip_pct[1]}th] percentile")
    ax.set_title(f"Bar height = records per bin; bins with <{min_count} "
                 f"records suppressed", fontsize=9, loc="left")

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cm)
    fig.colorbar(sm, ax=ax, pad=0.02).set_label(
        "feature value (percentile rank, bin mean)\nlow \u2192 high")

    fig.savefig(outfile, dpi=dpi, format="png")
    plt.close(fig)
    return log, {"records_in_suppressed_bins": dropped}
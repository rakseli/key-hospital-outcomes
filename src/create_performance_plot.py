#!/usr/bin/env python3
"""
Plot estimator metrics with confidence intervals as an accessible grid.

Expected JSON format (same as before):
{
  "SomeEstimator_mae":     [score, ci_low, ci_high],
  "SomeEstimator_mse":     [score, ci_low, ci_high],
  "SomeEstimator_rmse":    [score, ci_low, ci_high],
  "SomeEstimator_tailmae": [score, ci_low, ci_high],
  "SomeEstimator_r2score": [score, ci_low, ci_high],
  "SomeEstimator_mae-":    [...]   <- skipped: metric name is followed by "-"
}

Rules:
- last "_" separates estimator name (left) from metric name (right)
- if the metric part contains a "-" (e.g. "mae-", "mae-old"), the entry is skipped
- one panel per metric, two panels per row; an odd last panel is centred
- R² is higher-is-better (best = max, arrow "↑"); all others are errors
  (best = min, arrow "↓")
- both a PNG and a TIFF are written on every run
- the R² axis starts at --r2_floor (default -0.2) so a very poor baseline
  doesn't squeeze the 0-1 range; points/CI ends below it are drawn as "<" at
  the axis edge and labelled with their true value (the caption should say so)

Accessibility choices (OUP "Making Figures Accessible", 2024):
- meaning never carried by colour alone: every estimator is named on every
  panel's y-axis, and the best estimator per metric is marked with a different
  marker shape (diamond vs circle) plus its value label
- single dark colour with high contrast on white (well above 3:1 for graphics,
  4.5:1 for text); black text throughout
- Arial (sans-serif), 12-14 pt, no
  bold/italic/caps styling
- axis labels use "\u2193"/"\u2191" for direction; explain it in the figure caption
- value labels placed on plain white background, not on top of shapes
- a TIFF (RGB, LZW, 600 dpi by default) is written automatically on every run
- alt text written alongside the figure (concise, no formatting, ends with a full stop)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


METRIC_ORDER = ["mae", "mse", "rmse", "tailmae", "r2score"]
METRIC_LABELS = {"mae": "MAE", "mse": "MSE", "rmse": "RMSE", "tailmae": "Tail MAE",
                 "r2score": "R²"}
# Metrics where a larger value is better; all others are errors (lower is better)
HIGHER_IS_BETTER = {"r2score"}
# Lowest value shown on a metric's axis. A very poor baseline (e.g. R² = -3)
# would otherwise squeeze the interesting 0-1 range into a sliver. Points and
# CI ends below the floor are drawn at the axis edge as a triangle pointing
# off the axis; the value label still shows the true score.
AXIS_FLOOR = {"r2score": -0.2}
OFF_SCALE_NOTE = ("{metric} axis starts at {floor}; values below it are drawn "
                  "as triangles at the axis edge, labelled with their true value.")

ESTIMATOR_LABELS = {
    "best_estimator": "XGBoost Regressor",
    "baseline_pred_median": "Predict Median",
    "baseline_pred_mean": "Predict Mean",
    "baseline_drg_expected": "DRG Expected",
}
METRIC_SPOKEN = {
    "mae": "mean absolute error",
    "mse": "mean squared error",
    "rmse": "root mean squared error",
    "tailmae": "tail mean absolute error",
    "r2score": "R squared",
}

# Dark blue: ~10:1 contrast on white. Grid is decorative, kept light.
POINT_COLOUR = "#1F3A68"
CI_COLOUR = "#1F3A68"
GRID_COLOUR = "#BBBBBB"
TEXT_COLOUR = "#000000"
ARROW_DOWN = "\u2193"  # lower is better
ARROW_UP = "\u2191"    # higher is better
ARROW_SIZE = 18        # font size of the arrow, independent of the label size


def higher_is_better(metric: str) -> bool:
    return metric in HIGHER_IS_BETTER


def axis_floor(metric, values):
    """The floor for this metric's axis if any of `values` falls below it,
    else None (no clipping needed)."""
    floor = AXIS_FLOOR.get(metric)
    vals = np.asarray(values, dtype=float)
    if floor is None or not np.any(vals[~np.isnan(vals)] < floor):
        return None
    return floor


def off_scale_sentence(metric, floor):
    return OFF_SCALE_NOTE.format(metric=METRIC_SPOKEN.get(metric, metric),
                                 floor=f"{floor:g}".replace("\u2212", "minus "))


def best_index(scores, metric):
    """Index of the best score for this metric (ignores NaN)."""
    return int(np.nanargmax(scores) if higher_is_better(metric) else np.nanargmin(scores))


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


def est_label(est: str) -> str:
    return ESTIMATOR_LABELS.get(est, est)


def parse_key(key: str):
    if "_" not in key:
        raise ValueError(f"Key '{key}' does not contain '_' to split estimator/metric.")
    est, metric = key.rsplit("_", 1)
    return est, metric


def load_results(data_path: str):
    with Path(data_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    results, estimators, skipped = {}, set(), []
    for k, v in data.items():
        est, metric = parse_key(k)
        if "-" in metric:          # "-" after the metric name -> do not plot
            skipped.append(k)
            continue
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise ValueError(f"Value for '{k}' must be [score, ci_low, ci_high]. Got: {v}")
        score, lo, hi = map(float, v)
        estimators.add(est)
        results.setdefault(metric, {})[est] = (score, lo, hi)

    if skipped:
        print(f"Skipped {len(skipped)} entries marked with '-': {', '.join(sorted(skipped))}")
    return results, sorted(estimators)


def fmt(x, ref=None):
    """Format x with 3 significant figures relative to `ref` (the largest
    magnitude in the panel), so all labels in a panel share the same number
    of decimals and near-zero values (e.g. R² of a mean baseline) read as
    0.000 instead of scientific notation. Uses a true minus sign."""
    ref = abs(x) if ref is None else abs(ref)
    if ref == 0:
        return "0"
    if ref >= 1000 or ref < 0.001:
        out = f"{x:.2e}"
    else:
        decimals = max(0, 2 - int(np.floor(np.log10(ref))))
        out = f"{x:.{decimals}f}"
        if float(out) == 0:
            out = out.lstrip("-")          # no "-0.000"
    return out.replace("-", "\u2212", 1) if out.startswith("-") else out


def make_grid(fig, n_panels, ncols=2):
    """Axes in a grid with `ncols` panels per row. If the last row is not full,
    its panels are centred. Each panel spans two columns of a 2*ncols-wide
    GridSpec, so a lone last panel can sit at offset 1 (i.e. centred)."""
    nrows = int(np.ceil(n_panels / ncols))
    gs = fig.add_gridspec(nrows, 2 * ncols)
    axes = []
    for i in range(n_panels):
        r, c = divmod(i, ncols)
        in_row = min(ncols, n_panels - r * ncols)
        offset = ncols - in_row              # half-panel shift per missing panel
        start = 2 * c + offset
        axes.append(fig.add_subplot(gs[r, start:start + 2]))
    return axes


def plot_results(results, estimators, png_path=None, tiff_path=None,
                 sort_by="mae", png_dpi=300, tiff_dpi=600, show=False):
    metrics = [m for m in METRIC_ORDER if m in results] + \
              sorted(m for m in results if m not in METRIC_ORDER)
    if not metrics or not estimators:
        raise ValueError("No metrics/estimators found to plot.")

    # Same estimator order on every panel (sorted by `sort_by`, best first)
    if sort_by in results:
        sign = -1 if higher_is_better(sort_by) else 1
        estimators = sorted(
            estimators,
            key=lambda e: (sign * results[sort_by].get(e, (sign * np.inf,))[0], e),
        )

    n = len(estimators)
    longest = max(len(est_label(e)) for e in estimators)
    panel_h = max(2.6, 0.5 * n + 1.0)
    panel_w = 4.6 + 0.07 * longest
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig = plt.figure(figsize=(ncols * panel_w, nrows * panel_h))
    axes = make_grid(fig, len(metrics), ncols=ncols)
    y = np.arange(n)

    best_per_metric, clipped_metrics = {}, {}
    for ax, metric in zip(axes, metrics):
        scores, lo_err, hi_err, his = [], [], [], []
        for est in estimators:
            s, lo, hi = results[metric].get(est, (np.nan, np.nan, np.nan))
            scores.append(s)
            lo_err.append(0.0 if np.isnan(s) else max(0.0, s - lo))
            hi_err.append(0.0 if np.isnan(s) else max(0.0, hi - s))
            his.append(hi)
        scores = np.array(scores)
        best = best_index(scores, metric)
        best_per_metric[metric] = (est_label(estimators[best]), scores[best])

        los = scores - np.array(lo_err)
        his = scores + np.array(hi_err)
        floor = axis_floor(metric, los)
        if floor is not None:
            clipped_metrics[metric] = floor
        # Displayed positions: clipped at the floor (NaN stays NaN)
        f = -np.inf if floor is None else floor
        s_d, lo_d, hi_d = np.maximum(scores, f), np.maximum(los, f), np.maximum(his, f)

        # CIs drawn by hand so a clipped end gets a "<" instead of a cap
        for i in range(n):
            if np.isnan(scores[i]):
                continue
            ax.hlines(y[i], lo_d[i], hi_d[i], color=CI_COLOUR, lw=1.6, zorder=2)
            if his[i] >= f:
                ax.plot(hi_d[i], y[i], "|", ms=8, mew=1.6, color=CI_COLOUR, zorder=2)
            if los[i] >= f:
                ax.plot(lo_d[i], y[i], "|", ms=8, mew=1.6, color=CI_COLOUR, zorder=2)
            elif scores[i] >= f:          # only the CI runs off the axis
                ax.plot(f, y[i], "<", ms=6, color=CI_COLOUR, mec="none", zorder=3)

        # Shape, not colour, marks the best estimator; off-scale points are "<"
        on = ~np.isnan(scores) & (scores >= f)
        off = ~np.isnan(scores) & (scores < f)
        others = [i for i in range(n) if i != best and on[i]]
        ax.plot(scores[others], y[others], "o", ms=4.5, color=POINT_COLOUR,
                mec="none", zorder=3)
        if on[best]:
            ax.plot(scores[best], y[best], "D", ms=5.5, color=POINT_COLOUR,
                    mec="none", zorder=3)
        if off.any():
            ax.plot(np.full(off.sum(), f), y[off], "<", ms=8, color=POINT_COLOUR,
                    mec="none", zorder=3)

        # Value labels to the right of each CI, on white background
        xmin = np.nanmin(lo_d)
        xmax = np.nanmax(hi_d)
        span = (xmax - xmin) or abs(xmax) or 1.0
        ref = np.nanmax(np.abs(np.where(off, np.nan, scores))) if on.any() else None
        for i, s in enumerate(scores):
            if np.isnan(s):
                ax.text(xmin, y[i], "  no data", va="center", fontsize=11)
                continue
            ax.text(hi_d[i] + 0.03 * span, y[i], fmt(s, ref), va="center",
                    fontsize=11, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_xlim(xmin - 0.06 * span, xmax + 0.32 * span)

        ax.set_yticks(y)
        ax.set_yticklabels([est_label(e) for e in estimators])
        ax.set_ylim(n - 0.5, -0.5)          # first (best on sort_by) at top
        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left")
        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, axis="x", color=GRID_COLOUR, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout(h_pad=2.0, w_pad=2.5)
    add_direction_arrows(fig, axes, metrics)

    alt = make_alt_text(metrics, estimators, best_per_metric, sort_by)
    for m, fl in clipped_metrics.items():
        alt += " " + off_scale_sentence(m, fl)
        print("Caption note: " + OFF_SCALE_NOTE.format(
            metric=METRIC_LABELS.get(m, m), floor=f"{fl:g}".replace("-", "\u2212")))
    if png_path:
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, format="png", dpi=png_dpi, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved PNG to {png_path} ({png_dpi} dpi)")
    if tiff_path:
        save_tiff(fig, tiff_path, dpi=tiff_dpi)
    alt_base = png_path or tiff_path
    if alt_base:
        alt_path = Path(alt_base).with_suffix(".alt.txt")
        alt_path.write_text(alt + "\n", encoding="utf-8")
        print(f"Saved alt text to {alt_path}")
    else:
        print("Alt text:", alt)
    if show:
        plt.show()
    plt.close(fig)
    return alt


def save_tiff(fig, path, dpi=600):
    """Save the figure as a print-ready TIFF: RGB (no alpha), white background,
    LZW-compressed, with the DPI stored in the file. 600 dpi suits line art
    such as these plots; most journals ask for 300-600 dpi."""
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        path = path.with_suffix(".tiff")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, format="tiff", dpi=dpi, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    # Matplotlib writes RGBA; flatten to RGB, which journals and Word expect.
    from PIL import Image
    with Image.open(path) as im:
        if im.mode != "RGB":
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
            bg.save(path, compression="tiff_lzw", dpi=(dpi, dpi))
    print(f"Saved TIFF to {path} ({dpi} dpi)")
    return path


def add_direction_arrows(fig, axes, metrics):
    """Place a separately sized arrow just to the right of each x-axis label:
    down for lower-is-better metrics, up for higher-is-better ones."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, metric in zip(axes, metrics):
        arrow = ARROW_UP if higher_is_better(metric) else ARROW_DOWN
        bb = ax.xaxis.label.get_window_extent(renderer)
        x, y = fig.transFigure.inverted().transform((bb.x1, (bb.y0 + bb.y1) / 2))
        fig.text(x, y, " " + arrow, fontsize=ARROW_SIZE, ha="left", va="center",
                 color=TEXT_COLOUR)


NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def join_words(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def winner_sentences(metrics, best_est):
    """'X has the lowest A and B and the highest C.' -- one sentence per winner.
    best_est maps metric -> display name of the best estimator."""
    winners = {}
    for m in metrics:
        key = "highest" if higher_is_better(m) else "lowest"
        winners.setdefault(best_est[m], {}).setdefault(key, []).append(
            METRIC_SPOKEN.get(m, m))
    out = []
    for est, by_dir in winners.items():
        clauses = [f"the {d} {join_words(ms)}" for d, ms in by_dir.items()]
        out.append(f"{est} has {' and '.join(clauses)}.")
    return out


def make_alt_text(metrics, estimators, best, sort_by):
    names = join_words([METRIC_SPOKEN.get(m, m) for m in metrics])
    count = NUMBER_WORDS.get(len(metrics), str(len(metrics)))
    parts = [
        f"{count}-panel dot plot comparing {len(estimators)} estimators on {names}, "
        f"each with confidence intervals."
    ]
    parts += winner_sentences(metrics, {m: best[m][0] for m in metrics})
    return " ".join(parts)


def add_floor_args(ap):
    ap.add_argument("--r2_floor", type=float, default=AXIS_FLOOR["r2score"],
                    help="Lowest R\u00b2 shown on the axis; lower values are drawn at "
                         "the edge with their true value (default: %(default)s).")
    ap.add_argument("--no_r2_floor", action="store_true",
                    help="Show the full R\u00b2 range, however low.")


def apply_floor_args(args):
    if args.no_r2_floor:
        AXIS_FLOOR.pop("r2score", None)
    else:
        AXIS_FLOOR["r2score"] = args.r2_floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Path to JSON with metric results.")
    ap.add_argument("--out", default=None,
                    help="Output path without or with extension (e.g. results/perf.png). "
                         "Both <out>.png and <out>.tiff are written. "
                         "Default: <data file>_plot.png/.tiff.")
    ap.add_argument("--sort_by", default="mae", help="Metric used to order estimators (default: mae).")
    ap.add_argument("--png_dpi", type=int, default=300, help="PNG resolution (default: 300).")
    ap.add_argument("--tiff_dpi", type=int, default=600, help="TIFF resolution (default: 600).")
    ap.add_argument("--show", action="store_true", help="Also show the plot in a window.")
    add_floor_args(ap)
    args = ap.parse_args()
    apply_floor_args(args)

    if args.out:
        base = Path(args.out)
        if base.suffix.lower() in (".png", ".tif", ".tiff"):
            base = base.with_suffix("")
    else:
        d = Path(args.data_path)
        base = d.with_name(d.stem + "_plot")
    png_path = base.with_name(base.name + ".png")
    tiff_path = base.with_name(base.name + ".tiff")

    setup_style()
    results, estimators = load_results(args.data_path)
    plot_results(results, estimators, png_path=png_path, tiff_path=tiff_path,
                 sort_by=args.sort_by, png_dpi=args.png_dpi,
                 tiff_dpi=args.tiff_dpi, show=args.show)


if __name__ == "__main__":
    main()
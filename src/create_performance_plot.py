#!/usr/bin/env python3
"""
Plot estimator metrics with confidence intervals as an accessible 2x2 grid.

Expected JSON format (same as before):
{
  "SomeEstimator_mae":     [score, ci_low, ci_high],
  "SomeEstimator_mse":     [score, ci_low, ci_high],
  "SomeEstimator_rmse":    [score, ci_low, ci_high],
  "SomeEstimator_tailmae": [score, ci_low, ci_high],
  "SomeEstimator_mae-":    [...]   <- skipped: metric name is followed by "-"
}

Rules:
- last "_" separates estimator name (left) from metric name (right)
- if the metric part contains a "-" (e.g. "mae-", "mae-old"), the entry is skipped
- one panel per metric, laid out in a 2x2 grid

Accessibility choices (OUP "Making Figures Accessible", 2024):
- meaning never carried by colour alone: every estimator is named on every
  panel's y-axis, and the best estimator per metric is marked with a different
  marker shape (diamond vs circle) plus its value label
- single dark colour with high contrast on white (well above 3:1 for graphics,
  4.5:1 for text); black text throughout
- Arial (sans-serif), 12-14 pt, no
  bold/italic/caps styling
- axis labels use "\u2193" for direction; explain it in the figure caption
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


METRIC_ORDER = ["mae", "mse", "rmse", "tailmae"]
METRIC_LABELS = {"mae": "MAE", "mse": "MSE", "rmse": "RMSE", "tailmae": "Tail MAE"}
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
}

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


def fmt(x):
    if x == 0:
        return "0"
    mag = abs(x)
    if mag >= 1000 or mag < 0.001:
        return f"{x:.2e}"
    return f"{x:#.3g}".rstrip(".")


def plot_results(results, estimators, out_path=None, sort_by="mae",
                 tiff_path=None, tiff_dpi=600):
    metrics = [m for m in METRIC_ORDER if m in results] + \
              sorted(m for m in results if m not in METRIC_ORDER)
    if len(metrics) > 4:
        print(f"Note: more than 4 metrics found; plotting the first 4: {metrics[:4]}")
        metrics = metrics[:4]
    if not metrics or not estimators:
        raise ValueError("No metrics/estimators found to plot.")

    # Same estimator order on every panel (sorted by `sort_by`, lowest first)
    if sort_by in results:
        estimators = sorted(
            estimators,
            key=lambda e: (results[sort_by].get(e, (np.inf,))[0], e),
        )

    n = len(estimators)
    longest = max(len(est_label(e)) for e in estimators)
    panel_h = max(2.6, 0.5 * n + 1.0)
    panel_w = 4.6 + 0.07 * longest
    fig, axes = plt.subplots(2, 2, figsize=(2 * panel_w, 2 * panel_h))
    axes = axes.ravel()
    y = np.arange(n)

    best_per_metric = {}
    for ax, metric in zip(axes, metrics):
        scores, lo_err, hi_err, his = [], [], [], []
        for est in estimators:
            s, lo, hi = results[metric].get(est, (np.nan, np.nan, np.nan))
            scores.append(s)
            lo_err.append(0.0 if np.isnan(s) else max(0.0, s - lo))
            hi_err.append(0.0 if np.isnan(s) else max(0.0, hi - s))
            his.append(hi)
        scores = np.array(scores)
        best = int(np.nanargmin(scores))
        best_per_metric[metric] = (est_label(estimators[best]), scores[best])

        ax.errorbar(scores, y, xerr=[lo_err, hi_err], fmt="none",
                    ecolor=CI_COLOUR, elinewidth=1.6, capsize=4, capthick=1.6, zorder=2)
        # Shape, not colour, marks the best estimator
        others = [i for i in range(n) if i != best and not np.isnan(scores[i])]
        ax.plot(scores[others], y[others], "o", ms=4.5, color=POINT_COLOUR,
                mec="none", zorder=3)
        ax.plot(scores[best], y[best], "D", ms=5.5, color=POINT_COLOUR,
                mec="none", zorder=3)

        # Value labels to the right of each CI, on white background
        xmin = np.nanmin([s - l for s, l in zip(scores, lo_err)])
        xmax = np.nanmax(his)
        span = (xmax - xmin) or abs(xmax) or 1.0
        for i, s in enumerate(scores):
            if np.isnan(s):
                ax.text(xmin, y[i], "  no data", va="center", fontsize=11)
                continue
            ax.text(s + hi_err[i] + 0.03 * span, y[i], fmt(s), va="center",
                    fontsize=11, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_xlim(xmin - 0.06 * span, xmax + 0.32 * span)

        ax.set_yticks(y)
        ax.set_yticklabels([est_label(e) for e in estimators])
        ax.set_ylim(n - 0.5, -0.5)          # first (lowest sort_by error) at top
        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left")
        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, axis="x", color=GRID_COLOUR, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.tight_layout(h_pad=2.0, w_pad=2.5)
    add_direction_arrows(fig, axes[:len(metrics)])

    alt = make_alt_text(metrics, estimators, best_per_metric, sort_by)
    if tiff_path:
        save_tiff(fig, tiff_path, dpi=tiff_dpi)
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        alt_path = Path(out_path).with_suffix(".alt.txt")
        alt_path.write_text(alt + "\n", encoding="utf-8")
        print(f"Saved figure to {out_path}\nSaved alt text to {alt_path}")
    else:
        print("Alt text:", alt)
        plt.show()
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


def add_direction_arrows(fig, axes):
    """Place a separately sized arrow just to the right of each x-axis label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        bb = ax.xaxis.label.get_window_extent(renderer)
        x, y = fig.transFigure.inverted().transform((bb.x1, (bb.y0 + bb.y1) / 2))
        fig.text(x, y, " " + ARROW, fontsize=ARROW_SIZE, ha="left", va="center",
                 color=TEXT_COLOUR)


def make_alt_text(metrics, estimators, best, sort_by):
    names = ", ".join(METRIC_SPOKEN.get(m, m) for m in metrics)
    parts = [
        f"Four-panel dot plot comparing {len(estimators)} estimators on {names}, "
        f"each with confidence intervals."
    ]
    winners = {}
    for m in metrics:
        winners.setdefault(best[m][0], []).append(METRIC_SPOKEN.get(m, m))
    for est, ms in winners.items():
        joined = ms[0] if len(ms) == 1 else ", ".join(ms[:-1]) + " and " + ms[-1]
        parts.append(f"{est} has the lowest {joined}.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Path to JSON with metric results.")
    ap.add_argument("--out", default=None, help="Output image path (e.g. results.png). If not set, shows plot.")
    ap.add_argument("--sort_by", default="mae", help="Metric used to order estimators (default: mae).")
    ap.add_argument("--tiff", default=None,
                    help="TIFF output path. Default: same name as --out, or <data file>_plot.tiff.")
    ap.add_argument("--tiff_dpi", type=int, default=600, help="TIFF resolution (default: 600).")
    ap.add_argument("--no_tiff", action="store_true", help="Do not write a TIFF.")
    args = ap.parse_args()

    tiff_path = None
    if not args.no_tiff:
        if args.tiff:
            tiff_path = args.tiff
        elif args.out:
            tiff_path = Path(args.out).with_suffix(".tiff")
        else:
            d = Path(args.data_path)
            tiff_path = d.with_name(d.stem + "_plot.tiff")

    setup_style()
    results, estimators = load_results(args.data_path)
    plot_results(results, estimators, out_path=args.out, sort_by=args.sort_by,
                 tiff_path=tiff_path, tiff_dpi=args.tiff_dpi)


if __name__ == "__main__":
    main()
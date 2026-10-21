#!/usr/bin/env python3
"""
Plot estimator metrics per cohort as two accessible 2x2 figures:

  1. <out>_gender_nationality : Male, Female | Italian, non-Italian
  2. <out>_age                : Age 18-44, 45-59, 60-74, 75+

Reuses labels, style and TIFF export from plot_metrics_accessible.py,
which must be in the same folder.

Input: the same JSON as before. Only keys whose metric part contains "-"
are used here; text after the first "-" is the cohort name:

  "averaged_best_estimator_mae-gender-0":  [score, ci_low, ci_high]
   estimator = averaged_best_estimator, metric = mae, cohort = gender-0

Layout (same in both figures, so they read the same way):
- one panel per metric (MAE, MSE, RMSE, Tail MAE), each with its own y-axis
- cohorts along the x-axis, estimators side by side within each cohort
- each estimator has its own colour AND marker shape, so colour is never
  the only cue; one legend at the top is shared by all panels
- gender/nationality are unordered categories: points only, no lines, with
  "Gender" and "Nationality" group labels under the x-axis
- age groups are ordered: thin lines join each estimator's points to show
  the trend with age
- vertical bars are confidence intervals; "↓" next to each panel title
  means lower is better (explain this in the caption)
- each point's score is printed at 12 pt (the guide's 12-14 pt range),
  rotated to run vertically just above its own confidence bar: every label
  stays in its estimator's column, so labels never overlap each other
- the exact numbers also go to a CSV; alt text goes to a .alt.txt file
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from create_performance_plot import (
    METRIC_ORDER, METRIC_LABELS, METRIC_SPOKEN, ESTIMATOR_LABELS,
    GRID_COLOUR, TEXT_COLOUR, ARROW, ARROW_SIZE,
    setup_style, save_tiff, est_label, parse_key, fmt,
)


COHORT_NAMES = {
    "gender-0": "Male",
    "gender-1": "Female",
    "italian": "Italian",
    "non-italian": "non-Italian",
    "age-18-44": "Age 18-44",
    "age-45-59": "Age 45-59",
    "age-60-74": "Age 60-74",
    "age-75-plus": "Age 75+",   # in case the key is spelled "plus"
}

# Figure definitions: cohort keys in display order, grouped. Each inner list
# is one group on the x-axis; groups are separated by a gap.
FIGURES = {
    "gender_nationality": {
        "groups": [("Gender", ["gender-0", "gender-1"]),
                   ("Nationality", ["italian", "non-italian"])],
        "connect": False,
        "spoken": "gender (male, female) and nationality (Italian, non-Italian) cohorts",
    },
    "age": {
        "groups": [(None, ["age-18-44", "age-45-59", "age-60-74",
                           ["age-75-plut", "age-75-plus"]])],
        "connect": True,
        "spoken": "four age groups (18-44, 45-59, 60-74 and 75 and over)",
    },
}

# Okabe-Ito colours that keep >= 3:1 contrast on white, paired with distinct
# marker shapes. Assigned to estimators in ESTIMATOR_LABELS order.
SERIES_STYLES = [
    {"color": "#0072B2", "marker": "o"},   # blue
    {"color": "#D55E00", "marker": "s"},   # vermillion
    {"color": "#009E73", "marker": "^"},   # bluish green
    {"color": "#000000", "marker": "D"},   # black
]
MARKER_SIZE = 5
GROUP_GAP = 0.6          # extra x-space between cohort groups
DODGE_WIDTH = 0.72       # total horizontal spread of estimators in a cohort
VALUE_FONTSIZE = 12      # size of the score labels (guide: 12-14 pt)
SHOW_VALUES = True       # set False to hide the score labels


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_cohort_results(data_path):
    """Return results[metric][cohort][estimator] = (score, lo, hi)."""
    with Path(data_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    results, estimators = {}, set()
    for k, v in data.items():
        est, metric_part = parse_key(k)
        if "-" not in metric_part:
            continue                       # overall results: other script
        metric, cohort = metric_part.split("-", 1)
        if not cohort:
            continue                       # trailing "-" with no cohort name
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise ValueError(f"Value for '{k}' must be [score, ci_low, ci_high]. Got: {v}")
        estimators.add(est)
        results.setdefault(metric, {}).setdefault(cohort, {})[est] = tuple(map(float, v))

    known = [e for e in ESTIMATOR_LABELS if e in estimators]
    extra = sorted(estimators - set(known))
    unknown_cohorts = {c for m in results.values() for c in m} - set(COHORT_NAMES)
    if unknown_cohorts:
        print(f"Note: cohorts not in COHORT_NAMES (not plotted): {sorted(unknown_cohorts)}")
    return results, known + extra


def resolve_cohorts(groups, results):
    """Turn group definitions into [(group_label, [cohort_key, ...]), ...],
    keeping only cohorts present in the data. A list entry means "whichever
    of these spellings exists"."""
    present = {c for m in results.values() for c in m}
    out = []
    for label, keys in groups:
        found = []
        for k in keys:
            options = k if isinstance(k, list) else [k]
            hit = next((o for o in options if o in present), None)
            if hit:
                found.append(hit)
        if found:
            out.append((label, found))
    return out


def x_positions(groups):
    xs, x = [], 0.0
    for gi, (_, keys) in enumerate(groups):
        if gi:
            x += GROUP_GAP
        for _ in keys:
            xs.append(x)
            x += 1.0
    return xs


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_cohort_figure(results, estimators, spec, name):
    groups = resolve_cohorts(spec["groups"], results)
    if not groups:
        print(f"No data for the {name} figure; skipped.")
        return None, None
    cohorts = [c for _, keys in groups for c in keys]
    xs = np.array(x_positions(groups))
    metrics = [m for m in METRIC_ORDER if m in results][:4]

    n_est = len(estimators)
    offsets = (np.linspace(-DODGE_WIDTH / 2, DODGE_WIDTH / 2, n_est)
               if n_est > 1 else np.zeros(1))
    styles = {e: SERIES_STYLES[i % len(SERIES_STYLES)] for i, e in enumerate(estimators)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        for j, est in enumerate(estimators):
            vals = [results[metric].get(c, {}).get(est, (np.nan,) * 3) for c in cohorts]
            s = np.array([v[0] for v in vals])
            lo = np.array([v[0] - v[1] for v in vals]).clip(min=0)
            hi = np.array([v[2] - v[0] for v in vals]).clip(min=0)
            st = styles[est]
            ax.errorbar(
                xs + offsets[j], s, yerr=[lo, hi],
                fmt=st["marker"], ms=MARKER_SIZE, color=st["color"],
                ecolor=st["color"], elinewidth=1.4, capsize=3, capthick=1.4,
                linestyle="-" if spec["connect"] else "none", linewidth=1.0,
                zorder=3,
            )
            if SHOW_VALUES:
                for x, sv, h in zip(xs + offsets[j], s, hi):
                    if np.isnan(sv):
                        continue
                    ax.annotate(
                        fmt(sv), (x, sv + h), xytext=(0, 3), textcoords="offset points",
                        rotation=90, ha="center", va="bottom",
                        fontsize=VALUE_FONTSIZE, color=TEXT_COLOUR, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"),
                    )
        ax.margins(y=0.06)

        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left")
        ax.grid(True, axis="y", color=GRID_COLOUR, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Light separator between cohort groups (decorative)
        bounds = np.cumsum([len(k) for _, k in groups])[:-1]
        for b in bounds:
            ax.axvline((xs[b - 1] + xs[b]) / 2, color=GRID_COLOUR, linewidth=0.8)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([COHORT_NAMES[c] for c in cohorts])
    axes[0].set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    for ax in axes[2:4]:
        ax.tick_params(axis="x", labelbottom=True)

    handles = [
        Line2D([], [], color=styles[e]["color"], marker=styles[e]["marker"],
               ms=MARKER_SIZE + 2, linestyle="-" if spec["connect"] else "none",
               label=est_label(e))
        for e in estimators
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(n_est, 4),
               frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.4,
               columnspacing=1.6)

    has_group_labels = any(label for label, _ in groups)
    fig.tight_layout(rect=(0, 0.04 if has_group_labels else 0, 1, 0.94),
                     h_pad=2.5, w_pad=3.0)
    if SHOW_VALUES:
        fit_value_labels(fig, axes[:len(metrics)])
    if has_group_labels:
        add_group_labels(fig, axes[2:4], groups, xs)
    add_title_arrows(fig, axes[:len(metrics)])

    alt = make_alt_text(results, estimators, metrics, cohorts, spec)
    return fig, alt


def fit_value_labels(fig, axes, pad_px=4):
    """Raise each panel's y-limit until the rotated value labels fit inside."""
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        changed = False
        for ax in axes:
            texts = [t for t in ax.texts if t.get_rotation() == 90]
            if not texts:
                continue
            top_px = max(t.get_window_extent(r).y1 for t in texts)
            ax_top = ax.get_window_extent(r).y1
            if top_px + pad_px > ax_top:
                y0, y1 = ax.get_ylim()
                per_px = (y1 - y0) / ax.get_window_extent(r).height
                ax.set_ylim(y0, y1 + (top_px + pad_px - ax_top) * per_px * 1.05)
                changed = True
        if not changed:
            break


def add_group_labels(fig, axes, groups, xs):
    """Write 'Gender' / 'Nationality' centred under each group's tick labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        if not ax.get_visible():
            continue
        tick_bottom = min(t.get_window_extent(r).y0 for t in ax.get_xticklabels())
        i = 0
        for label, keys in groups:
            centre = xs[i:i + len(keys)].mean()
            i += len(keys)
            if not label:
                continue
            x_disp = ax.transData.transform((centre, 0))[0]
            fx, fy = fig.transFigure.inverted().transform((x_disp, tick_bottom - 6))
            fig.text(fx, fy, label, ha="center", va="top", fontsize=12,
                     color=TEXT_COLOUR)


def add_title_arrows(fig, axes):
    """Separately sized '↓' right after each panel title (lower is better)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        bb = ax._left_title.get_window_extent(r)
        x, y = fig.transFigure.inverted().transform((bb.x1, (bb.y0 + bb.y1) / 2))
        fig.text(x, y, " " + ARROW, fontsize=ARROW_SIZE, ha="left", va="center",
                 color=TEXT_COLOUR)


def make_alt_text(results, estimators, metrics, cohorts, spec):
    names = ", ".join(METRIC_SPOKEN.get(m, m) for m in metrics)
    parts = [f"Four-panel plot of {names} for {len(estimators)} estimators "
             f"across {spec['spoken']}, with confidence intervals."]

    # Who is best in each metric/cohort cell
    best = {}
    for m in metrics:
        for c in cohorts:
            cell = results[m].get(c, {})
            if cell:
                best[(m, c)] = min(cell, key=lambda e: cell[e][0])
    winners = set(best.values())
    if len(winners) == 1:
        parts.append(f"{est_label(winners.pop())} has the lowest error in every "
                     f"cohort for every metric.")
    else:
        for m in metrics:
            ws = {best[(m, c)] for c in cohorts if (m, c) in best}
            if len(ws) == 1:
                parts.append(f"{est_label(next(iter(ws)))} has the lowest "
                             f"{METRIC_SPOKEN.get(m, m)} in every cohort.")
            else:
                cells = "; ".join(f"{COHORT_NAMES[c]}: {est_label(best[(m, c)])}"
                                  for c in cohorts if (m, c) in best)
                parts.append(f"Lowest {METRIC_SPOKEN.get(m, m)} by cohort: {cells}.")

    # Spread across cohorts for the first estimator, on the first metric
    if metrics and estimators:
        m, e = metrics[0], estimators[0]
        vals = {c: results[m][c][e][0] for c in cohorts
                if e in results[m].get(c, {})}
        if len(vals) > 1:
            hi_c = max(vals, key=vals.get)
            lo_c = min(vals, key=vals.get)
            parts.append(
                f"For {est_label(e)}, {METRIC_SPOKEN.get(m, m)} is highest for "
                f"{COHORT_NAMES[hi_c]} ({vals[hi_c]:.3g}) and lowest for "
                f"{COHORT_NAMES[lo_c]} ({vals[lo_c]:.3g}).")
    return " ".join(parts)


def write_csv(path, results, estimators, metrics, cohorts):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "cohort", "estimator", "score", "ci_low", "ci_high"])
        for m in metrics:
            for c in cohorts:
                for e in estimators:
                    if e in results[m].get(c, {}):
                        s, lo, hi = results[m][c][e]
                        w.writerow([METRIC_LABELS.get(m, m), COHORT_NAMES[c],
                                    est_label(e), s, lo, hi])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Path to JSON with metric results.")
    ap.add_argument("--out", default=None,
                    help="Output prefix, e.g. figs/cohorts.png -> figs/cohorts_age.png etc. "
                         "Default: <data file>_cohorts next to the data file.")
    ap.add_argument("--tiff_dpi", type=int, default=600, help="TIFF resolution (default: 600).")
    ap.add_argument("--no_tiff", action="store_true", help="Do not write TIFFs.")
    ap.add_argument("--show", action="store_true", help="Also open the figures in a window.")
    args = ap.parse_args()

    if args.out:
        base = Path(args.out)
        ext = base.suffix or ".png"
        base = base.with_suffix("")
    else:
        d = Path(args.data_path)
        base, ext = d.with_name(d.stem + "_cohorts"), ".png"
    base.parent.mkdir(parents=True, exist_ok=True)

    setup_style()
    results, estimators = load_cohort_results(args.data_path)
    if not results:
        raise SystemExit("No cohort entries (metric names with '-') found.")
    metrics = [m for m in METRIC_ORDER if m in results][:4]

    for name, spec in FIGURES.items():
        fig, alt = plot_cohort_figure(results, estimators, spec, name)
        if fig is None:
            continue
        stem = base.with_name(f"{base.name}_{name}")
        png = stem.with_suffix(ext)
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to {png}")
        if not args.no_tiff:
            save_tiff(fig, stem.with_suffix(".tiff"), dpi=args.tiff_dpi)
        stem.with_suffix(".alt.txt").write_text(alt + "\n", encoding="utf-8")
        cohorts = [c for _, k in resolve_cohorts(spec["groups"], results) for c in k]
        write_csv(stem.with_suffix(".csv"), results, estimators, metrics, cohorts)
        print(f"Saved alt text and CSV next to {png.name}")
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Plot estimator metrics per cohort as two accessible figures:

  1. <out>_gender_nationality : Male, Female | Italian, non-Italian
  2. <out>_age                : Age 18-44, 45-59, 60-74, 75+

Reuses labels, style, grid layout and TIFF export from
create_performance_plot.py, which must be in the same folder.

Input: the same JSON as before. Only keys whose metric part contains "-"
are used here; text after the first "-" is the cohort name:

  "averaged_best_estimator_mae-gender-0":  [score, ci_low, ci_high]
   estimator = averaged_best_estimator, metric = mae, cohort = gender-0

Layout (same in both figures, so they read the same way):
- one panel per metric (MAE, MSE, RMSE, Tail MAE, R²), each with its own
  y-axis; two panels per row, an odd last panel is centred
- cohorts along the x-axis, estimators side by side within each cohort
- each estimator has its own colour AND marker shape, so colour is never
  the only cue; one legend at the top is shared by all panels
- gender/nationality are unordered categories: points only, no lines, with
  "Gender" and "Nationality" group labels under the x-axis
- age groups are ordered: thin lines join each estimator's points to show
  the trend with age
- the R² axis starts at --r2_floor (default -0.2) so a very poor baseline
  doesn't squeeze the 0-1 range; points/CI ends below it are drawn as "v" at
  the axis bottom and labelled with their true value (the caption should say so)
- vertical bars are confidence intervals; "↓" next to a panel title means
  lower is better, "↑" (R²) means higher is better (explain in the caption)
- each point's score is printed at 12 pt (the guide's 12-14 pt range),
  rotated to run vertically just above its own confidence bar: every label
  stays in its estimator's column, so labels never overlap each other
- each figure is saved as both PNG and TIFF; the exact numbers also go to
  a CSV; alt text goes to a .alt.txt file
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
    GRID_COLOUR, TEXT_COLOUR, ARROW_DOWN, ARROW_UP, ARROW_SIZE, NUMBER_WORDS,
    setup_style, save_tiff, est_label, parse_key, fmt, make_grid,
    higher_is_better, join_words, axis_floor, off_scale_sentence, OFF_SCALE_NOTE,
    add_floor_args, apply_floor_args,
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
    metrics = [m for m in METRIC_ORDER if m in results]

    n_est = len(estimators)
    offsets = (np.linspace(-DODGE_WIDTH / 2, DODGE_WIDTH / 2, n_est)
               if n_est > 1 else np.zeros(1))
    styles = {e: SERIES_STYLES[i % len(SERIES_STYLES)] for i, e in enumerate(estimators)}

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig = plt.figure(figsize=(12, 4.6 * nrows))
    axes = make_grid(fig, len(metrics), ncols=ncols)

    clipped = {}
    for ax, metric in zip(axes, metrics):
        # Same number of decimals for every value label in this panel
        ref = max((abs(v[0]) for c in cohorts for v in results[metric].get(c, {}).values()),
                  default=0.0)
        all_lows = [min(v[0], v[1]) for c in cohorts
                    for v in results[metric].get(c, {}).values()]
        floor = axis_floor(metric, all_lows)
        if floor is not None:
            clipped[metric] = floor
            # labels share decimals with the in-range values only
            in_range = [abs(v[0]) for c in cohorts
                        for v in results[metric].get(c, {}).values() if v[0] >= floor]
            ref = max(in_range, default=None)
        f = -np.inf if floor is None else floor

        for j, est in enumerate(estimators):
            vals = [results[metric].get(c, {}).get(est, (np.nan,) * 3) for c in cohorts]
            s = np.array([v[0] for v in vals])
            lo = np.minimum(np.array([v[1] for v in vals]), s)
            hi = np.maximum(np.array([v[2] for v in vals]), s)
            s_d, lo_d, hi_d = np.maximum(s, f), np.maximum(lo, f), np.maximum(hi, f)
            on = ~np.isnan(s) & (s >= f)
            off = ~np.isnan(s) & (s < f)
            x = xs + offsets[j]
            st = styles[est]
            c = st["color"]
            # CIs drawn by hand so a clipped end gets a "v" instead of a cap
            ok = ~np.isnan(s)
            ax.vlines(x[ok], lo_d[ok], hi_d[ok], color=c, lw=1.4, zorder=2)
            cap = dict(marker="_", ms=6, mew=1.4, color=c, linestyle="none", zorder=2)
            ax.plot(x[ok & (hi >= f)], hi_d[ok & (hi >= f)], **cap)
            ax.plot(x[ok & (lo >= f)], lo_d[ok & (lo >= f)], **cap)
            ax.plot(x[on & (lo < f)], np.full((on & (lo < f)).sum(), f), "v",
                    ms=5, color=c, mec="none", zorder=3)
            # Points; off-scale points become a "v" at the floor. The connecting
            # line (age figure) is broken at off-scale points.
            ax.plot(x, np.where(on, s, np.nan), marker=st["marker"],
                    ms=MARKER_SIZE, color=c, zorder=3,
                    linestyle="-" if spec["connect"] else "none", linewidth=1.0)
            if off.any():
                ax.plot(x[off], np.full(off.sum(), f), "v", ms=MARKER_SIZE + 3,
                        color=c, mec="none", zorder=3)
            if SHOW_VALUES:
                for xv, sv, top in zip(x, s, hi_d):
                    if np.isnan(sv):
                        continue
                    ax.annotate(
                        fmt(sv, ref), (xv, top), xytext=(0, 3), textcoords="offset points",
                        rotation=90, ha="center", va="bottom",
                        fontsize=VALUE_FONTSIZE, color=TEXT_COLOUR, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"),
                    )
        ax.margins(y=0.06)
        if floor is not None:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(f - 0.04 * (y1 - f), y1)

        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left")
        ax.grid(True, axis="y", color=GRID_COLOUR, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Light separator between cohort groups (decorative)
        bounds = np.cumsum([len(k) for _, k in groups])[:-1]
        for b in bounds:
            ax.axvline((xs[b - 1] + xs[b]) / 2, color=GRID_COLOUR, linewidth=0.8)

        # Every panel gets its own cohort labels: with a centred last panel
        # there is no single "bottom row" to carry them.
        ax.set_xticks(xs)
        ax.set_xticklabels([COHORT_NAMES[c] for c in cohorts])
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)

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
    legend_frac = 0.5 / fig.get_figheight()          # ~0.5 in for the legend
    fig.tight_layout(rect=(0, 0, 1, 1 - legend_frac),
                     h_pad=4.0 if has_group_labels else 2.5, w_pad=3.0)
    if SHOW_VALUES:
        fit_value_labels(fig, axes)
    if has_group_labels:
        add_group_labels(fig, axes, groups, xs)
    add_title_arrows(fig, axes, metrics)

    alt = make_alt_text(results, estimators, metrics, cohorts, spec)
    for m, fl in clipped.items():
        alt += " " + off_scale_sentence(m, fl)
        print(f"Caption note ({name}): " + OFF_SCALE_NOTE.format(
            metric=METRIC_LABELS.get(m, m), floor=f"{fl:g}".replace("-", "\u2212")))
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


def add_title_arrows(fig, axes, metrics):
    """Separately sized arrow right after each panel title:
    '↓' lower is better, '↑' higher is better (R²)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax, metric in zip(axes, metrics):
        arrow = ARROW_UP if higher_is_better(metric) else ARROW_DOWN
        bb = ax._left_title.get_window_extent(r)
        x, y = fig.transFigure.inverted().transform((bb.x1, (bb.y0 + bb.y1) / 2))
        fig.text(x, y, " " + arrow, fontsize=ARROW_SIZE, ha="left", va="center",
                 color=TEXT_COLOUR)


def make_alt_text(results, estimators, metrics, cohorts, spec):
    names = join_words([METRIC_SPOKEN.get(m, m) for m in metrics])
    count = NUMBER_WORDS.get(len(metrics), str(len(metrics)))
    parts = [f"{count}-panel plot of {names} for {len(estimators)} estimators "
             f"across {spec['spoken']}, with confidence intervals."]

    def word(m):
        return "highest" if higher_is_better(m) else "lowest"

    # Who is best in each metric/cohort cell
    best = {}
    for m in metrics:
        for c in cohorts:
            cell = results[m].get(c, {})
            if cell:
                pick = max if higher_is_better(m) else min
                best[(m, c)] = pick(cell, key=lambda e: cell[e][0])
    winners = set(best.values())
    if len(winners) == 1:
        parts.append(f"{est_label(winners.pop())} has the best score in every "
                     f"cohort for every metric.")
    else:
        for m in metrics:
            ws = {best[(m, c)] for c in cohorts if (m, c) in best}
            if len(ws) == 1:
                parts.append(f"{est_label(next(iter(ws)))} has the {word(m)} "
                             f"{METRIC_SPOKEN.get(m, m)} in every cohort.")
            else:
                cells = "; ".join(f"{COHORT_NAMES[c]}: {est_label(best[(m, c)])}"
                                  for c in cohorts if (m, c) in best)
                parts.append(f"{word(m).capitalize()} {METRIC_SPOKEN.get(m, m)} "
                             f"by cohort: {cells}.")

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
                    help="Output prefix, e.g. figs/cohorts -> figs/cohorts_age.png and "
                         "figs/cohorts_age.tiff etc. Default: <data file>_cohorts next "
                         "to the data file.")
    ap.add_argument("--png_dpi", type=int, default=300, help="PNG resolution (default: 300).")
    ap.add_argument("--tiff_dpi", type=int, default=600, help="TIFF resolution (default: 600).")
    ap.add_argument("--show", action="store_true", help="Also open the figures in a window.")
    add_floor_args(ap)
    args = ap.parse_args()
    apply_floor_args(args)

    if args.out:
        base = Path(args.out)
        if base.suffix.lower() in (".png", ".tif", ".tiff"):
            base = base.with_suffix("")
    else:
        d = Path(args.data_path)
        base = d.with_name(d.stem + "_cohorts")
    base.parent.mkdir(parents=True, exist_ok=True)

    setup_style()
    results, estimators = load_cohort_results(args.data_path)
    if not results:
        raise SystemExit("No cohort entries (metric names with '-') found.")
    metrics = [m for m in METRIC_ORDER if m in results]

    for name, spec in FIGURES.items():
        fig, alt = plot_cohort_figure(results, estimators, spec, name)
        if fig is None:
            continue
        stem = base.with_name(f"{base.name}_{name}")
        png = stem.with_name(stem.name + ".png")
        fig.savefig(png, format="png", dpi=args.png_dpi, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved PNG to {png} ({args.png_dpi} dpi)")
        save_tiff(fig, stem.with_name(stem.name + ".tiff"), dpi=args.tiff_dpi)
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
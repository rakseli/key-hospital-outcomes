import matplotlib
matplotlib.use("Agg")                     # no display needed
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import argparse
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from train_and_eval_models import SEED, load_params
from plotting_utils import setup_style

FEATURE = "n_nursing_diag"


def plot_pdp(pipe, X, feature=FEATURE):
    """Bar-style partial dependence plot of a whole-number count feature.

    Treating the count as categorical makes sklearn evaluate it at every
    observed (whole-number) value and draw one bar per value. The full
    pipeline is used, so the x-axis is on the original (unscaled) feature
    scale. Returns (fig, disp).
    """
    setup_style()                            # must run before the figure is created
    fig, ax = plt.subplots(figsize=(12, 4))
    disp = PartialDependenceDisplay.from_estimator(
        pipe,
        X,                                   # background data
        features=[feature],
        categorical_features=[feature],
        kind="average",
        ax=ax,
        subsample=None
    )
    ax = disp.axes_[0, 0]
    grid = np.asarray(disp.pd_results[0]["grid_values"][0])
    pd_values = np.asarray(disp.pd_results[0]["average"][0])
    for bar in ax.containers[0]:             # from_estimator has no bar_kw
        bar.set_color("0.35")

    # x: whole-number ticks 0, 2, 4, ... upright, no negative ticks.
    # The axis starts half a bar before 0 so the first bar is not cut off.
    x_max = int(grid.max())
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlim(-0.5, x_max + 0.5)

    # y: starts at 0 with evenly spaced whole-number ticks.
    y_locator = MaxNLocator(nbins=6, integer=True, min_n_ticks=4)
    ax.yaxis.set_major_locator(y_locator)
    y_ticks = y_locator.tick_values(0, pd_values.max())
    ax.set_ylim(0, y_ticks[y_ticks >= pd_values.max()][0])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_xlabel("Number of Nursing Diagnoses", fontsize=12, fontfamily="sans-serif")
    ax.set_ylabel("Partial Dependence (Predicted Length of Stay, days)",
                  fontsize=12, fontfamily="sans-serif")
    fig.tight_layout()
    return fig, disp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to CSV file")
    parser.add_argument("--final_model_path", default="../results/final_model.pkl")
    parser.add_argument("--output_path", default="../results/pdp_n_nursing_diag.png")
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)
    labels = df['length_of_stay'].copy()
    labels = labels.astype('float32')
    features = df.drop(columns=['length_of_stay', 'drg_day_expected', 'ID']).copy()
    features = features.astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=SEED, test_size=0.25)
    del df, labels, features
    final_model_pipe = load_params(args.final_model_path)

    fig, _ = plot_pdp(final_model_pipe, X_test)
    fig.savefig(args.output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
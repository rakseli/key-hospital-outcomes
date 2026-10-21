import matplotlib
matplotlib.use("Agg")                     # no display needed
import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from train_and_eval_models import SEED,load_params
from compute_shap_values import to_frame
from plotting_utils import setup_style

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to CSV file")
    parser.add_argument("--final_model_path",default="../results/final_model.pkl")
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)
    labels = df['length_of_stay'].copy()
    labels = labels.astype('float32')
    features = df.drop(columns=['length_of_stay','drg_day_expected','ID']).copy()
    features = features.astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(features, labels,random_state=SEED,test_size=0.25)
    del df, labels, features
    final_model_pipe = load_params(args.final_model_path)
    model = final_model_pipe[-1]             
    pre = final_model_pipe[:-1][-1]
    X_train_trans = to_frame(X_train,pre)
    X_test_trans = to_frame(X_test,pre)
    fig, ax = plt.subplots(figsize=(12, 4))
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X_test_trans,                               # background data
        features=['n_nursing_diag'],
        feature_names=X_test_trans.columns.values,
        kind="average",
        line_kw={"color": "black", "linewidth": 1.5},
        grid_resolution=50,
        subsample=None,
        ax=ax,
    )
    disp.axes_[0, 0].set_xlabel("Number of Nursing Diagnoses",fontsize=12, fontfamily="sans-serif")
    disp.axes_[0, 0].set_ylabel("Partial dependence", fontsize=12,fontfamily="sans-serif")
    setup_style()
    fig.tight_layout()
    fig.savefig("../results/pdp_n_nursing_diag.png", dpi=600, bbox_inches="tight",)
    plt.close(fig)

if __name__ == "__main__":
    main()
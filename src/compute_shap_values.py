import pandas as pd
import re
import time
import argparse
import numpy as np
import pickle
import json
import shap
import os
import sys
import warnings
sys.setrecursionlimit(10000)
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from train_and_eval_models import SEED,load_params
from save_shap_explanations import save_explanation, load_explanation
from plotting_utils import shap_hexbin, shap_beeswarm_binned,setup_style

def to_frame(X_raw: pd.DataFrame,transform_func) -> pd.DataFrame:
    out = transform_func.transform(X_raw)
    feature_names = out.columns.values
    if hasattr(out, "toarray"):
        out = out.toarray()
    return pd.DataFrame(out, columns=feature_names, index=X_raw.index)

def save_shap(plot_fn, path, **kw):
    plt.figure()
    plot_fn(show=False, **kw)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to CSV file")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tree_mode",action="store_true")
    parser.add_argument("--shapley_values_root",default="../results")
    parser.add_argument("--cluster_values_root",default="../results")
    parser.add_argument("--final_model_path",default="../results/final_model.pkl")

    args = parser.parse_args()
    if args.test:
        N_BACKGROUND=100
    else:
        N_BACKGROUND=5000
    acronyms = ["ARP", "DRG", "ICD","ICD9","ID"]
    cols_to_rename ={
        "N ICD Diag":"N ICD Diagnoses",
        "N Nursing Diag":"N Nursing Diagnoses",
        "N Procs": "N Processes",
        "Residence Dimissione A Domicilio Del Paziente Con Attivazion":"Patient discharge home with activation of home care",
        "Residence Decesso":"Discharge: Dead",
        "Residence Domicilio":"Discharge: Home",
        "Residence Domicilio In Attesa Completamento Iter":"Discharge: Discharge home pending completion of the care pathway",
        "Residence Trasferimento Ad Altro Istituto":"Discharge: Transfer to another healthcare facility",
        "Residence Trasferimento Ad Altro Regime O Tipo Di Assistenza": "Discharge: Transfer to another care setting or type of care",
        "Residence Volontario": "Discharge: Discharge against medical advice (voluntary discharge)"
    }
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
    cols = X_train_trans.columns.str.replace("_", " ", regex=False).str.title()
    pattern = r"\b(" + "|".join(map(re.escape, [a.title() for a in acronyms])) + r")\b"
    X_train_trans.columns = cols.str.replace(pattern, lambda m: m.group(1).upper(), regex=True)
    X_test_trans.columns = cols.str.replace(pattern, lambda m: m.group(1).upper(), regex=True)
    X_train_trans = X_train_trans.rename(columns=cols_to_rename)
    X_test_trans = X_test_trans.rename(columns=cols_to_rename)

    if args.tree_mode:
        plot_prefix = "tree_dependent_"
        explainer = shap.TreeExplainer(model,feature_perturbation="tree_path_dependent",model_output="raw")
    else:
        background = shap.maskers.Independent(X_train_trans, max_samples=N_BACKGROUND)
        explainer = shap.TreeExplainer(model,data=background,feature_perturbation="interventional",model_output="log_loss")
        plot_prefix = "interventional_"
    if args.test:
        X_test_trans, y_test = X_test_trans[:100], y_test[:100]
        plot_prefix = "test_" + plot_prefix

    y_test_arr = np.asarray(y_test, dtype=float)
    shapley_value_path = os.path.join(args.shapley_values_root,f"{plot_prefix}shapley_values.npz")
    if os.path.exists(shapley_value_path):
        explanation = load_explanation(shapley_value_path)
    else:    
        explanation = explainer(X_test_trans,y_test_arr)
        save_explanation(explanation,shapley_value_path,y_test_arr)
    
    #use xgboost_distances_r2 as metric
    clustering_cutoff=1
    N = 10
    #get global importances
    imp = np.abs(explanation.values).mean(0)
    #get top N features
    top = np.argsort(-imp)[:N]
    sv_top = explanation[:, top]
    names = sv_top.feature_names
    #safety checks
    for col in names:
        #we skip these as we assume attacker does not have access to hospital system
        if col in ['ARP Weight',"DRG Weight"]:
            continue
        counts = X_test_trans[col].value_counts()
        failing = counts[counts <= 10]
        if not failing.empty:
            print(f"{col}: {len(failing)} values with ≤10 rows")
            print(failing, "\n")


    #cluster the top for fast cluster computation
    cluster_values_path = os.path.join(args.cluster_values_root,f"{plot_prefix}data_clustering.pkl")
    if os.path.exists(cluster_values_path):
        with open(cluster_values_path, 'rb') as f:
            clust = pickle.load(f)
    else:
        clust = shap.utils.hclust(sv_top.data, y_test_arr,linkage="single",metric="xgboost_distances_r2")           
        with open(cluster_values_path, 'wb') as f:
            pickle.dump(clust, f)

    #set plotting style
    setup_style()

    save_shap(lambda **k: shap.plots.violin(explanation,plot_type="layered_violin",sort=True,max_display=10,layered_violin_max_num_bins=10,use_log_scale=True, **k), f"../results/{plot_prefix}shap_violin.png")
    save_shap(lambda **k: shap.plots.bar(sv_top,clustering=clust,clustering_cutoff=clustering_cutoff,max_display=N, **k), f"../results/{plot_prefix}{N}_most_important_features_clustering_cutoff_{clustering_cutoff}_shap_bar.png")
    save_shap(lambda **k: shap.plots.bar(explanation,**k), f"../results/{plot_prefix}shap_bar.png")
    save_shap(lambda **k: shap.plots.heatmap(explanation,**k), f"../results/{plot_prefix}shap_heatmap.png")
    log, supp = shap_beeswarm_binned(explanation, f"../results/{plot_prefix}shap_beeswarm_binned.png")
    print(log)
    for feat in ["DRG Weight","N Processes","N ICD Diagnoses","N Nursing Diagnoses","Admission Type"]:
        print(f"Plotting feature: {feat}")
        save_shap(lambda **k:shap.plots.scatter(explanation[:, feat]),f"../results/{plot_prefix}scatter_{feat}.png")
        if feat == "N Nursing Diagnoses":
            save_shap(lambda **k:shap.plots.scatter(explanation[:, feat],color=explanation[:, "Weight Is Missing"]),f"../results/{plot_prefix}scatter_{feat}_weight_is_missing.png")

    gender = ["Male" if explanation[i, "Sex"].data == 0 else "Female" for i in range(explanation.shape[0])]
    italian = ["Italian" if explanation[i, "Italian"].data == 0 else "non-Italian" for i in range(explanation.shape[0])]

    save_shap(lambda **k:shap.plots.bar(explanation.cohorts(gender).abs.mean(0)),f"../results/{plot_prefix}gender_shap_bar.png")
    save_shap(lambda **k:shap.plots.bar(explanation.cohorts(italian).abs.mean(0)),f"../results/{plot_prefix}nationality_shap_bar.png")
    with warnings.catch_warnings():
        save_shap(lambda **k:shap.decision_plot(base_value=explanation.base_values[0],shap_values=explanation.values[0], feature_order="importance", link="identity",features=X_test_trans.iloc[0,:]),f"../results/{plot_prefix}decision_single_decision_plot.png")
        save_shap(lambda **k:shap.decision_plot(base_value=np.mean(explanation.base_values[:10]),shap_values=explanation.values[:10],feature_order="importance", link="identity",features=X_test_trans.iloc[:10,:]),f"../results/{plot_prefix}decision_multiple_decisions_plot.png")
    
    for feat in ["ARP Weight","DRG Weight","N Processes","N ICD Diagnoses","N Nursing Diagnoses"]:
        info = shap_hexbin(explanation, feat, f"../results/{plot_prefix}{feat}_hexabin.png")
        print(info)
if __name__ == "__main__":
    main()

"""
Redundancy of ONE feature relative to the others, using the same math as
shap.utils.hclust(metric="xgboost_distances_r2") but without computing the
full p x p matrix.

shap's full matrix costs p univariate fits + p*(p-1) pairwise fits.
This costs (k+1) univariate fits + 2k pairwise fits, where k = number of
candidate features you actually care about (k = p-1 if you want all of them).

Scale (same as shap): distance 0 => perfectly redundant, 1 => no redundancy.
The distance is asymmetric; shap's "single" linkage symmetrises with min(),
which is what `dist` reproduces here.
"""

import numpy as np
import sys
import pandas as pd
import sklearn.model_selection
import xgboost
import seaborn as sns
from train_and_eval_models import SEED, NotUseNursingDiagnoses
from shap.utils._clustering import xgboost_distances_r2
import matplotlib.pyplot as plt

def _fit(x_train, t_train, x_test, t_test, lr, subsample, n_est, esr):
    """A depth-1 univariate XGBoost model, matching shap's settings exactly."""
    m = xgboost.XGBRegressor(
        subsample=subsample,
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=1,
        early_stopping_rounds=esr,
    )
    m.fit(x_train, t_train, eval_set=[(x_test, t_test)], verbose=False)
    return m


def feature_redundancy(
    X,
    y,
    target,
    others=None,
    learning_rate=0.6,
    early_stopping_rounds=2,
    subsample=1.0,
    max_estimators=10_000,
    random_state=0,
):
    """Redundancy distances between `target` and every feature in `others`.

    Parameters
    ----------
    X : DataFrame or 2d array
    y : 1d array, the label
    target : column name or index, the one feature you care about
    others : list of names/indices, or None for "every other feature"

    Returns
    -------
    DataFrame indexed by feature name with columns:
        dist_target_to_other : 1 - R^2 of `other` explaining target's model of y
        dist_other_to_target : the reverse direction
        dist                 : min of the two (= shap's "single" linkage value)
        redundancy           : 1 - dist, i.e. proportion of variance explained
    """
    names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]
    Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    idx = lambda c: names.index(c) if isinstance(c, str) else int(c)

    t = idx(target)
    others_idx = [k for k in range(len(names)) if k != t] if others is None else [k for k in map(idx, others) if k != t]

    # identical split to shap's, since train_test_split only depends on n_rows
    X_tr, X_te, y_tr, y_te = sklearn.model_selection.train_test_split(Xv, y, random_state=random_state)

    # step 1: univariate models of y, only for the features we need
    tr_p, te_p = {}, {}
    for i in [t] + others_idx:
        m = _fit(X_tr[:, i:i + 1], y_tr, X_te[:, i:i + 1], y_te,
                 learning_rate, subsample, max_estimators, early_stopping_rounds)
        tr_p[i] = m.predict(X_tr[:, i:i + 1])
        te_p[i] = m.predict(X_te[:, i:i + 1])

    # step 2: how well does feature j reproduce feature i's model of y?
    def r2(i, j):
        var = np.var(te_p[i])
        if var < 1e-4:  # constant / no-signal feature, same guard shap uses
            return 0.0
        m = _fit(X_tr[:, j:j + 1], tr_p[i], X_te[:, j:j + 1], te_p[i],
                 learning_rate, subsample, max_estimators, early_stopping_rounds)
        return max(0.0, 1 - np.mean((te_p[i] - m.predict(X_te[:, j:j + 1])) ** 2) / var)

    rows = []
    for j in others_idx:
        d_tj, d_jt = 1 - r2(t, j), 1 - r2(j, t)
        rows.append({
            "feature": names[j],
            "dist_target_to_other": d_tj,
            "dist_other_to_target": d_jt,
            "dist": min(d_tj, d_jt),
        })

    out = pd.DataFrame(rows).set_index("feature")
    out["redundancy"] = 1 - out["dist"]
    return out.sort_values("dist")


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n = 600
    a = rng.randn(n)
    df = pd.DataFrame({
        "a": a,
        "a_copy": a + rng.randn(n) * 0.05,
        "a_noisy": a + rng.randn(n) * 1.0,
        "b": rng.randn(n),
        "junk": rng.randn(n),
    })
    y = (df["a"] * 2 + df["b"] + rng.randn(n) * 0.2).values

    res = feature_redundancy(df, y, "a")
    #xgboost funct returns distances for all features regarding to y
    original_dist = xgboost_distances_r2(df.to_numpy(),y)
    #assert that distance vector from "a" to others matches the original --> first row in distance matrix except the distance to itself
    np.testing.assert_allclose(original_dist[0,1:], res["dist_target_to_other"].to_numpy(), atol=1e-6, rtol=0)
    df = pd.read_csv("../data/dataset_formatted.csv")
    labels = df['length_of_stay'].copy()
    labels = labels.astype('float32')
    features = df.drop(columns=['length_of_stay','drg_day_expected','ID']).copy()
    features = features.astype('float32')
    tr = NotUseNursingDiagnoses()
    features = tr.transform(features)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels,random_state=SEED,test_size=0.25)
    print("Calculating distances for n_nursing_diag")
    res = feature_redundancy(X_test, y_test, "n_nursing_diag")
    # keep only meaningful links: R^2 > 0.1  <=>  dist < 0.9
    print(res[res["redundancy"] > 0.1])
    print("Calculating distances for weight_is_missing")
    res = feature_redundancy(X_test, y_test, "weight_is_missing")
    # keep only meaningful links: R^2 > 0.1  <=>  dist < 0.9
    print(res[res["redundancy"] > 0.1])


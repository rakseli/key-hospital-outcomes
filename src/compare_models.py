"""Diff two pickled sklearn estimators / pipelines.

usage: python compare_pickles.py old.pkl new.pkl
"""
import pickle
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
        return d['estimator']['model']


def is_estimator(x):
    return hasattr(x, "get_params")


def equal(a, b):
    if type(a) is not type(b):
        return False
    if a is b:
        return True
    try:
        if isinstance(a, float) and a != a and b != b:  # both NaN
            return True
        if hasattr(a, "toarray"):  # scipy sparse
            return a.shape == b.shape and (a != b).nnz == 0
        if isinstance(a, np.ndarray):
            return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


def collect(obj):
    """Flatten hyperparameters and fitted attributes into one dict."""
    out, nodes = {}, [("", obj)]
    if is_estimator(obj):
        for name, val in obj.get_params(deep=True).items():
            if is_estimator(val):
                nodes.append((name, val))  # its params show up as name__*
            else:
                out["param:" + name] = val
    for prefix, node in nodes:
        for attr, val in vars(node).items():
            if attr.endswith("_") and not attr.startswith("_"):
                out["attr:%s%s" % (prefix + "." if prefix else "", attr)] = val
    return out


def short(v, n=120):
    s = repr(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def main(p1, p2):
    a, b = collect(load(p1)), collect(load(p2))
    diffs = 0
    for k in sorted(set(a) - set(b)):
        print("- only in %s: %s = %s" % (p1, k, short(a[k])))
        diffs += 1
    for k in sorted(set(b) - set(a)):
        print("+ only in %s: %s = %s" % (p2, k, short(b[k])))
        diffs += 1
    for k in sorted(set(a) & set(b)):
        if not equal(a[k], b[k]):
            print("~ %s\n    %s: %s\n    %s: %s"
                  % (k, p1, short(a[k]), p2, short(b[k])))
            diffs += 1
    print("\n%d difference(s)." % diffs)


if __name__ == "__main__":
    main(*sys.argv[1:3])
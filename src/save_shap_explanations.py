"""Minimal save/load for shap.Explanation via .npz (no pickle)."""
import numpy as np
import shap

_FIELDS = (
    "values", "base_values", "data", "display_data", "instance_names",
    "feature_names", "output_names", "output_indexes", "lower_bounds",
    "upper_bounds", "error_std", "main_effects", "hierarchical_values",
    "clustering", "compute_time",
)
_NAME_LISTS = ("instance_names", "feature_names", "output_names")


def _npz(path):
    """np.savez appends .npz when missing; make load agree."""
    path = str(path)
    return path if path.endswith(".npz") else path + ".npz"


def _field(expl, name):
    """Read a field, undoing slicer wrappers so it round-trips."""
    v = getattr(expl, name, None)
    # shap returns output_indexes as a (shape, array) tuple, not the array given
    if name == "output_indexes" and isinstance(v, tuple):
        v = v[1]
    return v


def save_explanation(expl, path, y=None):
    """Write every populated field of `expl` to `path` (.npz). Returns the path written.

    For model_output="log_loss" explanations, base_values is a callable (or an
    object array of callables from `explainer(X, y)`). Pass the labels `y` of
    the explained rows and they are converted to numbers before writing.
    """
    if _needs_materialising(expl.base_values):
        if y is None:
            raise TypeError(
                "base_values is a callable / object array (model_output='log_loss'); "
                "pass y=<labels of the explained rows> so it can be evaluated."
            )
        materialise_base_values(expl, y)
    out = {}
    for f in _FIELDS:
        v = _field(expl, f)
        if v is None:
            continue
        arr = np.asarray(v)
        if arr.dtype == object:
            raise TypeError(
                f"{f} has object dtype (mixed types / sparse / ragged); it would need "
                "pickle to load. Convert it to a numeric or string array first."
            )
        out[f] = arr
    path = _npz(path)
    np.savez_compressed(path, **out)
    return path


def load_explanation(path):
    """Rebuild a shap.Explanation from `path`."""
    with np.load(_npz(path), allow_pickle=False) as d:
        kw = {}
        for f in d.files:
            v = d[f]
            if v.ndim == 0:                       # scalar base_values, compute_time, ...
                v = v.item()
            elif f in _NAME_LISTS and v.ndim == 1:  # 1-D name lists back to python lists
                v = v.tolist()
            kw[f] = v
    return shap.Explanation(**kw)


def _needs_materialising(bv):
    return callable(bv) or (isinstance(bv, np.ndarray) and bv.dtype == object)


def materialise_base_values(expl, y):
    """Turn a callable base value into one plain number per explained row.

    Background
    ----------
    With ``model_output="log_loss"`` TreeExplainer does not know the base value
    up front, because it depends on the label of each row:

        base value for label c  =  average loss over the background data,
                                   when every background row is given label c

    So instead of a number, ``explainer.expected_value`` is a *function* of the
    label. ``explainer(X, y)`` then copies that function into every row of
    ``base_values``, leaving an array of functions (dtype=object) that can't be
    saved or plotted.

    This helper calls that function once for each distinct label in ``y``,
    then fills ``expl.base_values`` with the resulting numbers, row by row.

    Parameters
    ----------
    expl : shap.Explanation
        Explanation whose ``base_values`` is a function or an object array.
        Modified in place.
    y : array-like
        The labels of the explained rows, in the same order as ``expl.values``.
        Used as given: 0/1 for classification, continuous for regression.
    """
    y = np.asarray(y)
    bv = expl.base_values

    # --- Step 1: find the function that computes a base value from a label ---
    if callable(bv):
        # base_values IS the function (Explanation built by hand from expected_value)
        base_value_of = bv
    elif isinstance(bv, np.ndarray) and bv.dtype == object and callable(bv.flat[0]):
        # base_values is an array full of copies of the function (from explainer(X, y));
        # every element is the same function, so take the first one
        base_value_of = bv.flat[0]
    elif isinstance(bv, np.ndarray) and bv.dtype == object:
        # object array of numbers wrapped in 1-element arrays -> just unwrap them
        expl.base_values = np.array([float(np.ravel(x)[0]) for x in bv])
        return expl
    else:
        # already ordinary numbers, nothing to do
        return expl

    # --- Step 2: sanity check that y lines up with the explained rows ---
    n_rows = expl.values.shape[0]
    if y.shape[0] != n_rows:
        raise ValueError(f"y has {y.shape[0]} rows, but the explanation has {n_rows}")

    # --- Step 3: compute the base value once per distinct label ---
    # For classification this is just 2 calls (labels 0 and 1).
    # For regression there may be one call per row, which is still fine.
    distinct_labels = np.unique(y)
    base_value_for_label = {}
    for label in distinct_labels:
        result = base_value_of(label)                 # array with one element
        base_value_for_label[label] = float(np.ravel(result)[0])

    # --- Step 4: give each row the base value that matches its own label ---
    per_row = np.array([base_value_for_label[label] for label in y])

    expl.base_values = per_row
    return expl
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

def infer_num_and_bin_cols(df: pd.DataFrame):
    """
    Infer counts of binary vs numerical feature columns.

    Assumption: no NaN values.

    Rule:
      - binary column: unique values are a subset of {0, 1}
      - numerical column: anything else
    """
    bin_cols = 0
    num_cols = 0

    for c in df.columns:
        uniq = set(pd.unique(df[c]))
        if uniq.issubset({0, 1}):
            bin_cols += 1
        else:
            print(f"Found numberical col: {c}")
            print(list(uniq)[:10])
            num_cols += 1

    return num_cols, bin_cols, len(df)

def make_synth_binary_data(
    n_rows: int,
    num_cols: int,
    bin_cols: int,
    noise: float = 10.0,
    bin_prob: float = 0.5,
    seed: int = 0,
):
    """
    Synthetic regression data:
      - num_cols continuous features (from make_regression)
      - bin_cols binary {0,1} features

    Returns
    -------
    X : np.ndarray of shape (n_rows, num_cols + bin_cols)
    y : np.ndarray of shape (n_rows,)
    """
    rng = np.random.default_rng(seed)
    n_informative = num_cols - 3 if num_cols > 5 else 2
    effective_rank = num_cols - 2 if num_cols > 5 else 2

    # numeric features + target
    X_num, y = make_regression(
        n_samples=n_rows,
        n_features=num_cols,
        effective_rank=effective_rank,
        n_informative = n_informative,
        noise=noise,
        random_state=seed,
    )

    # binary features
    X_bin = rng.binomial(1, bin_prob, size=(n_rows, bin_cols)).astype(np.float32)

    # combine
    X = np.hstack([X_num.astype(np.float32), X_bin])

    return X, y


if __name__ == "__main__":
    features, labels  = make_synth_binary_data(n_rows=1000,num_cols=5,bin_cols=100)
    print(f"Created dataset with features {features.shape} and labels {labels.shape}")

import pandas as pd
import time
import logging
import argparse
import functools
import matplotlib.pyplot as plt
import sys
import numpy as np
import pickle
import json

from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.linear_model import Ridge, LinearRegression
from xgboost import XGBRegressor
from scipy.stats import randint, uniform, loguniform, shapiro
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, make_scorer
from sklearn.dummy import DummyRegressor
from math import sqrt
from statistics import mean
from create_synthetic_data import infer_num_and_bin_cols, make_synth_binary_data
from safety_checks import min_freq

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/model_evaluation_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)

def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            minutes, seconds = divmod(elapsed, 60)
            logging.info(f"{func.__name__} took {int(minutes)}m {seconds:.3f}s with args {args}")
    return wrapper

class NotUseNursingDiagnoses(BaseEstimator, TransformerMixin):
    """Use all columns except nursing diagnoses"""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        #redacted to protect privacy
        self.nursing_cols = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        cols = [c for c in X_out.columns if c not in set(self.nursing_cols)]
        X_out = X_out.loc[:, cols]
        assert len(X_out.columns) == len(cols)
        return X_out


class UseExpertFeatures(BaseEstimator, TransformerMixin):
    """Use all columns except one hot encoded icd, drg and process codes"""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.not_wanted_prefixes = ["icd9_","drg_code_","proc_"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        cols = [c for c in X_out.columns if not any(not_wanted in c for not_wanted in self.not_wanted_prefixes)]
        X_out = X_out.loc[:, cols]
        assert len(X_out.columns) == len(cols)
        return X_out


class UseExpertFeaturesNotUseNursingDiagnoses(UseExpertFeatures,NotUseNursingDiagnoses):
    """Use all columns except one hot encoded icd, drg and process codes and nursing diagnoses"""

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def transform(self, X):
        X_out = X.copy()
        expert_cols = [c for c in X_out.columns if not any(not_wanted in c for not_wanted in self.not_wanted_prefixes)]
        cols = [c for c in expert_cols if c not in set(self.nursing_cols)]
        cols = list(set(cols))
        X_out = X_out.loc[:, cols]
        assert len(X_out.columns)==len(cols)
        return X_out

class UseAllFeatures(BaseEstimator, TransformerMixin):
    """Use all available features"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
    
class UseMinimalFeatures(BaseEstimator, TransformerMixin):
    """Use only features in wanted_cols"""

    def __init__(self):
        self.wanted_cols = ["drg_weight","n_icd_diag","n_procs","sex","age",
                            "major_surgery_unit","cardiovasc_unit",
                            "general_surgery_unit","special_surgery_unit",
                            "internal_med_unit","oncology_unit",
                            "orthopedy_surgery_unit","craniofac_unit",
                            "infect_diseases_unit","icu_unit"
                            ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        cols = [c for c in X_out.columns if c in set(self.wanted_cols)]
        X_out = X_out.loc[:, cols]
        assert len(X_out.columns) == len(cols)
        return X_out

class ColumnNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self,y_cols):
        self.y_cols = y_cols
        if self.y_cols is None:
            self.scaler = StandardScaler(with_mean=False)
        else:
            self.scaler = StandardScaler()
    def fit(self, X, y=None):
        if self.y_cols is not None:
            existing_cols = [c for c in X.columns if c in set(self.y_cols)]
            self.scaler.fit(X.loc[:, existing_cols])
        else:
            self.scaler.fit(X)
        return self

    def transform(self, X):
        X_out = X.copy()
        if self.y_cols is not None:
            existing_cols = [c for c in X_out.columns if c in set(self.y_cols)]
            X_out.loc[:, existing_cols] = self.scaler.transform(X_out.loc[:, existing_cols])
        else:
            X_out = self.scaler.transform(X_out)
        return X_out

SEED=37

RF_PARAM_DISTS = {
    "model__n_estimators": randint(200, 1001),
    "model__max_depth": randint(3, 15),
    "model__min_samples_split": randint(2, 20),
    "model__min_samples_leaf": randint(1, 11),
    "model__max_features": ["sqrt","log2",0.33],
}

XGB_PARAM_DISTS = {
    "model__n_estimators": randint(200, 1001),
    "model__learning_rate": loguniform(1e-3, 3e-1),
    "model__max_depth": randint(3, 15),
    "model__subsample": [0.5,0.8,1.0],   
    "model__colsample_bytree": [0.5,0.8,1.0],  
    "model__reg_alpha": uniform(0.0, 9.0),         
    "model__reg_lambda": uniform(0.0, 9.0),
}

RIDGE_PARAM_DISTS = {
    "model__alpha": loguniform(1e-4, 1e3),
}


FEATURE_SET_OPTIONS = [
    NotUseNursingDiagnoses(),
    UseExpertFeatures(),
    UseExpertFeaturesNotUseNursingDiagnoses(),
    UseMinimalFeatures(),
    UseAllFeatures()
]

MODEL_ITERATIONS = {
    'xgb':500, 
    'rf':500, 
    'ridge':500,
}

# Feature set is a categorical hyperparameter shared by all models:
COMMON_PARAM_DISTS = {
    "features": FEATURE_SET_OPTIONS
}

COLS_TO_NORMALIZE = ['age','urbanization_degree','patient_education',
                       'drg_weight','weight','height',
                       'n_nursing_diag','drg_aprv30','severity_of_illness',
                       'risk_of_mortality','arp_weight','elix_score',
                       'country_income_category','n_procs','n_icd_diag']

def save_params(params_dict,output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(params_dict, f)

def load_params(params_path):
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    return params

def load_data(args):
    df = pd.read_csv(args.data_path)
    labels = df[args.label_col].copy()
    labels = labels.astype('float32')
    features = df.drop(columns=[args.label_col,'ID']).copy()
    features = features.astype('float32')
    if args.test:
        logging.info(f"Using synthetic data for testing")
        num_cols, bin_cols, n_rows = infer_num_and_bin_cols(features)
        features, labels = make_synth_binary_data(n_rows=1000,num_cols=num_cols,bin_cols=bin_cols)
        logging.info(f"Created a dataset mimimicing original with numeric_cols={num_cols},bin_cols={bin_cols}, and n_rows={n_rows}")
        
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,random_state=SEED,test_size=0.25)
    if not args.test:
        baseline_prediction = features_test['drg_day_expected'].copy()
        features_train = features_train.drop(columns=['drg_day_expected'])
        features_test = features_test.drop(columns=['drg_day_expected'])
    else:
        baseline_prediction = np.full_like(labels_train,np.mean(labels_train))


    n_boots = 100
    boot_test_sets = []
    if args.do_final_preds:
        for b in range(n_boots):
            features_test_boot, labels_test_boot, preds_baseline_boot = resample(
                features_test, labels_test,baseline_prediction,
                replace=True,                 
                n_samples=len(labels_test),      
                random_state=SEED + b
            )
            boot_test_sets.append((features_test_boot, labels_test_boot,preds_baseline_boot))

    #add full test set as last item of boots for point estimations
    boot_test_sets.append((features_test,labels_test,baseline_prediction))

    return features_train, labels_train, boot_test_sets

def make_pipeline(model_name: str,test=False) -> Pipeline:
    name = model_name.lower()
    if test:
        y_cols = None
    else:
        y_cols = COLS_TO_NORMALIZE
    if name in ("rf", "random_forest", "randomforest"):
        return Pipeline([
            ("features", UseAllFeatures()),  # will be overridden by search
            ("model", RandomForestRegressor(n_jobs=1)),
        ])

    if name == "ridge":
        return Pipeline([
            ("features", UseAllFeatures()),
            ("scaler", ColumnNormalizer(y_cols)),
            ("model", Ridge(copy_X=False,solver='sparse_cg')),
        ])

    if name in ("xgb", "xgboost"):
        return Pipeline([
            ("features", UseAllFeatures()),
            ("model", XGBRegressor(n_jobs=1)),
        ])

    raise ValueError("Unknown model. Use one of: random_forest, xgboost, ridge")

def get_param_distributions(model_name: str,test=False) -> dict:
    name = model_name.lower()
    #use empty dict so that synthetic data can be used
    if test:
        feature_sets = {}
    else:
        feature_sets = COMMON_PARAM_DISTS

    if name in ("rf", "random_forest", "randomforest"):
        return {**feature_sets, **RF_PARAM_DISTS}
    if name == "ridge":
        return {**feature_sets, **RIDGE_PARAM_DISTS}
    if name in ("xgb", "xgboost"):
        return {**feature_sets, **XGB_PARAM_DISTS}

    raise ValueError("Unknown model. Use one of: random_forest, xgboost, ridge")


def tail_mae(y_true, y_pred, q=0.95):
    thr = np.quantile(y_true, q)
    mask = y_true >= thr
    if min_freq(y_true[mask]):
        logging.info("Minimum freq principle is not fullfilled, returning NaN")
        return None
    logging.info(f"Calculating tail MAE with quantile={q} meaning stays over {thr} days ({mask.sum()}/{len(mask)})")

    return mean_absolute_error(y_true[mask], y_pred[mask])

def mask_between(s: pd.Series, left, right, inclusive="both") -> pd.Series:
    """True where s is between [left, right] (or per inclusive)."""
    return s.between(left, right, inclusive=inclusive)

def mask_ge(s: pd.Series, x) -> pd.Series:
    """True where s >= x."""
    return s.ge(x)

def mask_gender_1(s):
    mask = s['sex'].eq(1)
    return mask

def mask_gender_0(s):
    mask = s['sex'].eq(0)
    return mask

def mask_non_italian(s):
    mask = s['italian'].eq(1)
    return mask

def mask_italian(s):
    mask = s['italian'].eq(0)
    return mask

def mask_age_18_44(s: pd.Series) -> pd.Series:
    return mask_between(s['age'], 18, 44)

def mask_age_45_59(s: pd.Series) -> pd.Series:
    return mask_between(s['age'], 45, 59)

def mask_age_60_74(s: pd.Series) -> pd.Series:
    return mask_between(s['age'], 60, 74)

def mask_age_75_plus(s: pd.Series) -> pd.Series:
    return mask_ge(s['age'], 75)

def masking_func_selector(subset_name):
    mask_subsets = set(["gender-0","gender-1","non-italian","italian","age-18-44","age-45-59","age-60-74","age-75-plus"])
    if subset_name not in mask_subsets:
        raise ValueError(f"Subset name should be in {mask_subsets}, {subset_name} given!")
    
    if subset_name == 'gender-0':
        return mask_gender_0

    if subset_name == 'gender-1':
        return mask_gender_1
    
    if subset_name == 'non-italian':
        return mask_non_italian
    
    if subset_name == 'italian':
        return mask_italian

    if subset_name == 'age-18-44':
        return mask_age_18_44

    if subset_name == 'age-45-59':
        return mask_age_45_59
    
    if subset_name == 'age-60-74':
        return mask_age_60_74

    if subset_name == 'age-75-plus':
        return mask_age_75_plus



def format_output_values(score_lists,alpha=0.05):
    point_estimate = score_lists[0][0]
    lo, hi = np.quantile(score_lists[1], [alpha/2, 1 - alpha/2])
    return (float(point_estimate),float(lo), float(hi))

@timing
def run_search(model_name,features,labels,n_iter,test):
    result_dict = {}
    pipe = make_pipeline(model_name,test=test)
    param_dists = get_param_distributions(model_name,test=test)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dists,
        n_iter=n_iter,
        cv=5,
        n_jobs=8,
        verbose=2,
        pre_dispatch=8,
        random_state=SEED,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        refit=False
    )

    search.fit(features, labels)
    results = pd.DataFrame(search.cv_results_)
    result_dict['model']=model_name
    result_dict['best_params']=search.best_params_
    result_dict['best_score']=search.best_score_
    result_dict['mean_fit_time']=results.loc[search.best_index_, "mean_fit_time"]
    result_dict['estimator']=search.estimator
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to CSV file")
    parser.add_argument("--label_col", required=True, help="Label column name")
    parser.add_argument("--model",default=None, help="Model: random_forest | xgboost | ridge | None (fit all models)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--do_final_preds",action="store_true")
    parser.add_argument("--best_params_path",default="../results/best_params.pkl")

    args = parser.parse_args()

    features_train, labels_train, boot_test_sets = load_data(args)

    if not args.test:
        logging.info(f"Train set target ({args.label_col}) statistics: {labels_train.describe()}")
        logging.info(f"Train set target normality (unreliable p values for samples larger 5K, stat close to 1 support normality) {shapiro(labels_train)}")
        ax = labels_train.plot(kind="hist", bins=30, edgecolor="black")
        ax.set_xlabel(f"{args.label_col}")
        ax.set_title(f"Histogram of {args.label_col}")
        plt.tight_layout()
        plt.savefig(f"../results/training_set_{args.label_col}_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    del features_train, labels_train, boot_test_sets

    if args.model is None:
        models = ['ridge','rf','xgb']
    else:
        models = [args.model]

    if not args.do_final_preds:
        all_results = []
        for model_name in models:
            if args.test:
                n_iter = 10
            else:
                n_iter = MODEL_ITERATIONS[model_name]

            logging.info(f"For model {model_name}, searching params from pool of {n_iter}")
            features_train, labels_train, boot_test_sets = load_data(args)
            res = run_search(model_name,n_iter=n_iter,features=features_train,labels=labels_train,test=args.test)
            logging.info(f"Res: {res}")
            all_results.append(res)

            #the best score is the largest negated value of the metric, thus returning the max of error
            best_results = max(all_results, key=lambda d: d["best_score"])
            logging.info(f"Best model from all candidates is {best_results['model']}")
            logging.info(f"Best model parameters are {best_results['best_params']}")
        save_params(best_results,args.best_params_path)
    else:
        logging.info(f"Loading best estimator and params from {args.best_params_path}")
        best_results = load_params(args.best_params_path)
        logging.info(best_results)
    
    if args.test or args.do_final_preds:
        features_train, labels_train, boot_test_sets = load_data(args)
        estimator = best_results['estimator']
        best_params = best_results['best_params'] 
        new_pipe = clone(estimator)
        new_pipe.set_params(**best_params)
        #Out of random cross validation context, set seed for model and allow it use all available computing resources 
        new_pipe.set_params(model__n_jobs=-1, model__random_state=SEED)
        logging.info("Fitting model with full training set")
        new_pipe.fit(features_train, labels_train)
        final_model_path = "../results/final_model.pkl"
        logging.info(f"Saving the final model into: {final_model_path}")
        save_params(new_pipe,final_model_path)
        # Predicts the training-set mean of y for every sample
        mean_baseline = DummyRegressor(strategy="mean")
        # Predicts the training-set median of y for every sample
        median_baseline = DummyRegressor(strategy="median")
        mean_baseline.fit(features_train, labels_train)
        median_baseline.fit(features_train, labels_train)
        all_test_results = {
            "best_estimator_mae":[[],[]],
            "best_estimator_mse": [[],[]],
            "best_estimator_tailmae":[[],[]],
            "baseline_drg_expected_mae":[[],[]],
            "baseline_drg_expected_mse":[[],[]],
            "baseline_drg_expected_tailmae":[[],[]],
            'baseline_pred_mean_mae': [[],[]],
            'baseline_pred_mean_mse': [[],[]],
            'baseline_pred_mean_tailmae': [[],[]],
            'baseline_pred_median_mae': [[],[]],
            'baseline_pred_median_mse': [[],[]],
            'baseline_pred_median_tailmae': [[],[]]
        }
        sub_groups =  ["gender-0","gender-1","non-italian","italian","age-18-44","age-45-59","age-60-74","age-75-plus"]
        new_keys = []
        for sub_m in sub_groups:
            for k in all_test_results.keys():
                new_keys.append(f"{k}-{sub_m}")
        
        for n in new_keys:
            all_test_results[n]=[[],[]]



        for i,boot in enumerate(boot_test_sets):
            if i == len(boot_test_sets) - 1:
                logging.info(f"Calculating point estimate of test performance using full test set (last item of the boot list)")
                index_to_append = 0
            else:
                index_to_append = 1
                logging.info(f"Calculating test scores for boot {i}")
            features_test_boot, labels_test_boot,baseline_prediction = boot
            labels_pred = new_pipe.predict(features_test_boot)
            # general metrics
            all_test_results['best_estimator_mae'][index_to_append].append(mean_absolute_error(labels_test_boot, labels_pred))
            all_test_results['best_estimator_mse'][index_to_append].append(mean_squared_error(labels_test_boot,labels_pred))
            all_test_results['best_estimator_tailmae'][index_to_append].append(tail_mae(labels_test_boot,labels_pred))
            if not args.test:
                all_test_results['baseline_drg_expected_mae'][index_to_append].append(mean_absolute_error(labels_test_boot, baseline_prediction))
                all_test_results['baseline_drg_expected_mse'][index_to_append].append(mean_squared_error(labels_test_boot,baseline_prediction))
                all_test_results['baseline_drg_expected_tailmae'][index_to_append].append(tail_mae(labels_test_boot,baseline_prediction))

            y_pred_mean = mean_baseline.predict(features_test_boot)
            y_pred_median = median_baseline.predict(features_test_boot)
            all_test_results['baseline_pred_mean_mae'][index_to_append].append(mean_absolute_error(labels_test_boot, y_pred_mean))
            all_test_results['baseline_pred_mean_mse'][index_to_append].append(mean_squared_error(labels_test_boot,y_pred_mean))
            all_test_results['baseline_pred_mean_tailmae'][index_to_append].append(tail_mae(labels_test_boot,y_pred_mean))
            all_test_results['baseline_pred_median_mae'][index_to_append].append(mean_absolute_error(labels_test_boot, y_pred_median))
            all_test_results['baseline_pred_median_mse'][index_to_append].append(mean_squared_error(labels_test_boot,y_pred_median))
            all_test_results['baseline_pred_median_tailmae'][index_to_append].append(tail_mae(labels_test_boot,y_pred_median))
            for sub_m in sub_groups:
                masking_func = masking_func_selector(sub_m)
                mask = masking_func(features_test_boot)
                y_true_sub = labels_test_boot[mask]
                y_pred_sub = labels_pred[mask]
                y_pred_mean_sub = y_pred_mean[mask]
                y_pred_median_sub = y_pred_median[mask]
                y_pred_baseline_drg_expected_sub = baseline_prediction[mask]
                if min_freq(y_true_sub):
                    logging.info(f"Sub group {sub_m} is not safe to report, excluding the metric...")
                    continue

                logging.info(f"Calculating subgroup {sub_m} scores. Subgroup contains {mask.sum()}/{len(mask)} patients")
                all_test_results[f"best_estimator_mae-{sub_m}"][index_to_append].append(mean_absolute_error(y_true_sub,y_pred_sub))
                all_test_results[f"best_estimator_mse-{sub_m}"][index_to_append].append(mean_squared_error(y_true_sub,y_pred_sub))
                all_test_results[f"best_estimator_tailmae-{sub_m}"][index_to_append].append(tail_mae(y_true_sub,y_pred_sub))

                if not args.test:
                    all_test_results[f'baseline_drg_expected_mae-{sub_m}'][index_to_append].append(mean_absolute_error(y_true_sub, y_pred_baseline_drg_expected_sub))
                    all_test_results[f'baseline_drg_expected_mse-{sub_m}'][index_to_append].append(mean_squared_error(y_true_sub,y_pred_baseline_drg_expected_sub))
                    all_test_results[f'baseline_drg_expected_tailmae-{sub_m}'][index_to_append].append(tail_mae(y_true_sub,y_pred_baseline_drg_expected_sub))
                
                all_test_results[f'baseline_pred_mean_mae-{sub_m}'][index_to_append].append(mean_absolute_error(y_true_sub, y_pred_mean_sub))
                all_test_results[f'baseline_pred_mean_mse-{sub_m}'][index_to_append].append(mean_squared_error(y_true_sub,y_pred_mean_sub))
                all_test_results[f'baseline_pred_mean_tailmae-{sub_m}'][index_to_append].append(tail_mae(y_true_sub,y_pred_mean_sub))

                all_test_results[f'baseline_pred_median_mae-{sub_m}'][index_to_append].append(mean_absolute_error(y_true_sub, y_pred_median_sub))
                all_test_results[f'baseline_pred_median_mse-{sub_m}'][index_to_append].append(mean_squared_error(y_true_sub,y_pred_median_sub))
                all_test_results[f'baseline_pred_median_tailmae-{sub_m}'][index_to_append].append(tail_mae(y_true_sub,y_pred_median_sub))

        logging.info("Done calculating test scores")
        results = {k: format_output_values(v) for k, v in all_test_results.items()}
        rmses = {k.replace("mse","rmse"): (sqrt(v[0]),sqrt(v[1]),sqrt(v[2])) for k,v in results.items() if "mse" in k}
        all_results = results | rmses
        
        


        with open(f"../results/final_test_results_{args.label_col}.json","w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        logging.info(all_results)

if __name__ == "__main__":
    main()

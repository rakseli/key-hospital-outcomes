# key-hospital-outcomes

Code for replicating the analysis in [article]()

The pipeline predicts hospital **length of stay** from administrative and clinical admission data. It compares Ridge, Random Forest and XGBoost regressors against simple baselines, then explains the final model with SHAP, a feature redundancy analysis and partial dependence plots.

> **Note on redacted code**
> Some parts of the code (column mappings, categorical column lists, nursing diagnosis columns, etc.) are redacted to protect patient privacy. They are marked with `#redacted to protect privacy`. Researchers who have been granted access to the data can request the missing parts.

## Requirements
The code can be run using a container defined in `container.def`
## Setup
All scripts use paths relative to `src/` (`../data`, `../results`, `../logs`), so run them from inside `src/`. Create the output directories first:

```bash
mkdir -p data results logs
cd src
```
Place the source data in `data/`:

- `data/dataset_english.csv`: dataset
- `data/dataset_input_unit_categories.csv`: inpatient unit categories, joined on the index
## Research pipeline

### 1. Create the dataset: `create_dataset.py`

```bash
python create_dataset.py --task los
```

Cleans and encodes the raw data, then writes `data/dataset_formatted.csv`:

### 2. Model selection and evaluation: `train_and_eval_models.py`

**a) Hyperparameter search** (training set only):

```bash
python train_and_eval_models.py --data_path ../data/dataset_formatted.csv --label_col length_of_stay
```
**b) Final fit and test evaluation:**

```bash
python train_and_eval_models.py --data_path ../data/dataset_formatted.csv --label_col length_of_stay --do_final_preds
```

- Refits the best pipeline on the full training set and saves it to `results/final_model.pkl`.
- Evaluates it on the test set alongside three baselines: the DRG expected length of stay (`drg_day_expected`), the training mean, and the training median.
- Writes `results/final_test_results_length_of_stay.json`.
### 3. SHAP analysis: `compute_shap_values.py`

```bash
python compute_shap_values.py --data_path ../data/dataset_formatted.csv            # interventional (5,000-row background)
python compute_shap_values.py --data_path ../data/dataset_formatted.csv --tree_mode  # tree-path-dependent
```
### 4. Variable redundancy: `redundancy_analysis.py`

```bash
python redundancy_analysis.py
```

Measures how redundant one feature is with every other feature, using the same `xgboost_distances_r2` metric as `shap.utils.hclust` but without building the full p × p matrix. The script first checks its output against SHAP's implementation on a toy example, then reports the features redundant with `n_nursing_diag` and `weight_is_missing` (R² > 0.1).

### 5. Partial dependence: `pdp_analysis.py`

```bash
python pdp_analysis.py --data_path ../data/dataset_formatted.csv
```
Plots the partial dependence of predicted length of stay on the number of nursing diagnoses and saves it to `results/pdp_n_nursing_diag.png`.

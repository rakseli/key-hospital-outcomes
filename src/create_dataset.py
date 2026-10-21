import numpy as np
import pandas as pd
import logging
import argparse
import time
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/data_creation_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)

data_path = "../data/dataset_english.csv"
input_unit_path = "../data/dataset_input_unit_categories.csv"

#redacted to protect privacy
input_unit_column_map = {}
#redacted to protect privacy
columns_map = {}

#redacted to protect privacy
cat_cols = []

#ordinal columns
ordinal_cols = ["urbanization_degree","patient_education","severity_of_illness","risk_of_mortality","country_income_category"]


parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["los", "transfer"], required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.info(f"Reading data from {data_path}")
    df = pd.read_csv(data_path,header=0, index_col=0)
    logging.info(f"Raw dataset contains {len(df)} rows")

    logging.info(f"Renaming columns based on column map: {columns_map}")
    df = df.rename(columns=columns_map)

    logging.info(f"Setting gender to 0 or 1")
    df['sex']-=1


    logging.info(f"Reading the input unit data from {input_unit_path}")
    df_input_unit = pd.read_csv(input_unit_path,header=0, index_col=0)

    logging.info(f"Renaming columns based on column map: {input_unit_column_map}")
    df_input_unit = df_input_unit.rename(columns=input_unit_column_map)
    df_input_unit = df_input_unit.fillna(0)

    inpatient_unit_cols_to_drop = ["inpatient_unit","inpatient_unit_desc"]
    logging.info(f"Dropping unnecessary columns from input unit data: {inpatient_unit_cols_to_drop}")
    df_input_unit = df_input_unit.drop(columns=inpatient_unit_cols_to_drop)
    
    logging.info(f"Joining dataframes")
    full_df = df.join(df_input_unit)
    logging.info(f"Full dataset contains {len(full_df.columns.values)} columns")

    income_cols = ["lowincome", "lowmidincome", "upmidincome", "highincome"]
    logging.info(f"Combining income country cols {income_cols} into one")
    vals = full_df[income_cols].to_numpy()
    n_flags = vals.sum(axis=1)
    codes = np.where(n_flags == 1, vals.argmax(axis=1), np.nan)
    full_df["country_income_category"] = pd.Series(codes, index=df.index)
    logging.info(f"Number of columns after adding country income: {len(full_df.columns.values)}")

    logging.info(f"Setting categorical variables to categorical data format: {cat_cols}")
    full_df[cat_cols] = full_df[cat_cols].astype("category")
    categorical_value_descriptions = full_df.describe(include="category")
    for c in cat_cols:
        logging.info(f" Variable: {c}, count: {categorical_value_descriptions.loc['count',c]}, unique: {categorical_value_descriptions.loc['unique',c]}, top: {categorical_value_descriptions.loc['top',c]}, freq: {categorical_value_descriptions.loc['freq',c]}")
    
    #redacted to protect privacy
    cols_to_drop = []
    cols_to_drop.extend(income_cols)

    for c in cols_to_drop:
        if c in cat_cols:
            cat_cols.remove(c)
    logging.info(f"Number of columns before dropping: {len(full_df.columns.values)}")
    logging.info(f"Dropping unnecessary col, n={len(cols_to_drop)}")
    dif = set(cols_to_drop).difference(set(full_df.columns.values))
    logging.info(f" Diff: {dif} not in columns")
    full_df = full_df.drop(columns=cols_to_drop)
    #format missing values to nan
    logging.info(f"Number of columns after dropping: {len(full_df.columns.values)}")
    logging.info("Formatting numerical missing value identifiers to NaN")
    #urbanization_degree missing values coded as 999
    mask = (full_df["urbanization_degree"]==999)
    full_df.loc[mask, "urbanization_degree"] = np.nan
    #patient_education missing values coded as 6
    mask = (full_df["patient_education"]==6)
    full_df.loc[mask, "patient_education"] = np.nan
    #patient_employment missing values coded as 999
    mask = (full_df["patient_employment"]==999)
    full_df.loc[mask, "patient_employment"] = np.nan
    #patient_occupation missing values coded as 999
    mask = (full_df["patient_occupation"]==999)
    full_df.loc[mask, "patient_occupation"] = np.nan
    #patient_social_status missing values coded as 999
    mask = (full_df["patient_social_status"]==999)
    full_df.loc[mask, "patient_social_status"] = np.nan
    #patient_marital_status missing values coded as 999
    mask = (full_df["patient_marital_status"]==999)
    full_df.loc[mask, "patient_marital_status"] = np.nan
    logging.info(f"Setting malforded '3' in nursing diagnoses to zero (0)")
    #redacted to protect privacy
    missing_summary = (full_df[cat_cols].isna().agg(["sum", "mean"]).T.rename(columns={"sum": "missing_count", "mean": "missing_fraction"}))
    missing_summary["missing_percent"] = 100 * missing_summary["missing_fraction"]
    missing_summary = missing_summary[missing_summary['missing_percent']>0]
    logging.info(f"Summary of missing values: {missing_summary}")
    categorical_missing_value_cols = ["urbanization_degree", "patient_education",
                                  "patient_marital_status","patient_social_status",
                                 "patient_occupation","patient_employment"
                                 ]
    logging.info(f"Number of columns before adding any encoded categorical columns: {len(full_df.columns.values)}")

    logging.info(f"Adding binary indicator 'is_missing' for columns {categorical_missing_value_cols}")
    for c in categorical_missing_value_cols:
        full_df[f"{c}_is_missing"] = full_df[c].isna().astype(int)
        full_df[f"{c}_is_missing"] = full_df[f"{c}_is_missing"].astype("category")
        cat_cols.append(f"{c}_is_missing")
    

    for c in categorical_missing_value_cols:
        logging.info(f"Replacing missing values in {c} with mode {full_df[c].mode().iloc[0]}")
        full_df[c] = full_df[c].fillna(full_df[c].mode().iloc[0])


    logging.info(f"Formatting weight as nan if weight<20 or weight>600")
    mask = (full_df["weight"]<20) | (full_df["weight"]>600)
    full_df.loc[mask, "weight"] = np.nan
    logging.info(f"Missing or malformed weight values: {mask.sum()}/{len(full_df)}")
    logging.info(f"Formatting height as nan if height<55 or height>272")
    mask = (full_df["height"]<55) | (full_df["height"]>272)
    logging.info(f"Missing or malformed height values: {mask.sum()}/{len(full_df)}")
    full_df.loc[mask, "height"] = np.nan

    numerical_cols = list(set(full_df.columns.values).difference(set(cat_cols)))
    missing_summary = (full_df[numerical_cols].isna().agg(["sum", "mean"]).T
                       .rename(columns={"sum": "missing_count", "mean": "missing_fraction"}))
    missing_summary["missing_percent"] = 100 * missing_summary["missing_fraction"]
    missing_summary = missing_summary[missing_summary['missing_percent']>0]
    logging.info(f"Missing numerical values: {missing_summary}")

    numerical_missing_value_cols = ["weight","height"]
    logging.info(f"Adding binary indicator 'is_missing' for columns {numerical_missing_value_cols}")
    for c in numerical_missing_value_cols:
        full_df[f"{c}_is_missing"] = full_df[c].isna().astype(int)
        full_df[f"{c}_is_missing"] = full_df[f"{c}_is_missing"].astype("category")
        cat_cols.append(f"{c}_is_missing")
    
    for c in numerical_missing_value_cols:
        logging.info(f"Replacing missing values in {c} with median {full_df[c].median()}")
        full_df[c] = full_df[c].fillna(full_df[c].median())

    logging.info(f"Dropping one row that is missing the multiple categorical variables")
    #redacted to protect privacy

    logging.info(f"Formatting ordinal columns to be from 0 to N instead of 1 to N (country_income_category already starting from 0)")
    ordinal_cols_to_modify = ["urbanization_degree","patient_education","severity_of_illness","risk_of_mortality"]
    for o in ordinal_cols_to_modify:
        full_df[o] = full_df[o].astype(int)
        full_df[o]-=1

    #redacted to protect privacy
    proc_cols = []
    #redacted to protect privacy
    icd_cols = []
    logging.info(f"Adding columns for diagnose and process counts (sums over cols {proc_cols}), and {icd_cols}")
    full_df['n_procs'] = full_df[proc_cols].notna().sum(axis=1)
    full_df['n_icd_diag'] = full_df[icd_cols].notna().sum(axis=1)
    cols_to_drop_after_encoding = []
    logging.info("Starting to one-hot encode the categorical data")
    logging.info("Encoding procs")
    prev_n_cols = len(full_df.columns.values)
    #redacted to protect privacy
    cols = []
    full_df[cols] = full_df[cols].astype("string")
    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()          
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("proc_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)

    new_n_cols = len(full_df.columns.values)    
    logging.info("Procs encoded")
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing processes added to data")
    logging.info("Encoding diagnoses")
    #redacted to protect privacy
    cols = []
    full_df[cols] = full_df[cols].astype("string")
    prev_n_cols = len(full_df.columns.values)

    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()         
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("icd9_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)

    new_n_cols = len(full_df.columns.values)    
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing diagnoses added to data")
    logging.info("Diagnoses encoded")

    logging.info("Encoding mdc")
    cols = ["mdc"]
    full_df[cols] = full_df[cols].astype("string")
    prev_n_cols = len(full_df.columns.values)

    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()         
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("mdc_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)
    new_n_cols = len(full_df.columns.values)    
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing mdc added to data")
    logging.info("Mdc encoded")

    logging.info("Encoding DRG codes")
    cols = ["drg_code"]
    full_df[cols] = full_df[cols].astype("string")
    prev_n_cols = len(full_df.columns.values)

    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()         
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("drg_code_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)
    new_n_cols = len(full_df.columns.values)    
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing drg codes added to data")
    logging.info("Drg codes encoded")

    logging.info("Encoding discharge")
    full_df['residence'] = full_df['residence'].str.lower().str.replace(' ', '_')
    cols = ["residence"]
    full_df[cols] = full_df[cols].astype("string")
    prev_n_cols = len(full_df.columns.values)

    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()         
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("residence_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)

    new_n_cols = len(full_df.columns.values)    
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing discharge added to data")
    logging.info("Discharge encoded")

    logging.info("Encoding patient marital status")
    cols = ["patient_marital_status"]
    full_df[cols] = full_df[cols].astype("string")
    prev_n_cols = len(full_df.columns.values)

    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()         
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("patient_marital_status_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)

    new_n_cols = len(full_df.columns.values)    
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing patient marital status added to data")
    logging.info("Patient marital status encoded")

    logging.info("Encoding patient social status")
    cols = ["patient_social_status"]
    full_df[cols] = full_df[cols].astype("string")
    prev_n_cols = len(full_df.columns.values)

    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()         
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("patient_social_status_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)
    new_n_cols = len(full_df.columns.values)    
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing patient social status added to data")
    logging.info("Patient social status encoded")

    logging.info("Encoding patient occupation")
    cols = ["patient_occupation"]
    full_df[cols] = full_df[cols].astype("string")
    prev_n_cols = len(full_df.columns.values)

    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()         
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("patient_occupation_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)
    new_n_cols = len(full_df.columns.values)    
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing patient occupation added to data")
    logging.info("Patient occupation encoded")

    logging.info("Encoding patient employment")
    cols = ["patient_employment"]
    full_df[cols] = full_df[cols].astype("string")
    prev_n_cols = len(full_df.columns.values)

    ind = (
        full_df[cols]
        .stack()                    
        .str.get_dummies()         
        .groupby(level=0).max()     
        .reindex(full_df.index, fill_value=0)
    )
    ind = ind.add_prefix("patient_employment_")
    full_df = full_df.join(ind)
    cols_to_drop_after_encoding.extend(cols)
    new_n_cols = len(full_df.columns.values)    
    logging.info(f"{new_n_cols-prev_n_cols} new columns representing patient employment added to data")
    logging.info("Patient employment encoded")

    logging.info(f"Dropping cols {cols_to_drop_after_encoding} that have been encoded")
    full_df = full_df.drop(columns=cols_to_drop_after_encoding)

    logging.info(f"Encoding drg_type 1=surgical and 0=medical")
    full_df["drg_type"] = full_df["drg_type"].map({"surgical": 1, "medical": 0})

    exclude = ['drg_weight','arp_weight']
    cols = full_df.columns.difference(exclude)
    logging.info(f"Encoding columsn as int32 (except {exclude})")
    full_df[cols] = full_df[cols].astype('int32')
    output_path = "../data/dataset_formatted.csv"
    logging.info(f"In total {len(full_df.columns.values)} columns in data")
    logging.info(f"Writing formatted data to file: {output_path}")
    full_df.to_csv(output_path,index=True)
    logging.info(f"Done")


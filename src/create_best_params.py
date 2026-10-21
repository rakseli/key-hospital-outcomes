from train_and_eval_models import (
        NotUseNursingDiagnoses,
        make_pipeline,
        load_params,
        save_params)

if __name__ == "__main__":
    save_dict = {}
    best_found_params = {'features': NotUseNursingDiagnoses(), 
                        'model__colsample_bytree': 0.5, 
                        'model__learning_rate': 0.05740169654656397,
                         'model__max_depth': 6,
                          'model__n_estimators': 635,
                          'model__reg_alpha': 2.088940771137303,
                           'model__reg_lambda': 6.508793986161287,
                            'model__subsample': 0.5,
                            'model__enable_categorical':False}
    pipe = make_pipeline("xgb")
    pipe.set_params(**best_found_params)
    print(pipe)
    print("Saving best params")
    params_path = "../results/best_params_old_version.pkl"
    save_dict['best_params']=best_found_params
    save_dict['estimator']=pipe
    save_params(save_dict,params_path)
    d = load_params(params_path)
    print(f"Succesfully loaded params: {d}")
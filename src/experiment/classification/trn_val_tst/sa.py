import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd
import xgboost as xgb
from src.experiment.routines import eval_classification, eval_loss, save_feature_importance
from typing import List
from catboost import CatBoost
import lightgbm as lgb
import wandb
from src.datamodules.cross_validation import RepeatedStratifiedKFoldCVSplitter
from src.experiment.classification.shap import explain_shap
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from datetime import datetime
from tqdm import tqdm
from src import utils
import pickle


log = utils.get_logger(__name__)


def process(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.name

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.perform_split()
    feature_names = datamodule.get_feature_names()
    class_names = datamodule.get_class_names()
    outcome_name = datamodule.get_outcome_name()
    df = datamodule.get_df()
    df['pred'] = 0
    ids_tst = datamodule.ids_tst
    if ids_tst is not None and len(ids_tst) > 0:
        is_test = True
    else:
        is_test = False

    cv_splitter = RepeatedStratifiedKFoldCVSplitter(
        datamodule=datamodule,
        is_split=config.cv_is_split,
        n_splits=config.cv_n_splits,
        n_repeats=config.cv_n_repeats,
        random_state=config.seed
    )

    best = {}
    if config.direction == "min":
        best["optimized_metric"] = np.Inf
    elif config.direction == "max":
        best["optimized_metric"] = 0.0

    cv_progress = pd.DataFrame(columns=['fold', 'optimized_metric'])

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for fold_idx, (ids_trn, ids_val) in tqdm(enumerate(cv_splitter.split())):
        datamodule.ids_trn = ids_trn
        datamodule.ids_val = ids_val
        datamodule.refresh_datasets()
        X_trn = df.loc[df.index[ids_trn], feature_names].values
        y_trn = df.loc[df.index[ids_trn], outcome_name].values
        df.loc[df.index[ids_trn], f"fold_{fold_idx:04d}"] = "train"
        X_val = df.loc[df.index[ids_val], feature_names].values
        y_val = df.loc[df.index[ids_val], outcome_name].values
        df.loc[df.index[ids_val], f"fold_{fold_idx:04d}"] = "val"
        if is_test:
            X_tst = df.loc[df.index[ids_tst], feature_names].values
            y_tst = df.loc[df.index[ids_tst], outcome_name].values
            df.loc[df.index[ids_tst], f"fold_{fold_idx:04d}"] = "test"

        if 'csv' in config.logger:
            config.logger.csv["version"] = f"fold_{fold_idx}"
        if 'wandb' in config.logger:
            config.logger.wandb["version"] = f"fold_{fold_idx}_{start_time}"

        # Init lightning loggers
        loggers: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    loggers.append(hydra.utils.instantiate(lg_conf))

        log.info("Logging hyperparameters!")
        utils.log_hyperparameters_sa(config, loggers)

        if config.model_type == "xgboost":
            model_params = {
                'num_class': config.model_xgboost.output_dim,
                'booster': config.model_xgboost.booster,
                'eta': config.model_xgboost.learning_rate,
                'max_depth': config.model_xgboost.max_depth,
                'gamma': config.model_xgboost.gamma,
                'sampling_method': config.model_xgboost.sampling_method,
                'subsample': config.model_xgboost.subsample,
                'objective': config.model_xgboost.objective,
                'verbosity': config.model_xgboost.verbosity,
                'eval_metric': config.model_xgboost.eval_metric,
            }

            dmat_trn = xgb.DMatrix(X_trn, y_trn, feature_names=feature_names)
            dmat_val = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
            if is_test:
                dmat_tst = xgb.DMatrix(X_tst, y_tst, feature_names=feature_names)

            evals_result = {}
            model = xgb.train(
                params=model_params,
                dtrain=dmat_trn,
                evals=[(dmat_trn, "train"), (dmat_val, "val")],
                num_boost_round=config.max_epochs,
                early_stopping_rounds=config.patience,
                evals_result=evals_result
            )

            y_trn_pred_prob = model.predict(dmat_trn)
            y_val_pred_prob = model.predict(dmat_val)
            y_trn_pred_raw = model.predict(dmat_trn, output_margin=True)
            y_val_pred_raw = model.predict(dmat_val, output_margin=True)
            y_trn_pred = np.argmax(y_trn_pred_prob, 1)
            y_val_pred = np.argmax(y_val_pred_prob, 1)
            if is_test:
                y_tst_pred_prob = model.predict(dmat_tst)
                y_tst_pred_raw = model.predict(dmat_tst, output_margin=True)
                y_tst_pred = np.argmax(y_tst_pred_prob, 1)

            loss_info = {
                'epoch': list(range(len(evals_result['train'][config.model_xgboost.eval_metric]))),
                'train/loss': evals_result['train'][config.model_xgboost.eval_metric],
                'val/loss': evals_result['val'][config.model_xgboost.eval_metric]
            }

            fi = model.get_score(importance_type='weight')
            feature_importances = pd.DataFrame.from_dict({'feature': list(fi.keys()), 'importance': list(fi.values())})

        elif config.model_type == "catboost":
            model_params = {
                'classes_count': config.model_catboost.output_dim,
                'loss_function': config.model_catboost.loss_function,
                'learning_rate': config.model_catboost.learning_rate,
                'depth': config.model_catboost.depth,
                'min_data_in_leaf': config.model_catboost.min_data_in_leaf,
                'max_leaves': config.model_catboost.max_leaves,
                'task_type': config.model_catboost.task_type,
                'verbose': config.model_catboost.verbose,
                'iterations': config.model_catboost.max_epochs,
                'early_stopping_rounds': config.model_catboost.patience
            }

            model = CatBoost(params=model_params)
            model.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True)
            model.set_feature_names(feature_names)

            y_trn_pred_prob = model.predict(X_trn, prediction_type="Probability")
            y_val_pred_prob = model.predict(X_val, prediction_type="Probability")
            y_trn_pred_raw = model.predict(X_trn, prediction_type="RawFormulaVal")
            y_val_pred_raw = model.predict(X_val, prediction_type="RawFormulaVal")
            y_trn_pred = np.argmax(y_trn_pred_prob, 1)
            y_val_pred = np.argmax(y_val_pred_prob, 1)
            if is_test:
                y_tst_pred_prob = model.predict(X_tst, prediction_type="Probability")
                y_tst_pred_raw = model.predict(X_tst, prediction_type="RawFormulaVal")
                y_tst_pred = np.argmax(y_tst_pred_prob, 1)

            metrics_trn = pd.read_csv(f"catboost_info/learn_error.tsv", delimiter="\t")
            metrics_val = pd.read_csv(f"catboost_info/test_error.tsv", delimiter="\t")
            loss_info = {
                'epoch': metrics_trn.iloc[:, 0],
                'train/loss': metrics_trn.iloc[:, 1],
                'val/loss': metrics_val.iloc[:, 1]
            }

            feature_importances = pd.DataFrame.from_dict({'feature': model.feature_names_, 'importance': list(model.feature_importances_)})

        elif config.model_type == "lightgbm":
            model_params = {
                'num_class': config.model_lightgbm.output_dim,
                'objective': config.model_lightgbm.objective,
                'boosting': config.model_lightgbm.boosting,
                'learning_rate': config.model_lightgbm.learning_rate,
                'num_leaves': config.model_lightgbm.num_leaves,
                'device': config.model_lightgbm.device,
                'max_depth': config.model_lightgbm.max_depth,
                'min_data_in_leaf': config.model_lightgbm.min_data_in_leaf,
                'feature_fraction': config.model_lightgbm.feature_fraction,
                'bagging_fraction': config.model_lightgbm.bagging_fraction,
                'bagging_freq': config.model_lightgbm.bagging_freq,
                'verbose': config.model_lightgbm.verbose,
                'metric': config.model_lightgbm.metric
            }

            ds_trn = lgb.Dataset(X_trn, label=y_trn, feature_name=feature_names)
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_trn, feature_name=feature_names)

            evals_result = {}
            model = lgb.train(
                params=model_params,
                train_set=ds_trn,
                num_boost_round=config.max_epochs,
                valid_sets=[ds_val, ds_trn],
                valid_names=['val', 'train'],
                evals_result=evals_result,
                early_stopping_rounds=config.patience,
                verbose_eval=True
            )

            y_trn_pred_prob = model.predict(X_trn, num_iteration=model.best_iteration)
            y_val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
            y_trn_pred_raw = model.predict(X_trn, num_iteration=model.best_iteration, raw_score=True)
            y_val_pred_raw = model.predict(X_val, num_iteration=model.best_iteration, raw_score=True)
            y_trn_pred = np.argmax(y_trn_pred_prob, 1)
            y_val_pred = np.argmax(y_val_pred_prob, 1)
            if is_test:
                y_tst_pred_prob = model.predict(X_tst, num_iteration=model.best_iteration)
                y_tst_pred_raw = model.predict(X_tst, num_iteration=model.best_iteration, raw_score=True)
                y_tst_pred = np.argmax(y_tst_pred_prob, 1)

            loss_info = {
                'epoch': list(range(len(evals_result['train'][config.model_lightgbm.metric]))),
                'train/loss': evals_result['train'][config.model_lightgbm.metric],
                'val/loss': evals_result['val'][config.model_lightgbm.metric]
            }

            feature_importances = pd.DataFrame.from_dict({'feature': model.feature_name(), 'importance': list(model.feature_importance())})

        elif config.model_type == "logistic_regression":
            model = LogisticRegression(
                penalty=config.logistic_regression.penalty,
                l1_ratio=config.logistic_regression.l1_ratio,
                C=config.logistic_regression.C,
                multi_class=config.logistic_regression.multi_class,
                solver=config.logistic_regression.solver,
                max_iter=config.logistic_regression.max_iter,
                tol=config.logistic_regression.tol,
                verbose=config.logistic_regression.verbose,
            ).fit(X_trn, y_trn)

            y_trn_pred_prob = model.predict_proba(X_trn)
            y_val_pred_prob = model.predict_proba(X_val)
            y_trn_pred_raw = model.predict_proba(X_trn)
            y_val_pred_raw = model.predict_proba(X_val)
            y_trn_pred = model.predict(X_trn)
            y_val_pred = model.predict(X_val)
            if is_test:
                y_tst_pred_prob = model.predict_proba(X_tst)
                y_tst_pred_raw = model.predict_proba(X_tst)
                y_tst_pred = model.predict(X_tst)

            loss_info = {
                'epoch': [0],
                'train/loss': [0],
                'val/loss': [0]
            }

            feature_importances = pd.DataFrame.from_dict(
                {
                    'feature': ['Intercept'] + feature_names,
                    'importance': [model.intercept_] + list(model.coef_.ravel())
                }
            )

        elif config.model_type == "svm":
            model = svm.SVC(
                C=config.svm.C,
                kernel=config.svm.kernel,
                decision_function_shape=config.svm.decision_function_shape,
                max_iter=config.svm.max_iter,
                tol=config.svm.tol,
                verbose=config.svm.verbose,
                probability=True
            ).fit(X_trn, y_trn)

            y_trn_pred_prob = model.predict_proba(X_trn)
            y_val_pred_prob = model.predict_proba(X_val)
            y_trn_pred_raw = model.predict_proba(X_trn)
            y_val_pred_raw = model.predict_proba(X_val)
            y_trn_pred = model.predict(X_trn)
            y_val_pred = model.predict(X_val)
            if is_test:
                y_tst_pred_prob = model.predict_proba(X_tst)
                y_tst_pred_raw = model.predict_proba(X_tst)
                y_tst_pred = model.predict(X_tst)

            loss_info = {
                'epoch': [0],
                'train/loss': [0],
                'val/loss': [0]
            }

            feature_importances = None

        else:
            raise ValueError(f"Model {config.model_type} is not supported")

        metrics_trn = eval_classification(config, class_names, y_trn, y_trn_pred, y_trn_pred_prob, loggers, 'train', is_log=True, is_save=False)
        for m in metrics_trn.index.values:
            cv_progress.at[fold_idx, f"train_{m}"] = metrics_trn.at[m, 'train']
        metrics_val = eval_classification(config, class_names, y_val, y_val_pred, y_val_pred_prob, loggers, 'val', is_log=True, is_save=False)
        for m in metrics_val.index.values:
            cv_progress.at[fold_idx, f"val_{m}"] = metrics_val.at[m, 'val']
        if is_test:
            metrics_tst = eval_classification(config, class_names, y_tst, y_tst_pred, y_tst_pred_prob, loggers, 'test', is_log=True, is_save=False)
            for m in metrics_tst.index.values:
                cv_progress.at[fold_idx, f"test_{m}"] = metrics_tst.at[m, 'test']

        if 'wandb' in config.logger:
            wandb.define_metric(f"epoch")
            wandb.define_metric(f"train/loss")
            wandb.define_metric(f"val/loss")
        eval_loss(loss_info, loggers, is_log=True, is_save=False)

        for logger in loggers:
            logger.save()
        if 'wandb' in config.logger:
            wandb.finish()

        if config.optimized_part == "train":
            metrics_main = metrics_trn
        elif config.optimized_part == "val":
            metrics_main = metrics_val
        elif config.optimized_part == "test":
            metrics_main = metrics_tst
        else:
            raise ValueError(f"Unsupported config.optimized_part: {config.optimized_part}")

        if config.direction == "min":
            if metrics_main.at[config.optimized_metric, config.optimized_part] < best["optimized_metric"]:
                is_renew = True
            else:
                is_renew = False
        elif config.direction == "max":
            if metrics_main.at[config.optimized_metric, config.optimized_part] > best["optimized_metric"]:
                is_renew = True
            else:
                is_renew = False

        if is_renew:
            best["optimized_metric"] = metrics_main.at[config.optimized_metric, config.optimized_part]
            best["model"] = model
            best['loss_info'] = loss_info

            if config.model_type == "xgboost":
                def predict_func(X):
                    X = xgb.DMatrix(X, feature_names=feature_names)
                    y = best["model"].predict(X)
                    return y
            elif config.model_type == "catboost":
                def predict_func(X):
                    y = best["model"].predict(X, prediction_type="Probability")
                    return y
            elif config.model_type == "lightgbm":
                def predict_func(X):
                    y = best["model"].predict(X, num_iteration=best["model"].best_iteration)
                    return y
            elif config.model_type == "logistic_regression":
                def predict_func(X):
                    y = best["model"].predict_proba(X)
                    return y
            elif config.model_type == "svm":
                def predict_func(X):
                    y = best["model"].predict_proba(X)
                    return y
            else:
                raise ValueError(f"Model {config.model_type} is not supported")

            best['predict_func'] = predict_func
            best['feature_importances'] = feature_importances
            best['fold'] = fold_idx
            best['ids_trn'] = ids_trn
            best['ids_val'] = ids_val
            df.loc[df.index[ids_trn], "pred"] = y_trn_pred
            df.loc[df.index[ids_val], "pred"] = y_val_pred
            for cl_id, cl in enumerate(class_names):
                df.loc[df.index[ids_trn], f"pred_prob_{cl_id}"] = y_trn_pred_prob[:, cl_id]
                df.loc[df.index[ids_val], f"pred_prob_{cl_id}"] = y_val_pred_prob[:, cl_id]
                df.loc[df.index[ids_trn], f"pred_raw_{cl_id}"] = y_trn_pred_raw[:, cl_id]
                df.loc[df.index[ids_val], f"pred_raw_{cl_id}"] = y_val_pred_raw[:, cl_id]
            if is_test:
                df.loc[df.index[ids_tst], "pred"] = y_tst_pred
                for cl_id, cl in enumerate(class_names):
                    df.loc[df.index[ids_tst], f"pred_prob_{cl_id}"] = y_tst_pred_prob[:, cl_id]
                    df.loc[df.index[ids_tst], f"pred_raw_{cl_id}"] = y_tst_pred_raw[:, cl_id]

        cv_progress.at[fold_idx, 'fold'] = fold_idx
        cv_progress.at[fold_idx, 'optimized_metric'] = metrics_main.at[config.optimized_metric, config.optimized_part]

    cv_progress.to_excel(f"cv_progress.xlsx", index=False)
    cv_ids = df.loc[:, [f"fold_{fold_idx:04d}" for fold_idx in cv_progress.loc[:,'fold'].values]]
    cv_ids.to_excel(f"cv_ids.xlsx", index=True)
    predictions = df.loc[:, [f"fold_{best['fold']:04d}", outcome_name, "pred"] + [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]]
    predictions.to_excel(f"predictions.xlsx", index=True)

    datamodule.ids_trn = best['ids_trn']
    datamodule.ids_val = best['ids_val']

    datamodule.plot_split(f"_best_{best['fold']:04d}")

    y_trn = df.loc[df.index[datamodule.ids_trn], outcome_name].values
    y_trn_pred = df.loc[df.index[datamodule.ids_trn], "pred"].values
    y_trn_pred_prob = df.loc[df.index[datamodule.ids_trn], [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values
    y_val = df.loc[df.index[datamodule.ids_val], outcome_name].values
    y_val_pred = df.loc[df.index[datamodule.ids_val], "pred"].values
    y_val_pred_prob = df.loc[df.index[datamodule.ids_val], [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values
    if is_test:
        y_tst = df.loc[df.index[datamodule.ids_tst], outcome_name].values
        y_tst_pred = df.loc[df.index[datamodule.ids_tst], "pred"].values
        y_tst_pred_prob = df.loc[df.index[datamodule.ids_tst], [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values

    metrics_trn = eval_classification(config, class_names, y_trn, y_trn_pred, y_trn_pred_prob, None, 'train', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
    metrics_names = metrics_trn.index.values
    metrics_trn_cv_mean = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names], columns=['train'])
    for metric in metrics_names:
        metrics_trn_cv_mean.at[f"{metric}_cv_mean", 'train'] = cv_progress[f"train_{metric}"].mean()
    metrics_trn = pd.concat([metrics_trn, metrics_trn_cv_mean])
    metrics_trn.to_excel(f"metrics_train_best_{best['fold']:04d}.xlsx", index=True)

    metrics_val = eval_classification(config, class_names, y_val, y_val_pred, y_val_pred_prob, None, 'val', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
    metrics_val_cv_mean = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names], columns=['val'])
    for metric in metrics_names:
        metrics_val_cv_mean.at[f"{metric}_cv_mean", 'val'] = cv_progress[f"val_{metric}"].mean()
    metrics_val = pd.concat([metrics_val, metrics_val_cv_mean])
    metrics_val.to_excel(f"metrics_val_best_{best['fold']:04d}.xlsx", index=True)

    if is_test:
        metrics_tst = eval_classification(config, class_names, y_tst, y_tst_pred, y_tst_pred_prob, None, 'test', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
        metrics_tst_cv_mean = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names], columns=['test'])
        for metric in metrics_names:
            metrics_tst_cv_mean.at[f"{metric}_cv_mean", 'test'] = cv_progress[f"test_{metric}"].mean()
        metrics_tst = pd.concat([metrics_tst, metrics_tst_cv_mean])

        metrics_val_tst_cv_mean = pd.DataFrame(index=[f"{x}_cv_mean_val_test" for x in metrics_names], columns=['val', 'test'])
        for metric in metrics_names:
            val_test_value = 0.5 * (metrics_val.at[f"{metric}_cv_mean", 'val'] + metrics_tst.at[f"{metric}_cv_mean", 'test'])
            metrics_val_tst_cv_mean.at[f"{metric}_cv_mean_val_test", 'val'] = val_test_value
            metrics_val_tst_cv_mean.at[f"{metric}_cv_mean_val_test", 'test'] = val_test_value
        metrics_val = pd.concat([metrics_val, metrics_val_tst_cv_mean.loc[:, ['val']]])
        metrics_tst = pd.concat([metrics_tst, metrics_val_tst_cv_mean.loc[:, ['test']]])
        metrics_val.to_excel(f"metrics_val_best_{best['fold']:04d}.xlsx", index=True)
        metrics_tst.to_excel(f"metrics_test_best_{best['fold']:04d}.xlsx", index=True)

    eval_loss(best['loss_info'], None, is_log=True, is_save=False, file_suffix=f"_best_{best['fold']:04d}")

    if config.optimized_part == "train":
        metrics_main = metrics_trn
    elif config.optimized_part == "val":
        metrics_main = metrics_val
    elif config.optimized_part == "test":
        metrics_main = metrics_tst
    else:
        raise ValueError(f"Unsupported config.optimized_part: {config.optimized_part}")

    if config.model_type == "xgboost":
        best["model"].save_model(f"epoch_{best['model'].best_iteration}_best_{best['fold']:04d}.model")
    elif config.model_type == "catboost":
        best["model"].save_model(f"epoch_{best['model'].best_iteration_}_best_{best['fold']:04d}.model")
    elif config.model_type == "lightgbm":
        best["model"].save_model(f"epoch_{best['model'].best_iteration}_best_{best['fold']:04d}.txt", num_iteration=best['model'].best_iteration)
    elif config.model_type == "logistic_regression":
        pickle.dump(best["model"], open(f"logistic_regression_best_{best['fold']:04d}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    elif config.model_type == "svm":
        pickle.dump(best["model"], open(f"svm_best_{best['fold']:04d}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Model {config.model_type} is not supported")

    save_feature_importance(best['feature_importances'], config.num_top_features)

    expl_data = {
        'model': best["model"],
        'predict_func': best['predict_func'],
        'df': df,
        'feature_names': feature_names,
        'class_names': class_names,
        'outcome_name': outcome_name,
        'ids_all': np.arange(df.shape[0]),
        'ids_trn': datamodule.ids_trn,
        'ids_val': datamodule.ids_val,
        'ids_tst': datamodule.ids_tst
    }

    if config.is_shap == True:
        explain_shap(config, expl_data)

    optimized_metric = config.get("optimized_metric")
    optimized_mean = config.get("optimized_mean")
    if optimized_metric:
        if optimized_mean == "":
            return metrics_main.at[optimized_metric, config.optimized_part]
        else:
            return metrics_main.at[f"{optimized_metric}_{optimized_mean}", config.optimized_part]

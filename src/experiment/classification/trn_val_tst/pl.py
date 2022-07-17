from typing import List, Optional
import torch
import hydra
from omegaconf import DictConfig
from src.models.tabnet.model import TabNetModel
from src.models.node.model import NodeModel
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from tqdm import tqdm
from pytorch_lightning.loggers import LightningLoggerBase
import numpy as np
import pandas as pd
from src.datamodules.cross_validation import RepeatedStratifiedKFoldCVSplitter
from src.experiment.classification.shap import explain_shap
from datetime import datetime
from src.experiment.routines import eval_classification, save_feature_importance
from pathlib import Path
from src import utils


log = utils.get_logger(__name__)


def process(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

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
    num_features = len(feature_names)
    config.in_dim = num_features
    con_features_ids, cat_features_ids = datamodule.get_con_cat_feature_ids()
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
    ckpt_name = config.callbacks.model_checkpoint.filename

    for fold_idx, (ids_trn, ids_val) in tqdm(enumerate(cv_splitter.split())):
        datamodule.ids_trn = ids_trn
        datamodule.ids_val = ids_val
        datamodule.refresh_datasets()
        df.loc[df.index[ids_trn], f"fold_{fold_idx:04d}"] = "train"
        df.loc[df.index[ids_val], f"fold_{fold_idx:04d}"] = "val"
        if is_test:
            df.loc[df.index[ids_tst], f"fold_{fold_idx:04d}"] = "test"

        config.callbacks.model_checkpoint.filename = ckpt_name + f"_fold_{fold_idx:04d}"

        if 'csv' in config.logger:
            config.logger.csv["version"] = f"fold_{fold_idx}"
        if 'wandb' in config.logger:
            config.logger.wandb["version"] = f"fold_{fold_idx}_{start_time}"

        # Init lightning model
        if config.model_type == "tabnet":
            config.model = config["tabnet"]
        elif config.model_type == "node":
            config.model = config["node"]
        else:
            raise ValueError(f"Unsupported model: {config.model_type}")

        log.info(f"Instantiating model <{config.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(config.model)
        model.ids_con = con_features_ids
        model.ids_cat = cat_features_ids

        # Init lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        # Init lightning loggers
        loggers: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    loggers.append(hydra.utils.instantiate(lg_conf))

        # Init lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
        )

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=loggers,
        )

        # Train the model
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        # Evaluate model on test set, using the best model achieved during training
        if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            test_dataloader = datamodule.test_dataloader()
            if test_dataloader is not None and len(test_dataloader) > 0:
                trainer.test(model, test_dataloader, ckpt_path="best")
            else:
                log.info("Test data is empty!")

        datamodule.dataloaders_evaluate = True
        trn_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        tst_dataloader = datamodule.test_dataloader()
        datamodule.dataloaders_evaluate = False

        model.produce_probabilities = True
        y_trn = df.loc[df.index[ids_trn], outcome_name].values
        y_val = df.loc[df.index[ids_val], outcome_name].values
        y_trn_pred_prob = torch.cat(trainer.predict(model, dataloaders=trn_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
        y_val_pred_prob = torch.cat(trainer.predict(model, dataloaders=val_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
        if is_test:
            y_tst = df.loc[df.index[ids_tst], outcome_name].values
            y_tst_pred_prob = torch.cat(trainer.predict(model, dataloaders=tst_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
        model.produce_probabilities = False
        y_trn_pred_raw = torch.cat(trainer.predict(model, dataloaders=trn_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
        y_val_pred_raw = torch.cat(trainer.predict(model, dataloaders=val_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
        y_trn_pred = np.argmax(y_trn_pred_prob, 1)
        y_val_pred = np.argmax(y_val_pred_prob, 1)
        if is_test:
            y_tst_pred_raw = torch.cat(trainer.predict(model, dataloaders=tst_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
            y_tst_pred = np.argmax(y_tst_pred_prob, 1)
        model.produce_probabilities = True

        if config.model_type == "tabnet":
            feature_importances_raw = np.zeros((len(feature_names)))
            model.produce_importance = True
            raw_res = trainer.predict(model, dataloaders=trn_dataloader, return_predictions=True, ckpt_path="best")
            M_explain =  torch.cat([x[0] for x in raw_res])
            model.produce_importance = False
            feature_importances_raw += M_explain.sum(dim=0).cpu().detach().numpy()
            feature_importances_raw = feature_importances_raw / np.sum(feature_importances_raw)
            feature_importances = pd.DataFrame.from_dict(
                {
                    'feature': feature_names,
                    'importance': feature_importances_raw
                }
            )
        elif config.model_type == "node":
            feature_importances = None
        else:
            raise ValueError(f"Unsupported model: {config.model_type}")

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

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=loggers,
        )

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
            if Path(f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt").is_file():
                if config.model_type == "tabnet":
                    model = TabNetModel.load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.produce_probabilities = True
                    model.eval()
                    model.freeze()
                elif config.model_type == "node":
                    model = NodeModel.load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.produce_probabilities = True
                    model.eval()
                    model.freeze()
                else:
                    raise ValueError(f"Unsupported model: {config.model_type}")
            best["model"] = model
            best["trainer"] = trainer

            def predict_func(X):
                X = np.float32(X)
                best["model"].produce_probabilities = True
                X = torch.from_numpy(X)
                tmp = best["model"](X)
                return tmp.cpu().detach().numpy()

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
    cv_ids = df.loc[:, [f"fold_{fold_idx:04d}" for fold_idx in cv_progress.loc[:, 'fold'].values]]
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

    if config.optimized_part == "train":
        metrics_main = metrics_trn
    elif config.optimized_part == "val":
        metrics_main = metrics_val
    elif config.optimized_part == "test":
        metrics_main = metrics_tst
    else:
        raise ValueError(f"Unsupported config.optimized_part: {config.optimized_part}")

    if best['feature_importances'] is not None:
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

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    optimized_mean = config.get("optimized_mean")
    if optimized_metric:
        if optimized_mean == "":
            return metrics_main.at[optimized_metric, config.optimized_part]
        else:
            return metrics_main.at[f"{optimized_metric}_{optimized_mean}", config.optimized_part]

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
from src.experiment.classification.shap import perform_shap_explanation
from datetime import datetime
from src.experiment.routines import eval_classification_sa, save_feature_importance
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

    config.logger.wandb["project"] = config.project_name

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.perform_split()
    feature_names = datamodule.get_feature_names()
    num_features = len(feature_names)
    config.in_dim = num_features
    class_names = datamodule.get_class_names()
    outcome_name = datamodule.get_outcome_name()
    df = datamodule.get_df()
    df['pred'] = 0
    ids_tst = datamodule.ids_tst
    if ids_tst is not None:
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
    cv_progress = {'fold': [], 'optimized_metric':[]}

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
            config.model = config["model_tabnet"]
        elif config.model_type == "node":
            config.model = config["model_node"]
        else:
            raise ValueError(f"Unsupported model: {config.model_type}")

        log.info(f"Instantiating model <{config.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(config.model)

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

        def shap_kernel(X):
            model.produce_probabilities = True
            X = torch.from_numpy(X)
            tmp = model(X)
            return tmp.cpu().detach().numpy()

        metrics_trn = eval_classification_sa(config, class_names, y_trn, y_trn_pred, y_trn_pred_prob, loggers, 'train', is_log=False, is_save=False)
        metrics_val = eval_classification_sa(config, class_names, y_val, y_val_pred, y_val_pred_prob, loggers, 'val', is_log=False, is_save=False)
        if is_test:
            metrics_tst = eval_classification_sa(config, class_names, y_tst, y_tst_pred, y_tst_pred_prob, loggers, 'test', is_log=False, is_save=False)

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
            best['shap_kernel'] = shap_kernel
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

        cv_progress['fold'].append(fold_idx)
        cv_progress['optimized_metric'].append(metrics_main.at[config.optimized_metric, config.optimized_part])

    cv_progress_df = pd.DataFrame(cv_progress)
    cv_progress_df.set_index('fold', inplace=True)
    cv_progress_df.to_excel(f"cv_progress.xlsx", index=True)
    cv_ids = df.loc[:, [f"fold_{fold_idx:04d}" for fold_idx in cv_progress['fold']]]
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

    metrics_trn = eval_classification_sa(config, class_names, y_trn, y_trn_pred, y_trn_pred_prob, loggers, 'train', is_log=False, is_save=True, suffix=f"_best_{best['fold']:04d}")
    metrics_val = eval_classification_sa(config, class_names, y_val, y_val_pred, y_val_pred_prob, loggers, 'val', is_log=False, is_save=True, suffix=f"_best_{best['fold']:04d}")
    if is_test:
        metrics_tst = eval_classification_sa(config, class_names, y_tst, y_tst_pred, y_tst_pred_prob, loggers, 'test', is_log=False, is_save=True, suffix=f"_best_{best['fold']:04d}")

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

    if config.is_shap == True:
        shap_data = {
            'model': best["model"],
            'shap_kernel': best['shap_kernel'],
            'df': df,
            'feature_names': feature_names,
            'class_names': class_names,
            'outcome_name': outcome_name,
            'ids_all': np.arange(df.shape[0]),
            'ids_trn': datamodule.ids_trn,
            'ids_val': datamodule.ids_val,
            'ids_tst': datamodule.ids_tst
        }
        perform_shap_explanation(config, shap_data)

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_main.at[optimized_metric, config.optimized_part]

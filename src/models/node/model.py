from typing import Any, List
from torch import nn
from torchmetrics import MetricCollection, Accuracy, F1, Precision, Recall, CohenKappa, MatthewsCorrCoef, AUROC
from torchmetrics import CosineSimilarity, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, PearsonCorrCoef, R2Score, SpearmanCorrCoef
import wandb
from typing import Dict
import pytorch_lightning as pl
import torch
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_explain_matrix
from .architecture_blocks import DenseODSTBlock
from .utils import entmax15, entmoid15, Lambda


class NodeModel(pl.LightningModule):

    def __init__(
            self,
            task="classification",
            input_dim=1000,
            output_dim=2,

            num_trees=128,
            num_layers=2,
            flatten_output=False,
            depth=4,

            optimizer_lr=0.001,
            optimizer_weight_decay=0.0005,
            scheduler_step_size=20,
            scheduler_gamma=0.9,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self._build_network()

        self.produce_probabilities = False

        if self.hparams.task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            if self.hparams.output_dim < 2:
                raise ValueError(f"Classification with {self.hparams.output_dim} classes")
            self.metrics_dict = {
                'accuracy_macro': Accuracy(num_classes=self.hparams.output_dim, average='macro'),
                'accuracy_micro': Accuracy(num_classes=self.hparams.output_dim, average='micro'),
                'accuracy_weighted': Accuracy(num_classes=self.hparams.output_dim, average='weighted'),
                'f1_macro': F1(num_classes=self.hparams.output_dim, average='macro'),
                'f1_micro': F1(num_classes=self.hparams.output_dim, average='micro'),
                'f1_weighted': F1(num_classes=self.hparams.output_dim, average='weighted'),
                'precision_macro': Precision(num_classes=self.hparams.output_dim, average='macro'),
                'precision_micro': Precision(num_classes=self.hparams.output_dim, average='micro'),
                'precision_weighted': Precision(num_classes=self.hparams.output_dim, average='weighted'),
                'recall_macro': Recall(num_classes=self.hparams.output_dim, average='macro'),
                'recall_micro': Recall(num_classes=self.hparams.output_dim, average='micro'),
                'recall_weighted': Recall(num_classes=self.hparams.output_dim, average='weighted'),
                'cohens_kappa': CohenKappa(num_classes=self.hparams.output_dim),
                'matthews_corr': MatthewsCorrCoef(num_classes=self.hparams.output_dim),
            }
            self.metrics_summary = {
                'accuracy_macro': 'max',
                'accuracy_micro': 'max',
                'accuracy_weighted': 'max',
                'f1_macro': 'max',
                'f1_micro': 'max',
                'f1_weighted': 'max',
                'precision_macro': 'max',
                'precision_micro': 'max',
                'precision_weighted': 'max',
                'recall_macro': 'max',
                'recall_micro': 'max',
                'recall_weighted': 'max',
                'cohens_kappa': 'max',
                'matthews_corr': 'max',
            }
            self.metrics_prob_dict = {
                'auroc_macro': AUROC(num_classes=self.hparams.output_dim, average='macro'),
                'auroc_micro': AUROC(num_classes=self.hparams.output_dim, average='micro'),
                'auroc_weighted': AUROC(num_classes=self.hparams.output_dim, average='weighted'),
            }
            self.metrics_prob_summary = {
                'auroc_macro': 'max',
                'auroc_micro': 'max',
                'auroc_weighted': 'max',
            }
        elif self.hparams.task == "regression":
            self.loss_fn = torch.nn.L1Loss(reduction='mean')
            self.metrics_dict = {
                'CosineSimilarity': CosineSimilarity(),
                'MeanAbsoluteError': MeanAbsoluteError(),
                'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
                'MeanSquaredError': MeanSquaredError(),
                'PearsonCorrcoef': PearsonCorrCoef(),
                'R2Score': R2Score(),
                'SpearmanCorrcoef': SpearmanCorrCoef(),
            }
            self.metrics_summary = {
                'CosineSimilarity': 'min',
                'MeanAbsoluteError': 'min',
                'MeanAbsolutePercentageError': 'min',
                'MeanSquaredError': 'min',
                'PearsonCorrcoef': 'max',
                'R2Score': 'max',
                'SpearmanCorrcoef': 'max'
            }
            self.metrics_prob_dict = {}
            self.metrics_prob_summary = {}

        self.metrics_train = MetricCollection(self.metrics_dict)
        self.metrics_train_prob = MetricCollection(self.metrics_prob_dict)
        self.metrics_val = self.metrics_train.clone()
        self.metrics_val_prob = self.metrics_train_prob.clone()
        self.metrics_test = self.metrics_train.clone()
        self.metrics_test_prob = self.metrics_train_prob.clone()

    def _build_network(self):
        self.node = nn.Sequential(
            DenseODSTBlock(
                input_dim=self.hparams.input_dim,
                num_trees=self.hparams.num_trees,
                num_layers=self.hparams.num_layers,
                tree_output_dim=self.hparams.output_dim + 1,
                flatten_output=self.hparams.flatten_output,
                depth=self.hparams.depth,
                choice_function=entmax15,
                bin_function=entmoid15
            ),
            Lambda(lambda x: x[..., :self.hparams.output_dim].mean(dim=-2)),
        )

    def forward(self, x):
        # Returns output and Masked Loss. We only need the output
        x = self.node(x)
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x

    def on_fit_start(self) -> None:
        for stage_type in ['train', 'val', 'test']:
            for m, sum in self.metrics_summary.items():
                wandb.define_metric(f"{stage_type}/{m}", summary=sum)
            for m, sum in self.metrics_prob_summary.items():
                wandb.define_metric(f"{stage_type}/{m}", summary=sum)
            wandb.define_metric(f"{stage_type}/loss", summary='min')

    def step(self, batch: Any, stage:str):
        x, y, ind = batch
        out = self.forward(x)
        batch_size = x.size(0)
        if self.task == "regression":
            y = y.view(batch_size, -1)
        loss = self.loss_fn(out, y)

        logs = {"loss": loss}
        non_logs = {}
        if self.task == "classification":
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            non_logs["preds"] = preds
            non_logs["targets"] = y
            if stage == "train":
                logs.update(self.metrics_train(preds, y))
                try:
                    logs.update(self.metrics_train_prob(probs, y))
                except ValueError:
                    pass
            elif stage == "val":
                logs.update(self.metrics_val(preds, y))
                try:
                    logs.update(self.metrics_val_prob(probs, y))
                except ValueError:
                    pass
            elif stage == "test":
                logs.update(self.metrics_test(preds, y))
                try:
                    logs.update(self.metrics_val_prob(probs, y))
                except ValueError:
                    pass
        elif self.task == "regression":
            if stage == "train":
                logs.update(self.metrics_train(out, y))
            elif stage == "val":
                logs.update(self.metrics_val(out, y))
            elif stage == "test":
                logs.update(self.metrics_test(out, y))

        return loss, logs, non_logs

    def training_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "train")
        d = {f"train/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "val")
        d = {f"val/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def predict_step(self, batch, batch_idx):
        x, y, ind = batch
        out = self.forward(x)
        return out

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "test")
        d = {f"test/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.optimizer_lr,
            weight_decay=self.hparams.optimizer_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        )

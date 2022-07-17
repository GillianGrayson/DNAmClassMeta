from typing import Any, List
from torchmetrics import MetricCollection
import wandb
import pytorch_lightning as pl
import torch
from src.experiment.metrics import get_cls_pred_metrics, get_cls_prob_metrics, get_reg_metrics


class BaseModel(pl.LightningModule):

    def __init__(
            self,
            task,
            loss_type,
            input_dim,
            output_dim,
            optimizer_lr,
            optimizer_weight_decay,
            scheduler_step_size,
            scheduler_gamma,
    ):
        super().__init__()

        self.task = task
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        self.ids_cat = None
        self.ids_con = None

        self.produce_probabilities = False
        self.produce_importance = False

        if self.task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            if self.output_dim < 2:
                raise ValueError(f"Classification with {self.output_dim} classes")
            self.metrics = get_cls_pred_metrics(self.output_dim)
            self.metrics = {f'{k}_pl': v for k, v in self.metrics.items()}
            self.metrics_dict = {k:v[0] for k,v in self.metrics.items()}
            self.metrics_prob = get_cls_prob_metrics(self.output_dim)
            self.metrics_prob = {f'{k}_pl': v for k, v in self.metrics_prob.items()}
            self.metrics_prob_dict =  {k:v[0] for k,v in self.metrics_prob.items()}
        elif self.task == "regression":
            if self.loss_type == "MSE":
                self.loss_fn = torch.nn.MSELoss(reduction='mean')
            elif self.loss_type == "L1Loss":
                self.loss_fn = torch.nn.L1Loss(reduction='mean')
            else:
                raise ValueError("Unsupported loss_type")

            self.metrics = get_reg_metrics()
            self.metrics = {f'{k}_pl': v for k, v in self.metrics.items()}
            self.metrics_dict = {k: v[0] for k, v in self.metrics.items()}
            self.metrics_prob_dict = {}

        self.metrics_train = MetricCollection(self.metrics_dict)
        self.metrics_train_prob = MetricCollection(self.metrics_prob_dict)
        self.metrics_val = self.metrics_train.clone()
        self.metrics_val_prob = self.metrics_train_prob.clone()
        self.metrics_test = self.metrics_train.clone()
        self.metrics_test_prob = self.metrics_train_prob.clone()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure all MaxMetric doesn't store accuracy from these checks
        # self.max_metric.reset()
        pass

    def on_fit_start(self) -> None:
        for stage_type in ['train', 'val', 'test']:
            for m in self.metrics:
                wandb.define_metric(f"{stage_type}/{m}", summary=self.metrics[m][1])
            for m in self.metrics_prob:
                wandb.define_metric(f"{stage_type}/{m}", summary=self.metrics_prob[m][1])
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
                    logs.update(self.metrics_test_prob(probs, y))
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

    def predict_step(self, batch, batch_idx):
        x, y, ind = batch
        out = self.forward(x)
        return out

    def on_epoch_end(self):
        for m in self.metrics_dict:
            self.metrics_train[m].reset()
            self.metrics_val[m].reset()
            self.metrics_test[m].reset()
        for m in self.metrics_prob_dict:
            self.metrics_train_prob[m].reset()
            self.metrics_val_prob[m].reset()
            self.metrics_test_prob[m].reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        )

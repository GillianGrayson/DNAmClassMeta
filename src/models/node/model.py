from typing import Any, List
from torch import nn
import torch
from .architecture_blocks import DenseODSTBlock
from .utils import entmax15, entmoid15, Lambda
from src.models.base import BaseModel


class NodeModel(BaseModel):

    def __init__(
            self,
            task="regression",
            loss_type="MSE",
            input_dim=100,
            output_dim=1,
            optimizer_lr=0.001,
            optimizer_weight_decay=0.0005,
            scheduler_step_size=20,
            scheduler_gamma=0.9,

            num_trees=128,
            num_layers=2,
            flatten_output=False,
            depth=4,

            **kwargs
    ):
        super().__init__(
            task=task,
            loss_type=loss_type,
            input_dim=input_dim,
            output_dim=output_dim,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
        )
        self.save_hyperparameters()
        self._build_network()

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
        x = self.node(x)
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x

    def on_train_start(self) -> None:
        super().on_train_start()

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def step(self, batch: Any, stage: str):
        return super().step(batch=batch, stage=stage)

    def training_step(self, batch: Any, batch_idx: int):
        return super().training_step(batch=batch, batch_idx=batch_idx)

    def training_epoch_end(self, outputs: List[Any]):
        return super().training_epoch_end(outputs=outputs)

    def validation_step(self, batch: Any, batch_idx: int):
        return super().validation_step(batch=batch, batch_idx=batch_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        return super().validation_epoch_end(outputs=outputs)

    def test_step(self, batch: Any, batch_idx: int):
        return super().test_step(batch=batch, batch_idx=batch_idx)

    def test_epoch_end(self, outputs: List[Any]):
        return super().test_epoch_end(outputs=outputs)

    def predict_step(self, batch, batch_idx):
        return super().predict_step(batch=batch, batch_idx=batch_idx)

    def on_epoch_end(self):
        return super().on_epoch_end()

    def configure_optimizers(self):
        return super().configure_optimizers()

from typing import Any, List
import torch
from pytorch_tabnet.tab_network import TabNet
from src.models.base import BaseModel


class TabNetModel(BaseModel):

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

            n_d_n_a=8,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            virtual_batch_size=128,
            mask_type="sparsemax",

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
        self.tabnet = TabNet(
            input_dim=self.hparams.input_dim,
            output_dim=self.hparams.output_dim,
            n_d=self.hparams.n_d_n_a,
            n_a=self.hparams.n_d_n_a,
            n_steps=self.hparams.n_steps,
            gamma=self.hparams.gamma,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=[],
            n_independent=self.hparams.n_independent,
            n_shared=self.hparams.n_shared,
            epsilon=1e-15,
            virtual_batch_size=self.hparams.virtual_batch_size,
            momentum=0.02,
            mask_type=self.hparams.mask_type,
        )

    def forward(self, x):
        # Returns output and Masked Loss. We only need the output
        if self.produce_importance:
            return self.tabnet.forward_masks(x)
        else:
            x, _ = self.tabnet(x)
            if self.produce_probabilities:
                return torch.softmax(x, dim=1)
            else:
                return x

    def on_train_start(self) -> None:
        super().on_train_start()

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def step(self, batch: Any, stage:str):
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

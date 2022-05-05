from typing import Optional, Tuple
from src import utils
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset, WeightedRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import pandas as pd
from impyute.imputation.cs import fast_knn, mean, median, random, mice, mode, em
from collections import Counter
from src.utils.plot import save_figure, add_bar_trace, add_layout
import plotly.express as px
import plotly.graph_objects as go


log = utils.get_logger(__name__)

class DNAmDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            output: pd.DataFrame,
            outcome: str
    ):
        self.data = data
        self.output = output
        self.outcome = outcome
        self.num_subjects = self.data.shape[0]
        self.num_features = self.data.shape[1]
        self.ys = self.output.loc[:, self.outcome].values

    def __getitem__(self, idx: int):
        x = self.data.iloc[idx, :].to_numpy()
        y = self.ys[idx]
        return (x, y, idx)

    def __len__(self):
        return self.num_subjects


class DNAmDataModule(LightningDataModule):

    def __init__(
            self,
            task: str = "",
            feat_fn: str = "",
            label_fn: str = "",
            data_fn: str = "",
            outcome: str = "",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 42,
            imputation: str = "fast_knn",
            k: int = 1,
            **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.dataloaders_evaluate = False

        self.feat_names = pd.read_excel(self.hparams.feat_fn).loc[:, 'feature'].values

        self.classes_df = pd.read_excel(self.hparams.label_fn)
        self.classes_dict = {}
        for cl_id, cl in enumerate(self.classes_df.loc[:, self.hparams.outcome].values):
            self.classes_dict[cl] = cl_id

        self.raw = pd.read_excel(f"{self.hparams.data_fn}", index_col="subject_id")
        self.raw = self.raw.loc[self.raw[self.hparams.outcome].isin(self.classes_dict)]
        self.raw[f'{self.hparams.outcome}_origin'] = self.raw[self.hparams.outcome]
        self.raw[self.hparams.outcome].replace(self.classes_dict, inplace=True)

        self.ids_trn = self.raw.iloc[self.raw["Partition"] == "Train"]
        self.ids_val = self.raw.iloc[self.raw["Partition"] == "Val"]
        self.ids_tst = self.raw.iloc[self.raw["Partition"] == "Test"]

        self.data = self.raw.loc[:, self.feat_names]
        is_nans = self.data.isnull().values.any()
        if is_nans:
            n_nans = self.data.isna().sum().sum()
            log.info(f"Perform imputation for {n_nans} missed values")
            self.data = self.data.astype('float')
            if self.hparams.imputation == "median":
                imputed_training = median(self.data.loc[:, :].values)
            elif self.hparams.imputation == "mean":
                imputed_training = mean(self.data.loc[:, :].values)
            elif self.hparams.imputation == "fast_knn":
                imputed_training = fast_knn(self.data.loc[:, :].values, k=self.hparams.k)
            elif self.hparams.imputation == "random":
                imputed_training = random(self.data.loc[:, :].values)
            elif self.hparams.imputation == "mice":
                imputed_training = mice(self.data.loc[:, :].values)
            elif self.hparams.imputation == "em":
                imputed_training = em(self.data.loc[:, :].values)
            elif self.hparams.imputation == "mode":
                imputed_training = mode(self.data.loc[:, :].values)
            else:
                raise ValueError(f"Unsupported imputation: {self.hparams.imputation}")
            self.data.loc[:, :] = imputed_training
        self.data = self.data.astype('float32')

        self.output = self.raw.loc[:, [self.hparams.outcome, f'{self.hparams.outcome}_origin']]

        if not list(self.data.index.values) == list(self.output.index.values):
            raise ValueError(f"Error! Indexes have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.data.shape[1])

        self.dataset = DNAmDataset(self.data, self.output, self.hparams.outcome)
        self.dataset_trn: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_tst: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def refresh_datasets(self):
        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)
        self.dataset_tst = Subset(self.dataset, self.ids_tst)

    def perform_split(self):
        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)
        self.dataset_tst = Subset(self.dataset, self.ids_tst)

        log.info(f"total_count: {len(self.dataset)}")
        log.info(f"trn_count: {len(self.dataset_trn)}")
        log.info(f"val_count: {len(self.dataset_val)}")
        log.info(f"tst_count: {len(self.dataset_tst)}")

    def plot_split(self, suffix=''):
        dict_to_plot = {
            "Train": self.ids_trn,
            "Val": self.ids_val,
            "Test": self.ids_tst,
        }

        for name, ids in dict_to_plot.items():
            classes_counts = pd.DataFrame(Counter(self.output[f'{self.hparams.outcome}_origin'].values[ids]), index=[0])
            classes_counts = classes_counts.reindex(self.classes_df.loc[:, self.hparams.outcome].values, axis=1)
            fig = go.Figure()
            for st, st_id in self.classes_dict.items():
                add_bar_trace(fig, x=[st], y=[classes_counts.at[0, st]], text=[classes_counts.at[0, st]], name=st)
            add_layout(fig, f"", f"Count", "")
            fig.update_layout({'colorway': px.colors.qualitative.Set1})
            fig.update_xaxes(showticklabels=False)
            save_figure(fig, f"bar_{name}{suffix}")

    def get_trn_val_y(self):
        return self.dataset.ys[self.ids_trn_val]

    def train_dataloader(self):
        if self.dataloaders_evaluate:
            return DataLoader(
                dataset=self.dataset_trn,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False
            )
        else:
            ys_trn = self.dataset.ys[self.ids_trn]
            class_counter = Counter(ys_trn)
            class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
            weights = torch.FloatTensor([class_weights[y] for y in ys_trn])
            weighted_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            return DataLoader(
                dataset=self.dataset_trn,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                sampler=weighted_sampler
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_tst,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def get_feature_names(self):
        return self.data.columns.to_list()

    def get_outcome_name(self):
        return self.hparams.outcome

    def get_class_names(self):
        return list(self.classes_dict.keys())

    def get_df(self):
        df = pd.merge(self.output.loc[:, self.hparams.outcome], self.data, left_index=True, right_index=True)
        return df
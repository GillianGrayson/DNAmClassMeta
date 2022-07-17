from abc import abstractmethod, ABC
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np


class CVSplitter(ABC):

    def __init__(
            self,
            datamodule,
            is_split: bool = True,
            n_splits: int = 5,
            n_repeats: int = 1,
            random_state: int = 322
    ):
        self._datamodule = datamodule
        self._is_split = is_split
        self._n_splits = n_splits
        self._n_repeats = n_repeats
        self._random_state = random_state


    @abstractmethod
    def split(self):
        pass


class RepeatedStratifiedKFoldCVSplitter(CVSplitter):

    def __init__(self,
                 datamodule,
                 is_split: bool = True,
                 n_splits: int = 5,
                 n_repeats: int = 1,
                 random_state: int = 322,
                 ):
        super().__init__(datamodule, is_split, n_splits, n_repeats, random_state)
        self._k_fold = RepeatedStratifiedKFold(n_splits=self._n_splits, n_repeats=self._n_repeats, random_state=self._random_state)

    def split(self):

        self._split_feature = self._datamodule.split_feature

        if self._is_split:

            if self._split_feature is None:
                self._datamodule.setup()
                train_val_y = self._datamodule.get_trn_val_y()

                if self._datamodule.task in ['binary', 'multiclass', 'classification']:
                    splits = self._k_fold.split(X=range(len(train_val_y)), y=train_val_y, groups=train_val_y)
                elif self._datamodule.task == "regression":
                    ptp = np.ptp(train_val_y)
                    num_bins = 3
                    bins = np.linspace(np.min(train_val_y) - 0.1 * ptp, np.max(train_val_y) + 0.1 * ptp, num_bins + 1)
                    binned = np.digitize(train_val_y, bins) - 1
                    unique, counts = np.unique(binned, return_counts=True)
                    occ = dict(zip(unique, counts))
                    splits = self._k_fold.split(X=range(len(train_val_y)), y=binned, groups=binned)
                else:
                    raise ValueError(f'Unsupported self.datamodule.task: {self._datamodule.task}')

                for ids_trn, ids_val in splits:
                    yield ids_trn, ids_val

            else:
                self._datamodule.setup()
                train_val_split_feature = self._datamodule.get_trn_val_split_feature()
                train_val_split_feature = train_val_split_feature.to_frame(name='split_feature')
                train_val_split_feature['ids'] = np.arange(train_val_split_feature.shape[0])
                spl_feat_vals = train_val_split_feature['split_feature'].unique()

                if self._datamodule.task in ['binary', 'multiclass', 'classification']:
                    splits = self._k_fold.split(X=spl_feat_vals, y=np.ones(len(spl_feat_vals)))
                elif self._datamodule.task == "regression":
                    raise ValueError(f'Unsupported split by feature for the regression')
                else:
                    raise ValueError(f'Unsupported self.datamodule.task: {self._datamodule.task}')

                for ids_trn_feat, ids_val_feat in splits:
                    trn_values = spl_feat_vals[ids_trn_feat]
                    ids_trn = train_val_split_feature.loc[train_val_split_feature['split_feature'].isin(trn_values), 'ids'].values
                    val_values = spl_feat_vals[ids_val_feat]
                    ids_val = train_val_split_feature.loc[train_val_split_feature['split_feature'].isin(val_values), 'ids'].values
                    yield ids_trn, ids_val

        else:
            yield self._datamodule.ids_trn, self._datamodule.ids_val

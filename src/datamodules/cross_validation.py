from abc import abstractmethod, ABC
from sklearn.model_selection import RepeatedStratifiedKFold


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
    """
        Stratified K-fold cross-validation data module

    Args:
        data_module: data module containing data to be split
        n_splits: number of k-fold iterations/data splits
    """

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

        if self._is_split:
            # 0. Get data to split
            self._datamodule.setup()
            train_val_y = self._datamodule.get_trn_val_y()
            splits = self._k_fold.split(X=range(len(train_val_y)), y=train_val_y, groups=train_val_y)
            # 1. Iterate through splits
            for ids_trn, ids_val in splits:
                yield ids_trn, ids_val

        else:
            yield self._datamodule.ids_trn, self._datamodule.ids_val

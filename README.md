<div align="center">

# DNAmClassMeta

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

## Description

Repository with source code for paper "Disease classification for whole blood DNA methylation: meta-analysis, missing values imputation, and XAI" by A. Kalyakulina, I. Yusipov, M.G. Bacalini, C. Franceschi, M. Vedunova, M. Ivanchenko.

## Requirements

Install dependencies

```bash
# clone project
git clone https://github.com/GillianGrayson/DNAmClassMeta
cd DNAmClassMeta

# [OPTIONAL] create conda environment
conda create -n env_name python=3.8
conda activate env_name

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Data description

The `data` directory contains harmonized and non-harmonized data (methylation M-values for top-1000 best CpGs) for Parkinson disease and Schizophrenia for all considered GSE datasets:

```
└── data                         <- Project data
    ├── Parkinson                   <- Data for Parkinson disease
    │   ├── non_harmonized             <- Non-harmonized data
    │   │   ├── data.xlsx                 <- Dataframe with methylation data and additional subjects info
    │   │   ├── features.xlsx             <- Dataframe with features (models input)
    │   │   └── labels.xlsx               <- Dataframe with class labels (models output)
    │   └── harmonized                 <- Harmonized data
    │       ├── data.xlsx                 <- Dataframe with methylation data and additional subjects info
    │       ├── features.xlsx             <- Dataframe with features (models input)
    │       └── labels.xlsx               <- Dataframe with class labels (models output)
    └── Schizophrenia               <- Data for Schizophrenia
        ├── non_harmonized             <- Non-harmonized data
        │   ├── data.xlsx                 <- Dataframe with methylation data and additional subjects info
        │   ├── features.xlsx             <- Dataframe with features (models input)
        │   └── labels.xlsx               <- Dataframe with class labels (models output)
        └── harmonized                 <- Harmonized data
            ├── data.xlsx                 <- Dataframe with methylation data and additional subjects info
            ├── features.xlsx             <- Dataframe with features (models input)
            └── labels.xlsx               <- Dataframe with class labels (models output)
```

> `data.xlsx` is a dataframe, each row corresponds to subject (GSM), each column corresponds to feature. 
> In addition to methylation levels (M-values) there are another features: `Status` (control or case), `Dataset` (original GSE) and `Partition` (Train, Validation or Test).

> `features.xlsx` is a dataframe which contains features (CpGs), which will be used as input features of models. 
> Modifying this file will change the set of features (CpGs),which will be used for building a model.

> `labels.xlsx` is a dataframe which class labels.
> Modifying this file allows to select the subset of subjects which will participate in model.

## Configuring experiments

There are two types of experiments based on type of model:
- Stand-Alone (SA) models ([Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [XGBoost](https://github.com/dmlc/xgboost), [CatBoost](https://github.com/catboost/catboost), [LightGBM](https://github.com/microsoft/LightGBM))
- [PyTorch Lightning](https://www.pytorchlightning.ai) (PL) based models ([TabNet](https://github.com/dreamquark-ai/tabnet), [NODE](https://github.com/Qwicen/node))

Configuration files for the experiments can be found in the following directory:
```
└── configs
    └── experiment
        └── classification           
            └── trn_val_tst                 
                ├── sa.yaml         <- Configuration file for Stand-Alone (SA) models
                └── pl.yaml         <- Configuration file for PyTorch Lightning (PL) based models
```

There are common parts in these configuration files:

```yaml
disease: Parkinson        # Disease type. Options: [Parkinson, Schizophrenia]
data_type: harmonized     # Data type. Options: [non_harmonized, harmonized]
model_type: catboost      # Model type. Options: for SA: [logistic_regression, svm, xgboost, catboost, lightgbm], for PL: [tabnet, node]
outcome: "Status"         # Which column in `data.xlsx` contains class labels

optimized_metric: "accuracy_weighted"   # Target metric in optimization process. Options: [accuracy_weighted, f1_weighted, auroc_weighted]
optimized_mean: ""                      # Optimizing mean metric value across all cross-validation splits. Options: ["", cv_mean]
optimized_part: "val"                   # Which partition should be optimized? Options: [trn, val, tst]
direction: "max"                        # Optimization metrics should be minimized or maximized? Options: [max, min]

max_epochs: 2000  # Maximum number of epochs in training process
patience: 100     # How many validation epochs of not improving until training stops

is_shap: True                    # SHAP values calculation. Options: [True, False]
is_shap_save: False               # Save SHAP values to file?
shap_explainer: Tree              # Explainer type for SHAP values. Options: for SA: [Tree, Kernel], for PL: [Deep, Kernel]
shap_bkgrd: tree_path_dependent   # Background for calculating SHAP values. Options: [trn, all, tree_path_dependent]. Last option works only for GBDT models

datamodule:
  _target_: src.datamodules.dnam.DNAmDataModule   # Instantiated datamodule
  feat_fn: "feat.xlsx"                            # Filename of target features file
  label_fn: "label.xlsx"                          # Filename of target labels file
  data_fn: "data.xlsx"                            # Filename of data file
  outcome: "Status"                               # Which column in `data_fn' contains class labels
  batch_size: 128                                 # How many samples per batch to load
  num_workers: 0                                  # How many subprocesses to use for data loading
  pin_memory: False                               # Data loader will copy Tensors into CUDA pinned memory before returning them
  seed: 42                                        # Random seed
  imputation: "fast_knn"                          # Imputation method for missing values. Options: [median, mean, fast_knn, random, mice, em, mode]
  k: 1                                            # k for 'fast_knn' imputation method
```

Parameters of models can be changed in corresponding blocks of configuration files:

```yaml
logistic_regression:    # Logistic Regression model params
  ...

svm:    # SVM params
  ...

xgboost:    # XGBoost model params
  ...

catboost:   # CatBoost model params
  ...

lightgbm:   # LightGBM model params
  ...

tabnet:     # TabNet model params
  ...

node:       # NODE model params
  ...
```

## Running experiments

### Single experiment

Train SA model with configuration from `sa.yaml` file:

```bash
python run_classification_trn_val_tst_sa.py experiment=classification/trn_val_tst/sa.yaml
```

Train PL model with configuration from `pl.yaml` file:

```bash
python run_classification_trn_val_tst_pl.py experiment=classification/trn_val_tst/pl.yaml
```


### Hyperparameter Search

To perform the hyperparametric search for the optimal combination of the parameters on the grid add `--multirun` key and specify the parameters values:

```bash
python run_classification_trn_val_tst_pl.py experiment=classification/trn_val_tst/pl.yaml --multirun model_tabnet.optimizer_lr=0.001,0.005,0.01
```

## Results
After running the experiment the new working directory with results in `logs` will be created:

```
└── logs                            # Folder for the logs generated by experiments
    ├── runs                          # Folder for single runs
    │   └── experiment_name             # Experiment name
    │       ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the run
    │       │   ├── csv                     # csv logs
    │       │   ├── wandb                   # Weights&Biases logs
    │       │   └── ...                     # Resulting tables, figures, etc.
    │       └── ...
    └── multiruns                     # Folder for multiruns
        └── experiment_name             # Experiment name
            ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the multirun
            │   ├──1                        # Multirun job number
            │   ├──2
            │   └── ...
            └── ...

```

## Acknowledgements
The research was supported by the Ministry of Science and Higher Education of the Russian Federation, agreement No. 075-15-2020-808. 
The authors acknowledge the use of computational resources provided by the “Lobachevsky” supercomputer.

## License

This project is licensed under the MIT License.

import torchmetrics
import torch


def init_metric(self):
    if hasattr(self, 'average') and self.average != '':
        self._name = f"{self.metric_type}_{self.average}"
    else:
        self._name = f"{self.metric_type}"
    self._maximize = True


def calc_metric(self, y_true, y_score):
    y_true_torch = torch.from_numpy(y_true)
    y_score_torch = torch.from_numpy(y_score)
    if self.metric_type == "accuracy":
        value = torchmetrics.functional.accuracy(y_score_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
    elif self.metric_type == "f1":
        value = torchmetrics.functional.f1(y_score_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
    elif self.metric_type == "precision":
        value = torchmetrics.functional.precision(y_score_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
    elif self.metric_type == "recall":
        value = torchmetrics.functional.recall(y_score_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
    elif self.metric_type == "cohen_kappa":
        value = torchmetrics.functional.cohen_kappa(y_score_torch, y_true_torch, num_classes=self.num_classes)
    elif self.metric_type == "matthews_corrcoef":
        value = torchmetrics.functional.matthews_corrcoef(y_score_torch, y_true_torch, num_classes=self.num_classes)
    elif self.metric_type == "mean_absolute_error":
        value = torchmetrics.functional.mean_absolute_error(y_score_torch, y_true_torch)
    elif self.metric_type == "mean_absolute_percentage_error":
        value = torchmetrics.functional.mean_absolute_percentage_error(y_score_torch, y_true_torch)
    elif self.metric_type == "mean_squared_error":
        value = torchmetrics.functional.mean_squared_error(y_score_torch, y_true_torch)
    elif self.metric_type == "pearson_corrcoef":
        value = torchmetrics.functional.pearson_corrcoef(y_score_torch, y_true_torch)
    elif self.metric_type == "r2_score":
        value = torchmetrics.functional.r2_score(y_score_torch, y_true_torch)
    elif self.metric_type == "spearman_corrcoef":
        value = torchmetrics.functional.spearman_corrcoef(y_score_torch, y_true_torch)
    else:
        raise ValueError("Unsupported metrics")
    value = float(value.numpy())
    return value


def calc_metric_prob(self, y_true, y_prob):
    y_true_torch = torch.from_numpy(y_true)
    y_prob_torch = torch.from_numpy(y_prob)
    value = 0
    if self.metric_type == "auroc":
        try:
            value = torchmetrics.functional.auroc(y_prob_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
            value = float(value.numpy())
        except ValueError:
            pass
    else:
        raise ValueError("Unsupported metrics")
    return value


def get_classification_metrics_dict(num_classes, base_class):
    d = {
        "accuracy_macro": type(
            "accuracy_macro",
            (base_class,),
            {
                "metric_type": "accuracy",
                "average": "macro",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "accuracy_micro": type(
            "accuracy_micro",
            (base_class,),
            {
                "metric_type": "accuracy",
                "average": "micro",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "accuracy_weighted": type(
            "accuracy_weighted",
            (base_class,),
            {
                "metric_type": "accuracy",
                "average": "weighted",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "f1_macro": type(
            "f1_macro",
            (base_class,),
            {
                "metric_type": "f1",
                "average": "macro",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "f1_micro": type(
            "f1_micro",
            (base_class,),
            {
                "metric_type": "f1",
                "average": "micro",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "f1_weighted": type(
            "f1_weighted",
            (base_class,),
            {
                "metric_type": "f1",
                "average": "weighted",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "cohen_kappa": type(
            "cohen_kappa",
            (base_class,),
            {
                "metric_type": "cohen_kappa",
                "average": "",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "matthews_corrcoef": type(
            "matthews_corrcoef",
            (base_class,),
            {
                "metric_type": "matthews_corrcoef",
                "average": "",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "auroc_macro": type(
            "auroc_macro",
            (base_class,),
            {
                "metric_type": "auroc",
                "average": "macro",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric_prob
            }
        ),
        "auroc_micro": type(
            "auroc_micro",
            (base_class,),
            {
                "metric_type": "auroc",
                "average": "micro",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric_prob
            }
        ),
        "auroc_weighted": type(
            "auroc_weighted",
            (base_class,),
            {
                "metric_type": "auroc",
                "average": "weighted",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric_prob
            }
        ),
    }

    return d


def get_regression_metrics_dict(base_class):
    d = {
        "mean_absolute_error": type(
            "mean_absolute_error",
            (base_class,),
            {
                "metric_type": "mean_absolute_error",
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "mean_absolute_percentage_error": type(
            "mean_absolute_percentage_error",
            (base_class,),
            {
                "metric_type": "mean_absolute_percentage_error",
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "mean_squared_error": type(
            "mean_squared_error",
            (base_class,),
            {
                "metric_type": "mean_squared_error",
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "pearson_corrcoef": type(
            "pearson_corrcoef",
            (base_class,),
            {
                "metric_type": "pearson_corrcoef",
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "r2_score": type(
            "r2_score",
            (base_class,),
            {
                "metric_type": "r2_score",
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "spearman_corrcoef": type(
            "spearman_corrcoef",
            (base_class,),
            {
                "metric_type": "spearman_corrcoef",
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
    }

    return d


import torchmetrics


def get_cls_pred_metrics(num_classes):
    metrics = {
        'accuracy_macro': (torchmetrics.Accuracy(num_classes=num_classes, average='macro'), 'max'),
        'accuracy_micro': (torchmetrics.Accuracy(num_classes=num_classes, average='micro'), 'max'),
        'accuracy_weighted': (torchmetrics.Accuracy(num_classes=num_classes, average='weighted'), 'max'),
        'f1_score_macro': (torchmetrics.F1Score(num_classes=num_classes, average='macro'), 'max'),
        'f1_score_micro': (torchmetrics.F1Score(num_classes=num_classes, average='micro'), 'max'),
        'f1_score_weighted': (torchmetrics.F1Score(num_classes=num_classes, average='weighted'), 'max'),
        'precision_macro': (torchmetrics.Precision(num_classes=num_classes, average='macro'), 'max'),
        'precision_micro': (torchmetrics.Precision(num_classes=num_classes, average='micro'), 'max'),
        'precision_weighted': (torchmetrics.Precision(num_classes=num_classes, average='weighted'), 'max'),
        'recall_macro': (torchmetrics.Recall(num_classes=num_classes, average='macro'), 'max'),
        'recall_micro': (torchmetrics.Recall(num_classes=num_classes, average='micro'), 'max'),
        'recall_weighted': (torchmetrics.Recall(num_classes=num_classes, average='weighted'), 'max'),
        'cohen_kappa': (torchmetrics.CohenKappa(num_classes=num_classes), 'max'),
        'matthews_corr_coef': (torchmetrics.MatthewsCorrCoef(num_classes=num_classes), 'max'),
    }
    return metrics


def get_cls_prob_metrics(num_classes):
    metrics = {
        'auroc_macro': (torchmetrics.AUROC(num_classes=num_classes, average='macro'), 'max'),
        'auroc_micro': (torchmetrics.AUROC(num_classes=num_classes, average='micro'), 'max'),
        'auroc_weighted': (torchmetrics.AUROC(num_classes=num_classes, average='weighted'), 'max'),
    }
    return metrics


def get_reg_metrics():
    metrics = {
        'cosine_similarity': (torchmetrics.CosineSimilarity(), 'min'),
        'mean_absolute_error': (torchmetrics.MeanAbsoluteError(), 'min'),
        'mean_absolute_percentage_error': (torchmetrics.MeanAbsolutePercentageError(), 'min'),
        'mean_squared_error': (torchmetrics.MeanSquaredError(), 'min'),
        'pearson_corr_coef': (torchmetrics.PearsonCorrCoef(), 'max'),
        'r2_score': (torchmetrics.R2Score(), 'max'),
        'spearman_corr_coef': (torchmetrics.SpearmanCorrCoef(), 'max'),
    }
    return metrics

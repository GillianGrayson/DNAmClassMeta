import pandas as pd
from src.experiment.metrics import get_classification_metrics_dict, get_regression_metrics_dict
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import wandb
import plotly.graph_objects as go
from src.utils.plot import save_figure, add_layout
import plotly.io as pio
pio.kaleido.scope.mathjax = None


def save_feature_importance(df, num_features):
    df.sort_values(['importance'], ascending=[False], inplace=True)
    df['importance'] = df['importance'] / df['importance'].sum()
    fig = go.Figure()
    ys = df['feature'][0:num_features][::-1]
    xs = df['importance'][0:num_features][::-1]
    fig.add_trace(
        go.Bar(
            x=xs,
            y=list(range(len(ys))),
            orientation='h',
            marker=dict(color='red', opacity=0.9)
        )
    )
    add_layout(fig, "Feature importance", "", "")
    fig.update_layout(legend_font_size=20)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(xs))),
            ticktext=ys
        )
    )
    fig.update_yaxes(autorange=False)
    fig.update_layout(yaxis_range=[-0.5, len(xs) - 0.5])
    fig.update_yaxes(tickfont_size=24)
    fig.update_xaxes(tickfont_size=24)
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        margin=go.layout.Margin(
            l=350,
            r=20,
            b=100,
            t=40,
            pad=0
        )
    )
    save_figure(fig, f"feature_importances")
    df.set_index('feature', inplace=True)
    df.to_excel("feature_importances.xlsx", index=True)


def eval_classification_sa(config, class_names, y_real, y_pred, y_pred_prob, loggers, part, is_log=True, is_save=True, suffix=''):
    metrics_classes_dict = get_classification_metrics_dict(config.out_dim, object)
    metrics_summary = {
        'accuracy_macro': 'max',
        'accuracy_micro': 'max',
        'accuracy_weighted': 'max',
        'f1_macro': 'max',
        'f1_micro': 'max',
        'f1_weighted': 'max',
        'cohen_kappa': 'max',
        'matthews_corrcoef': 'max',
    }
    metrics_summary['auroc_weighted'] = 'max'
    metrics_summary['auroc_macro'] = 'max'

    metrics = [metrics_classes_dict[m]() for m in metrics_summary]

    if is_log:
        if 'wandb' in config.logger:
            for m, sum in metrics_summary.items():
                wandb.define_metric(f"{part}/{m}", summary=sum)

    metrics_dict = {'metric': [m._name for m in metrics]}
    metrics_dict[part] = []
    log_dict = {}
    for m in metrics:
        if m._name in ['auroc_weighted', 'auroc_macro']:
            m_val = m(y_real, y_pred_prob)
        else:
            m_val = m(y_real, y_pred)
        metrics_dict[part].append(m_val)
        log_dict[f"{part}/{m._name}"] = m_val
    for logger in loggers:
        if is_log:
            logger.log_metrics(log_dict)

    if is_save:
        plot_confusion_matrix(y_real, y_pred, class_names, part, suffix=suffix)

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.set_index('metric', inplace=True)
    if is_save:
        metrics_df.to_excel(f"metrics_{part}{suffix}.xlsx", index=True)

    return metrics_df


def eval_regression_sa(config, y_real, y_pred, loggers, part, is_log=True, is_save=True, suffix=''):
    metrics_classes_dict = get_regression_metrics_dict(object)
    metrics_summary = {
        'mean_absolute_error': 'min',
        'mean_absolute_percentage_error': 'min',
        'mean_squared_error': 'min',
        'pearson_corrcoef': 'max',
        'r2_score': 'max',
        'spearman_corrcoef': 'max',
    }

    metrics = [metrics_classes_dict[m]() for m in metrics_summary]

    if is_log:
        if 'wandb' in config.logger:
            for m, sum in metrics_summary.items():
                wandb.define_metric(f"{part}/{m}", summary=sum)

    metrics_dict = {'metric': [m._name for m in metrics]}
    metrics_dict[part] = []
    log_dict = {}
    for m in metrics:
        m_val = m(y_real, y_pred)
        metrics_dict[part].append(m_val)
        log_dict[f"{part}/{m._name}"] = m_val
    for logger in loggers:
        if is_log:
            logger.log_metrics(log_dict)

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.set_index('metric', inplace=True)
    if is_save:
        metrics_df.to_excel(f"metrics_{part}{suffix}.xlsx", index=True)

    return metrics_df


def plot_confusion_matrix(y_real, y_pred, class_names, part, suffix=''):
    conf_mtx = confusion_matrix(y_real, y_pred)
    if len(conf_mtx) > 1:
        fig = ff.create_annotated_heatmap(conf_mtx, x=class_names, y=class_names, showscale=False)
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 60
        fig.update_layout(
            template="none",
            autosize=True,
            margin=go.layout.Margin(
                l=120,
                r=20,
                b=20,
                t=100,
                pad=0
            ),
            showlegend=False,
            xaxis=dict(
                title="Prediction",
                autorange=True,
                showgrid=True,
                zeroline=False,
                linecolor='black',
                showline=True,
                gridcolor='gainsboro',
                gridwidth=0.05,
                mirror=True,
                ticks='outside',
                titlefont=dict(
                    color='black',
                    size=45
                ),
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    color='black',
                    size=35
                ),
                exponentformat='e',
                showexponent='all'
            ),
            yaxis=dict(
                title="Real",
                autorange=True,
                showgrid=True,
                zeroline=False,
                linecolor='black',
                showline=True,
                gridcolor='gainsboro',
                gridwidth=0.05,
                mirror=True,
                ticks='outside',
                titlefont=dict(
                    color='black',
                    size=45
                ),
                showticklabels=True,
                tickangle=270,
                tickfont=dict(
                    color='black',
                    size=35
                ),
                exponentformat='e',
                showexponent='all'
            ),
        )
        save_figure(fig, f"confusion_matrix_{part}{suffix}")


def eval_loss(loss_info, loggers):
    for epoch_id, epoch in enumerate(loss_info['epoch']):
        log_dict = {
            'epoch': loss_info['epoch'][epoch_id],
            'train/loss': loss_info['train/loss'][epoch_id],
            'val/loss': loss_info['val/loss'][epoch_id]
        }
        for logger in loggers:
            logger.log_metrics(log_dict)

    loss_df = pd.DataFrame(loss_info)
    loss_df.set_index('epoch', inplace=True)
    loss_df.to_excel(f"loss.xlsx", index=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=loss_info['epoch'],
            y=loss_info['train/loss'],
            showlegend=True,
            name="Train",
            mode="lines",
            marker=dict(
                size=8,
                opacity=0.7,
                line=dict(
                    width=1
                )
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=loss_info['epoch'],
            y=loss_info['val/loss'],
            showlegend=True,
            name="Val",
            mode="lines",
            marker=dict(
                size=8,
                opacity=0.7,
                line=dict(
                    width=1
                )
            )
        )
    )
    add_layout(fig, "Epoch", 'Error', "")
    fig.update_layout({'colorway': ['blue', 'red']})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=90,
            r=20,
            b=75,
            t=45,
            pad=0
        )
    )
    fig.update_yaxes(autorange=False)
    fig.update_layout(yaxis_range=[0, max(loss_info['train/loss'] + loss_info['val/loss']) + 0.1])
    save_figure(fig, f"loss")

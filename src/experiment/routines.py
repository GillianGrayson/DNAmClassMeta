import pandas as pd
from src.experiment.metrics import get_cls_pred_metrics, get_cls_prob_metrics, get_reg_metrics
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import wandb
import plotly.graph_objects as go
from src.utils.plot import save_figure, add_layout
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import torch


def save_feature_importance(df, num_features):
    if df is not None:
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


def eval_classification(config, class_names, y_real, y_pred, y_pred_prob, loggers, part, is_log=True, is_save=True, file_suffix=''):
    metrics_pred = get_cls_pred_metrics(config.out_dim)
    metrics_prob = get_cls_prob_metrics(config.out_dim)

    if is_log:
        if 'wandb' in config.logger:
            for m in metrics_pred:
                wandb.define_metric(f"{part}/{m}", summary=metrics_pred[m][1])
            for m in metrics_prob:
                wandb.define_metric(f"{part}/{m}", summary=metrics_prob[m][1])

    metrics_df = pd.DataFrame(index=[m for m in metrics_pred] + [m for m in metrics_prob], columns=[part])
    metrics_df.index.name = 'metric'
    log_dict = {}
    for m in metrics_pred:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_torch = torch.from_numpy(y_pred)
        m_val = float(metrics_pred[m][0](y_pred_torch, y_real_torch).numpy())
        metrics_pred[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val
    for m in metrics_prob:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_prob_torch = torch.from_numpy(y_pred_prob)
        m_val = 0
        try:
            m_val = float(metrics_prob[m][0](y_pred_prob_torch, y_real_torch).numpy())
        except ValueError:
            pass
        metrics_prob[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val

    if loggers is not None:
        for logger in loggers:
            if is_log:
                logger.log_metrics(log_dict)

    if is_save:
        plot_confusion_matrix(y_real, y_pred, class_names, part, suffix=file_suffix)
        metrics_df.to_excel(f"metrics_{part}{file_suffix}.xlsx", index=True)

    return metrics_df


def eval_regression(config, y_real, y_pred, loggers, part, is_log=True, is_save=True, file_suffix=''):
    metrics = get_reg_metrics()

    if is_log:
        if 'wandb' in config.logger:
            for m in metrics:
                wandb.define_metric(f"{part}/{m}", summary=metrics[m][1])

    metrics_df = pd.DataFrame(index=[m for m in metrics], columns=[part])
    metrics_df.index.name = 'metric'
    log_dict = {}
    for m in metrics:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_torch = torch.from_numpy(y_pred)
        m_val = float(metrics[m][0](y_pred_torch, y_real_torch).numpy())
        metrics[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val

    if loggers is not None:
        for logger in loggers:
            if is_log:
                logger.log_metrics(log_dict)

    if is_save:
        metrics_df.to_excel(f"metrics_{part}{file_suffix}.xlsx", index=True)

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


def eval_loss(loss_info, loggers, is_log=True, is_save=True, file_suffix=''):
    for epoch_id, epoch in enumerate(loss_info['epoch']):
        log_dict = {
            'epoch': loss_info['epoch'][epoch_id],
            'train/loss': loss_info['train/loss'][epoch_id],
            'val/loss': loss_info['val/loss'][epoch_id]
        }
        if loggers is not None:
            for logger in loggers:
                if is_log:
                    logger.log_metrics(log_dict)

    if is_save:
        loss_df = pd.DataFrame(loss_info)
        loss_df.set_index('epoch', inplace=True)
        loss_df.to_excel(f"loss{file_suffix}.xlsx", index=True)

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
        save_figure(fig, f"loss{file_suffix}")

import pandas as pd
import shap
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from src import utils
import plotly.graph_objects as go
from src.utils.plot import save_figure, add_layout, add_scatter_trace
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.mathjax = None


log = utils.get_logger(__name__)


def explain_samples(config, y_real, y_pred, indexes, shap_values, base_values, features, feature_names, class_names, path):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)
    is_correct_pred = (np.array(y_real) == np.array(y_pred))
    mistakes_ids = np.where(is_correct_pred == False)[0]
    corrects_ids = np.where(is_correct_pred == True)[0]

    num_mistakes = min(len(mistakes_ids), config.num_examples)

    for m_id in mistakes_ids[0:num_mistakes]:
        log.info(f"Plotting sample with error {indexes[m_id]}")

        ind_save = indexes[m_id].replace('/', '_')
        for cl_id, cl in enumerate(class_names):
            path_curr = f"{path}/mistakes/real({class_names[y_real[m_id]]})_pred({class_names[y_pred[m_id]]})/{ind_save}"
            Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[cl_id][m_id],
                    base_values=base_values[cl_id],
                    data=features[m_id],
                    feature_names=feature_names,
                ),
                show=False,
                max_display=config.num_top_features,
            )
            fig = plt.gcf()
            fig.savefig(f"{path_curr}/waterfall_{cl}.pdf", bbox_inches='tight')
            fig.savefig(f"{path_curr}/waterfall_{cl}.png", bbox_inches='tight')
            plt.close()

            shap.plots.decision(
                base_value=base_values[cl_id],
                shap_values=shap_values[cl_id][m_id],
                features=features[m_id],
                feature_names=feature_names,
                show=False,
            )
            fig = plt.gcf()
            fig.savefig(f"{path_curr}/decision_{cl}.pdf", bbox_inches='tight')
            fig.savefig(f"{path_curr}/decision_{cl}.png", bbox_inches='tight')
            plt.close()

            shap.plots.force(
                base_value=base_values[cl_id],
                shap_values=shap_values[cl_id][m_id],
                features=features[m_id],
                feature_names=feature_names,
                show=False,
                matplotlib=True,
            )
            fig = plt.gcf()
            fig.savefig(f"{path_curr}/force_{cl}.pdf", bbox_inches='tight')
            fig.savefig(f"{path_curr}/force_{cl}.png", bbox_inches='tight')
            plt.close()

    correct_samples = {x: 0 for x in range(len(class_names))}
    for c_id in corrects_ids:
        if correct_samples[y_real[c_id]] < config.num_examples:
            log.info(f"Plotting correct sample {indexes[c_id]} for {y_real[c_id]}")
            ind_save = indexes[c_id].replace('/', '_')
            for cl_id, cl in enumerate(class_names):
                path_curr = f"{path}/corrects/{class_names[y_real[c_id]]}/{ind_save}"
                Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)

                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[cl_id][c_id],
                        base_values=base_values[cl_id],
                        data=features[c_id],
                        feature_names=feature_names
                    ),
                    show=False,
                    max_display=config.num_top_features
                )
                fig = plt.gcf()
                fig.savefig(f"{path_curr}/waterfall_{cl}.pdf", bbox_inches='tight')
                fig.savefig(f"{path_curr}/waterfall_{cl}.png", bbox_inches='tight')
                plt.close()

                shap.plots.decision(
                    base_value=base_values[cl_id],
                    shap_values=shap_values[cl_id][c_id],
                    features=features[c_id],
                    feature_names=feature_names,
                    show=False,
                )
                fig = plt.gcf()
                fig.savefig(f"{path_curr}/decision_{cl}.pdf", bbox_inches='tight')
                fig.savefig(f"{path_curr}/decision_{cl}.png", bbox_inches='tight')
                plt.close()

                shap.plots.force(
                    base_value=base_values[cl_id],
                    shap_values=shap_values[cl_id][c_id],
                    features=features[c_id],
                    feature_names=feature_names,
                    show=False,
                    matplotlib=True,
                )
                fig = plt.gcf()
                fig.savefig(f"{path_curr}/force_{cl}.pdf", bbox_inches='tight')
                fig.savefig(f"{path_curr}/force_{cl}.png", bbox_inches='tight')
                plt.close()

            correct_samples[y_real[c_id]] += 1


def explain_shap(config, expl_data):
    model = expl_data['model']
    predict_func = expl_data['predict_func']
    df = expl_data['df']
    feature_names = expl_data['feature_names']
    class_names = expl_data['class_names']
    outcome_name = expl_data['outcome_name']

    if config.shap_explainer == 'Tree' and config.shap_bkgrd == 'tree_path_dependent':
        explainer = shap.TreeExplainer(model)
    else:
        ids_bkgrd = expl_data[f"ids_{config.shap_bkgrd}"]
        indexes_bkgrd = df.index[ids_bkgrd]
        X_bkgrd = df.loc[indexes_bkgrd, feature_names].values
        if config.shap_explainer == 'Tree':
            explainer = shap.TreeExplainer(model, data=X_bkgrd, feature_perturbation='interventional')
        elif config.shap_explainer == "Kernel":
            explainer = shap.KernelExplainer(predict_func, X_bkgrd)
        elif config.shap_explainer == "Deep":
            explainer = shap.DeepExplainer(model, torch.from_numpy(X_bkgrd))
        else:
            raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

    for part in ['trn', 'val', 'tst', 'all']:
        if expl_data[f"ids_{part}"] is not None:
            log.info(f"Calculating SHAP for {part}")
            Path(f"shap/{part}/global").mkdir(parents=True, exist_ok=True)

            ids = expl_data[f"ids_{part}"]
            indexes = df.index[ids]
            X = df.loc[indexes, feature_names].values
            y_pred_prob = df.loc[indexes, [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values
            y_pred_raw = df.loc[indexes, [f"pred_raw_{cl_id}" for cl_id, cl in enumerate(class_names)]].values

            if config.shap_explainer == "Tree":
                shap_values = explainer.shap_values(X)

                base_prob = list(np.mean(y_pred_prob, axis=0))

                base_prob_expl = []
                base_prob_num = []
                base_prob_den = 0
                for class_id in range(0, len(explainer.expected_value)):
                    base_prob_num.append(np.exp(explainer.expected_value[class_id]))
                    base_prob_den += np.exp(explainer.expected_value[class_id])
                for class_id in range(0, len(explainer.expected_value)):
                    base_prob_expl.append(base_prob_num[class_id] / base_prob_den)
                log.info(f"Base probability check: {np.linalg.norm(np.array(base_prob) - np.array(base_prob_expl))}")

                # Сonvert raw SHAP values to probability SHAP values
                shap_values_prob = copy.deepcopy(shap_values)
                for class_id in range(0, len(class_names)):
                    for subject_id in range(0, y_pred_prob.shape[0]):

                        # Сhecking raw SHAP values
                        real_raw = y_pred_raw[subject_id, class_id]
                        expl_raw = explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id])
                        diff_raw = real_raw - expl_raw
                        if abs(diff_raw) > 1e-5:
                            log.warning(f"Difference between raw for subject {subject_id} in class {class_id}: {abs(diff_raw)}")

                        # Checking conversion to probability space
                        real_prob = y_pred_prob[subject_id, class_id]
                        expl_prob_num = np.exp(explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id]))
                        expl_prob_den = 0
                        for c_id in range(0, len(explainer.expected_value)):
                            expl_prob_den += np.exp(explainer.expected_value[c_id] + sum(shap_values[c_id][subject_id]))
                        expl_prob = expl_prob_num / expl_prob_den
                        delta_prob = expl_prob - base_prob[class_id]
                        diff_prob = real_prob - expl_prob
                        if abs(diff_prob) > 1e-5:
                            log.warning(f"Difference between prediction for subject {subject_id} in class {class_id}: {abs(diff_prob)}")

                        # Сonvert raw SHAP values to probability SHAP values
                        shap_contrib_logodd = np.sum(shap_values[class_id][subject_id])
                        shap_contrib_prob = delta_prob
                        coeff = shap_contrib_prob / shap_contrib_logodd
                        for feature_id in range(0, X.shape[1]):
                            shap_values_prob[class_id][subject_id, feature_id] = shap_values[class_id][subject_id, feature_id] * coeff
                        diff_check = shap_contrib_prob - sum(shap_values_prob[class_id][subject_id])
                        if abs(diff_check) > 1e-5:
                            log.warning(f"Difference between SHAP contribution for subject {subject_id} in class {class_id}: {diff_check}")

                shap_values = shap_values_prob
                expected_value = base_prob
            elif config.shap_explainer == "Kernel":
                shap_values = explainer.shap_values(X)
                expected_value = explainer.expected_value
                log.info(f"Base probability check: {np.linalg.norm(np.mean(y_pred_prob, axis=0) - np.array(expected_value))}")
            elif config.shap_explainer == "Deep":
                model.produce_probabilities = True
                shap_values = explainer.shap_values(torch.from_numpy(X))
                expected_value = explainer.expected_value
                log.info(f"Base probability check: {np.linalg.norm(np.mean(y_pred_prob, axis=0) - np.array(expected_value))}")
            else:
                raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

            if config.is_shap_save:
                for cl_id, cl in enumerate(expl_data['class_names']):
                    df_shap = pd.DataFrame(index=indexes,  columns=feature_names, data=shap_values[cl_id])
                    df_shap.index.name = 'index'
                    df_shap.to_excel(f"shap/{part}/shap_{cl}.xlsx", index=True)

            shap.summary_plot(
                shap_values=shap_values,
                features=X,
                feature_names=feature_names,
                max_display=config.num_top_features,
                class_names=class_names,
                class_inds=list(range(len(class_names))),
                show=False,
                color=plt.get_cmap("Set1")
            )
            plt.savefig(f'shap/{part}/global/bar.png', bbox_inches='tight')
            plt.savefig(f'shap/{part}/global/bar.pdf', bbox_inches='tight')
            plt.close()

            for cl_id, cl in enumerate(expl_data['class_names']):
                Path(f"shap/{part}/global/{cl}").mkdir(parents=True, exist_ok=True)
                shap.summary_plot(
                    shap_values=shap_values[cl_id],
                    features=X,
                    feature_names=feature_names,
                    max_display=config.num_top_features,
                    show=False,
                    plot_type="bar"
                )
                plt.savefig(f'shap/{part}/global/{cl}/bar.png', bbox_inches='tight')
                plt.savefig(f'shap/{part}/global/{cl}/bar.pdf', bbox_inches='tight')
                plt.close()

                shap.summary_plot(
                    shap_values=shap_values[cl_id],
                    features=X,
                    feature_names=feature_names,
                    max_display=config.num_top_features,
                    plot_type="violin",
                    show=False,
                )
                plt.savefig(f"shap/{part}/global/{cl}/beeswarm.png", bbox_inches='tight')
                plt.savefig(f"shap/{part}/global/{cl}/beeswarm.pdf", bbox_inches='tight')
                plt.close()

                explanation = shap.Explanation(
                    values=shap_values[cl_id],
                    base_values=np.array([expected_value] * len(ids)),
                    data=X,
                    feature_names=feature_names
                )
                shap.plots.heatmap(
                    explanation,
                    show=False,
                    max_display=config.num_top_features,
                    instance_order=explanation.sum(1)
                )
                plt.savefig(f"shap/{part}/global/{cl}/heatmap.png", bbox_inches='tight')
                plt.savefig(f"shap/{part}/global/{cl}/heatmap.pdf", bbox_inches='tight')
                plt.close()

                Path(f"shap/{part}/features/{cl}").mkdir(parents=True, exist_ok=True)
                shap_values_class = shap_values[cl_id]
                mean_abs_impact = np.mean(np.abs(shap_values_class), axis=0)
                features_order = np.argsort(mean_abs_impact)[::-1]
                inds_to_plot = features_order[0:config.num_top_features]
                for feat_id, ind in enumerate(inds_to_plot):
                    feat = feature_names[ind]
                    shap.dependence_plot(
                        ind=ind,
                        shap_values=shap_values_class,
                        features=X,
                        feature_names=feature_names,
                        show=False,
                    )
                    plt.savefig(f"shap/{part}/features/{cl}/{feat_id}_{feat}.png", bbox_inches='tight')
                    plt.savefig(f"shap/{part}/features/{cl}/{feat_id}_{feat}.pdf", bbox_inches='tight')
                    plt.close()

            mean_abs_shap_values = np.sum([np.mean(np.absolute(shap_values[cl_id]), axis=0) for cl_id, cl in enumerate(class_names)], axis=0)
            order = np.argsort(mean_abs_shap_values)[::-1]
            features = np.asarray(feature_names)[order]
            features_best = features[0:config.num_top_features]
            for feat_id, feat in enumerate(features_best):
                fig = go.Figure()
                for cl_id, cl in enumerate(class_names):
                    shap_values_cl = shap_values[cl_id][:, order[feat_id]]
                    real_values = df.loc[indexes, feat].values
                    add_scatter_trace(fig, real_values, shap_values_cl, cl)
                add_layout(fig, f"{feat}", f"SHAP values for<br>{feat}", f"")
                fig.update_layout(legend_font_size=20)
                fig.update_layout(legend={'itemsizing': 'constant'})
                fig.update_layout(
                    margin=go.layout.Margin(
                        l=150,
                        r=20,
                        b=80,
                        t=35,
                        pad=0
                    )
                )
                fig.update_layout({'colorway': px.colors.qualitative.Set1})
                save_figure(fig, f"shap/{part}/features/{feat_id}_{feat}_scatter")

                for cl_id, cl in enumerate(class_names):
                    fig = go.Figure()
                    shap_values_cl = shap_values[cl_id][:, order[feat_id]]
                    real_values = df.loc[indexes, feat].values
                    add_scatter_trace(fig, real_values, shap_values_cl, "")
                    add_layout(fig, f"{feat}", f"SHAP values for<br>{feat}", f"")
                    fig.update_layout(legend_font_size=20)
                    fig.update_layout(legend={'itemsizing': 'constant'})
                    fig.update_layout(
                        margin=go.layout.Margin(
                            l=150,
                            r=20,
                            b=80,
                            t=20,
                            pad=0
                        )
                    )
                    fig.update_layout({'colorway': ['red']})
                    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey')
                    save_figure(fig, f"shap/{part}/features/{cl}/{feat_id}_{feat}_scatter")

                fig = go.Figure()
                for cl_id, cl in enumerate(class_names):
                    vals = df.loc[(df.index.isin(indexes)) & (df[outcome_name] == cl_id), feat].values
                    fig.add_trace(
                        go.Violin(
                            y=vals,
                            name=f"{cl}",
                            box_visible=True,
                            meanline_visible=True,
                            showlegend=False,
                            line_color='black',
                            fillcolor=px.colors.qualitative.Set1[cl_id],
                            marker=dict(color=px.colors.qualitative.Set1[cl_id], line=dict(color='black', width=0.3), opacity=0.8),
                            points='all',
                            bandwidth=np.ptp(vals) / 25,
                            opacity=0.8
                        )
                    )
                add_layout(fig, "", f"{feat}", f"")
                fig.update_layout({'colorway': px.colors.qualitative.Set1})
                fig.update_layout(legend={'itemsizing': 'constant'})
                fig.update_layout(title_xref='paper')
                fig.update_layout(legend_font_size=20)
                fig.update_layout(
                    margin=go.layout.Margin(
                        l=130,
                        r=20,
                        b=50,
                        t=20,
                        pad=0
                    )
                )
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.25,
                        xanchor="center",
                        x=0.5
                    )
                )
                save_figure(fig, f"shap/{part}/features/{feat_id}_{feat}_violin")

            explain_samples(
                config,
                df.loc[indexes, outcome_name].values,
                df.loc[indexes, "pred"].values,
                indexes,
                shap_values,
                expected_value,
                X,
                feature_names,
                class_names,
                f"shap/{part}/samples"
            )

import pandas as pd
from mosaic.deconvolve import nnls_deconvolve, elastic_net_deconvolve, nu_svr_deconvolve, xgb_deconvolve, rf_deconvolve, print_proportions
from mosaic.evaluate import evaluate_deconvolution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

celltype_index: pd.Series = pd.read_csv("benchmark_data/celltype_mapping.csv").set_index("original_label")["mapped_label"]

target_index: pd.Index = pd.Index(celltype_index.unique())
celltype_mapping = celltype_index.to_dict()

nnls_results = np.zeros((5, 2))
en_results = np.zeros((5, 2))
svr_results = np.zeros((5, 2))
rf_results = np.zeros((5, 2))
xgb_results = np.zeros((5, 2))


def plot_proportion_heatmap(model_props: dict[str, pd.Series], true_props: pd.Series,
                             title: str, dest: str) -> None:
    """Heatmap of estimated cell-type proportions, one column per model plus
    a "True" reference column on the right. Rows (cell types) are sorted by
    true proportion, descending, so the reference column reads as a clean
    gradient and the models' agreement with it is easy to scan by eye.

    Styling is plain seaborn default (``sns.heatmap`` with the "rocket"
    colormap) -- no hand-built gridlines, text-color logic, or column rule.

    Cell types where every model and the true proportion are 0.0 are dropped
    before plotting -- an all-zero row carries no signal and just adds a
    blank line to the figure.
    """
    row_order = true_props.sort_values(ascending=False).index
    columns = list(model_props.keys()) + ["True"]

    data = pd.DataFrame({name: props.reindex(row_order) for name, props in model_props.items()})
    data["True"] = true_props.reindex(row_order)

    data = data.loc[(data != 0.0).any(axis=1)]
    if data.empty:
        raise ValueError("plot_proportion_heatmap: every row is all-zero across "
                          "models and the true proportion -- nothing to plot.")

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(1.1 * len(columns) + 2.2, 0.42 * len(data) + 1.6))

    sns.heatmap(data, cmap="rocket_r", annot=True, fmt=".2f", cbar_kws={"label": "Proportion"}, ax=ax)

    ax.set_ylabel("")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")  # avoid label collision
    fig.tight_layout()
    fig.savefig(dest, dpi=200)
    plt.close(fig)


# fixed categorical order (blue, orange, aqua, yellow, magenta) -- the first
# five slots of the documented palette, validated for adjacent-pair CVD
# separation, which is what matters for bars grouped side by side
MODEL_PALETTE = {
    "NNLS": "#2a78d6",
    "Elastic Net": "#eb6834",
    "Nu-SVR": "#1baf7a",
    "Random Forests": "#eda100",
    "XGBoost": "#e87ba4",
}


def plot_benchmark_pcc_comparison(results: dict[str, np.ndarray], dest: str = "benchmarks.png") -> None:
    """Grouped bar chart of each model's Pearson correlation (PCC) across all
    five benchmarks -- the summary figure for the whole run, replacing the
    per-cell-type detail in :func:`plot_proportion_heatmap`.

    Parameters
    ----------
    results : dict of str -> numpy.ndarray
        Model name -> array of shape ``(5, 2)`` with columns
        ``(benchmark_index, pcc)``, i.e. exactly the ``*_results`` arrays
        ``do_benchmark`` already fills in. ``benchmark_index`` is 0-based;
        the x-axis is labelled 1-5.
    dest : str, default 'benchmarks.png'
        Path the figure is saved to.
    """
    records = [
        {"Benchmark": int(bench_idx) + 1, "Model": name, "PCC": pcc}
        for name, arr in results.items()
        for bench_idx, pcc in arr
    ]
    df = pd.DataFrame.from_records(records)

    models = [m for m in MODEL_PALETTE if m in results]
    palette = [MODEL_PALETTE[m] for m in models]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    sns.barplot(data=df, x="Benchmark", y="PCC", hue="Model", hue_order=models,
                palette=palette, ax=ax)

    # direct value labels -- three of the five palette hues sit below 3:1
    # contrast on a light surface, so labels carry the value regardless of
    # how well the fill color itself reads
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)

    ymin = min(0.0, df["PCC"].min() - 0.05)
    ymax = max(1.0, df["PCC"].max() + 0.08)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Pearson Correlation (PCC)")
    ax.set_title("Model Performance Across Benchmarks", fontsize=12, pad=12)
    sns.despine(ax=ax, left=True)
    ax.legend(title="Model", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(dest, dpi=200)
    plt.close(fig)


def do_benchmark():
    for i in range(1, 6):
        print(f"Benchmark {i}")

        true_props = pd.read_csv(f"benchmark_data/processed_data/test{i}/true_proportions.csv", index_col=0)
        true_props = true_props.groupby(level=0).sum()
        true_props = pd.Series(true_props.iloc[:, 0], index=true_props.index)

        sig = pd.read_csv(f"benchmark_data/processed_data/test{i}/signature.tsv", sep='\t', index_col=0)
        eval_bulk = pd.read_csv(f"benchmark_data/processed_data/test{i}/eval_bulk.tsv", sep='\t', index_col=0).iloc[:, 0]

        training_pb = pd.read_csv(f"benchmark_data/processed_data/test{i}/training_bulk.csv", index_col=0)
        training_pb_props = pd.read_csv(f"benchmark_data/processed_data/test{i}/training_bulk_props.csv", index_col=0)

        nnls_props = nnls_deconvolve(sig, eval_bulk)
        nnls_props = nnls_props.rename(index=celltype_mapping)
        nnls_props = nnls_props.groupby(level=0).sum()
        nnls_props = nnls_props.reindex(target_index, fill_value=0)

        en_props = elastic_net_deconvolve(sig, eval_bulk)
        en_props = en_props.rename(index=celltype_mapping)
        en_props = en_props.groupby(level=0).sum()
        en_props = en_props.reindex(target_index, fill_value=0)

        svr_props = nu_svr_deconvolve(sig, eval_bulk)
        svr_props = svr_props.rename(index=celltype_mapping)
        svr_props = svr_props.groupby(level=0).sum()
        svr_props = svr_props.reindex(target_index, fill_value=0)

        forests_props = rf_deconvolve(training_pb, training_pb_props, eval_bulk)
        forests_props = forests_props.rename(index=celltype_mapping)
        forests_props = forests_props.groupby(level=0).sum()
        forests_props = forests_props.reindex(target_index, fill_value=0)

        xgb_props = xgb_deconvolve(training_pb, training_pb_props, eval_bulk)
        xgb_props = xgb_props.rename(index=celltype_mapping)
        xgb_props = xgb_props.groupby(level=0).sum()
        xgb_props = xgb_props.reindex(target_index, fill_value=0)

        true_props = true_props.reindex(target_index, fill_value=0)

        plot_proportion_heatmap(
            model_props={
                "NNLS": nnls_props,
                "Elastic Net": en_props,
                "Nu-SVR": svr_props,
                "Random Forests": forests_props,
                "XGBoost": xgb_props,
            },
            true_props=true_props,
            title=f"Benchmark {i} Predictions",
            dest=f"benchmark{i}_heatmap.png",
        )

        nnls_eval = evaluate_deconvolution(nnls_props, true_props)
        en_eval = evaluate_deconvolution(en_props, true_props)
        svr_eval = evaluate_deconvolution(svr_props, true_props)

        forests_eval = evaluate_deconvolution(forests_props, true_props)
        xgb_eval = evaluate_deconvolution(xgb_props, true_props)

        nnls_results[i-1] = (i-1, nnls_eval['correlation'])
        en_results[i-1] = (i-1, en_eval['correlation'])
        svr_results[i-1] = (i-1, svr_eval['correlation'])

        rf_results[i-1] = (i-1, forests_eval['correlation'])
        xgb_results[i-1] = (i-1, xgb_eval['correlation'])



if __name__ == "__main__":
    nnls_results = np.zeros((5, 2))
    en_results = np.zeros((5, 2))
    svr_results = np.zeros((5, 2))
    rf_results = np.zeros((5, 2))
    xgb_results = np.zeros((5, 2))

    do_benchmark()
    print(nnls_results, en_results, svr_results)
    print(rf_results, xgb_results)

    plot_benchmark_pcc_comparison(
        results={
            "NNLS": nnls_results,
            "Elastic Net": en_results,
            "Nu-SVR": svr_results,
            "Random Forests": rf_results,
            "XGBoost": xgb_results,
        },
        dest="benchmarks.png",
    )

import pandas as pd
from mosaic.deconvolve import nnls_deconvolve, elastic_net_deconvolve, nu_svr_deconvolve, xgb_deconvolve, rf_deconvolve, print_proportions
from mosaic.evaluate import evaluate_deconvolution
import matplotlib.pyplot as plt
import numpy as np

celltype_index: pd.Series = pd.read_csv("eval_data/celltype_mapping.csv").set_index("original_label")["mapped_label"]

target_index: pd.Index = pd.Index(celltype_index.unique())
celltype_mapping = celltype_index.to_dict()

nnls_results = np.zeros((5, 2))
en_results = np.zeros((5, 2))
svr_results = np.zeros((5, 2))
rf_results = np.zeros((5, 2))
xgb_results = np.zeros((5, 2))

def do_benchmark1():
    for i in range(1, 6):
        print(f"Benchmark {i}")

        true_props = pd.read_csv(f"benchmark_data/benchmark1/test{i}/true_proportions.csv", index_col=0)
        true_props = true_props.groupby(level=0).sum()
        true_props = pd.Series(true_props.iloc[:, 0], index=true_props.index)

        sig = pd.read_csv(f"benchmark_data/benchmark1/test{i}/signature.tsv", sep='\t', index_col=0)
        eval_bulk = pd.read_csv(f"benchmark_data/benchmark1/test{i}/eval_bulk.tsv", sep='\t', index_col=0).iloc[:, 0]

        training_pb = pd.read_csv(f"benchmark_data/benchmark1/test{i}/training_bulk.csv", index_col=0)
        training_pb_props = pd.read_csv(f"benchmark_data/benchmark1/test{i}/training_bulk_props.csv", index_col=0)

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


def do_benchmark2():

    sig = pd.read_csv("benchmark_data/benchmark2/signature.tsv", sep='\t', index_col=0)
    training_pb = pd.read_csv(f"benchmark_data/benchmark2/training_bulk.csv", index_col=0)
    training_pb_props = pd.read_csv(f"benchmark_data/benchmark2/training_bulk_props.csv", index_col=0)

    true_props = pd.read_csv(f"benchmark_data/benchmark2/true_proportions.csv", index_col=0)
    true_props = true_props.groupby(level=0).sum()
    true_props = pd.Series(true_props.iloc[:, 0], index=true_props.index)

    eval_bulk = pd.read_csv(f"benchmark_data/benchmark2/eval_bulk.tsv", sep='\t', index_col=0).iloc[:, 0]

    nnls_props = nnls_deconvolve(sig, eval_bulk)
    nnls_props = nnls_props.groupby(level=0).sum()
    nnls_props = nnls_props.reindex(true_props.index, fill_value=0)

    en_props = elastic_net_deconvolve(sig, eval_bulk)
    en_props = en_props.groupby(level=0).sum()
    en_props = en_props.reindex(true_props.index, fill_value=0)

    svr_props = nu_svr_deconvolve(sig, eval_bulk)
    svr_props = svr_props.groupby(level=0).sum()
    svr_props = svr_props.reindex(true_props.index, fill_value=0)

    forests_props = rf_deconvolve(training_pb, training_pb_props, eval_bulk)
    forests_props = forests_props.groupby(level=0).sum()
    forests_props = forests_props.reindex(true_props.index, fill_value=0)

    xgb_props = xgb_deconvolve(training_pb, training_pb_props, eval_bulk)
    xgb_props = xgb_props.groupby(level=0).sum()
    xgb_props = xgb_props.reindex(true_props.index, fill_value=0)

    nnls_eval = evaluate_deconvolution(nnls_props, true_props)
    en_eval = evaluate_deconvolution(en_props, true_props)
    svr_eval = evaluate_deconvolution(svr_props, true_props)
    forests_eval = evaluate_deconvolution(forests_props, true_props)
    xgb_eval = evaluate_deconvolution(xgb_props, true_props)

    nnls_results[0] = (0, nnls_eval['correlation'])
    en_results[0] = (0, en_eval['correlation'])
    svr_results[0] = (0, svr_eval['correlation'])
    rf_results[0] = (0, forests_eval['correlation'])
    xgb_results[0] = (0, xgb_eval['correlation'])

BENCHMARK1 = True
BENCHMARK2 = False

if __name__ == "__main__":
    if BENCHMARK1:
        nnls_results = np.zeros((5, 2))
        en_results = np.zeros((5, 2))
        svr_results = np.zeros((5, 2))
        rf_results = np.zeros((5, 2))
        xgb_results = np.zeros((5, 2))

        do_benchmark1()
        print(nnls_results, en_results, svr_results)
        print(rf_results, xgb_results)

        plt.scatter(nnls_results[:, 0], nnls_results[:, 1], c="blue", label="NNLS", alpha=0.5)
        plt.scatter(en_results[:, 0], en_results[:, 1], c="green", label="Elastic Net", alpha=0.5)
        plt.scatter(svr_results[:, 0], svr_results[:, 1], c="red", label="SVR", alpha=0.5)
        plt.scatter(rf_results[:, 0], rf_results[:, 1], c="purple", label="Forests", alpha=0.5)
        plt.scatter(xgb_results[:, 0], xgb_results[:, 1], c="orange", label="XGBoost", alpha=0.5)

        plt.xlabel("Benchmark")
        plt.ylabel("PCC")
        plt.savefig("benchmarks.png")
        plt.show()

    if BENCHMARK2:

        nnls_results = np.zeros((1, 2))
        en_results = np.zeros((1, 2))
        svr_results = np.zeros((1, 2))
        rf_results = np.zeros((1, 2))
        xgb_results = np.zeros((1, 2))

        do_benchmark2()
        print(nnls_results, en_results, svr_results)
        print(rf_results, xgb_results)

        plt.scatter(nnls_results[:, 0], nnls_results[:, 1], c="blue", label="NNLS", alpha=0.5)
        plt.scatter(en_results[:, 0], en_results[:, 1], c="green", label="Elastic Net", alpha=0.5)
        plt.scatter(svr_results[:, 0], svr_results[:, 1], c="red", label="SVR", alpha=0.5)
        plt.scatter(rf_results[:, 0], rf_results[:, 1], c="purple", label="Forests", alpha=0.5)
        plt.scatter(xgb_results[:, 0], xgb_results[:, 1], c="orange", label="XGBoost", alpha=0.5)

        plt.xlabel("Benchmark")
        plt.ylabel("PCC")
        plt.savefig("benchmarks.png")
        plt.show()

import pandas as pd
from mosaic.deconvolve import nnls_deconvolve, elastic_net_deconvolve, nu_svr_deconvolve, print_proportions
from mosaic.evaluate import evaluate_deconvolution
import matplotlib.pyplot as plt
import numpy as np

celltype_mapping: pd.Series = pd.read_csv("benchmark_data/benchmark_labels.csv").set_index("original_label")["mapped_label"]
target_index = pd.Index(celltype_mapping.unique())

nnls_results = np.zeros((5, 2))
en_results = np.zeros((5, 2))
svr_results = np.zeros((5, 2))

def do_benchmark1():
    for i in range(1, 6):
        print(f"Benchmark {i}")

        true_props = pd.read_csv(f"benchmark_data/benchmark1/test{i}/true_proportions.csv", index_col=0)
        true_props = true_props.groupby(level=0).sum()
        true_props = pd.Series(true_props.iloc[:, 0], index=true_props.index)

        sig = pd.read_csv(f"benchmark_data/benchmark1/test{i}/signature.tsv", sep='\t', index_col=0)
        bulk = pd.read_csv(f"benchmark_data/benchmark1/test{i}/bulk.tsv", sep='\t', index_col=0).iloc[:, 0]

        nnls_props = nnls_deconvolve(sig, bulk)
        nnls_props = nnls_props.rename(index=celltype_mapping)
        nnls_props = nnls_props.groupby(level=0).sum()
        nnls_props = nnls_props.reindex(target_index, fill_value=0)

        en_props = elastic_net_deconvolve(sig, bulk)
        en_props = en_props.rename(index=celltype_mapping)
        en_props = en_props.groupby(level=0).sum()
        en_props = en_props.reindex(target_index, fill_value=0)

        svr_props = nu_svr_deconvolve(sig, bulk)
        svr_props = svr_props.rename(index=celltype_mapping)
        svr_props = svr_props.groupby(level=0).sum()
        svr_props = svr_props.reindex(target_index, fill_value=0)

        true_props = true_props.reindex(target_index, fill_value=0)
        nnls_eval = evaluate_deconvolution(nnls_props, true_props)
        en_eval = evaluate_deconvolution(en_props, true_props)
        svr_eval = evaluate_deconvolution(svr_props, true_props)

        nnls_results[i-1] = (i-1, nnls_eval['correlation'])
        en_results[i-1] = (i-1, en_eval['correlation'])
        svr_results[i-1] = (i-1, svr_eval['correlation'])


do_benchmark1()

print(nnls_results, en_results, svr_results)

plt.scatter(nnls_results[:, 0], nnls_results[:, 1], c="blue", label="NNLS")
plt.scatter(en_results[:, 0], en_results[:, 1], c="green", label="Elastic Net")
plt.scatter(svr_results[:, 0], svr_results[:, 1], c="red", label="SVR")
plt.xlabel("Benchmark")
plt.ylabel("PCC")
plt.ylim(0, 1)
plt.show()

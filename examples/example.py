import pandas as pd
from scipy.special import p_roots
from mosaic.deconvolve import nnls_deconvolve, print_proportions
from mosaic.evaluate import evaluate_deconvolution


true_props = pd.read_csv("benchmark_data/benchmark1/test1/true_proportions.csv", index_col=0)
true_props = true_props.groupby(level=0).sum()
true_props = pd.Series(true_props.iloc[:, 0], index=true_props.index)

sig = pd.read_csv("benchmark_data/benchmark1/test1/signature.tsv", sep='\t', index_col=0)
bulk = pd.read_csv("benchmark_data/benchmark1/test1/bulk.tsv", sep='\t', index_col=0).iloc[:, 0]

celltype_mapping: pd.Series = pd.read_csv("benchmark_data/benchmark_labels.csv").set_index("original_label")["mapped_label"]
target_index = pd.read_csv("benchmark_data/benchmark_labels.csv")["mapped_label"].unique()
target_index = pd.Index(target_index)

est_props = nnls_deconvolve(sig, bulk)
est_props = est_props.rename(index=celltype_mapping)
est_props = est_props.groupby(level=0).sum()

est_props = est_props.reindex(target_index, fill_value=0)
true_props = true_props.reindex(target_index, fill_value=0)
results = evaluate_deconvolution(est_props, true_props)

print_proportions(est_props, top_n=10)
print_proportions(true_props, top_n=10)

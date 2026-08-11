from mosaic.evaluate import get_true_proportions, get_adata_proportions
import scanpy as sc
import pandas as pd
from mosaic.preprocessing import generate_signature_matrix, generate_eval_pseudobulk, generate_training_pseudobulks
import os
import pyranges as pr
import scipy.sparse as sp
import anndata as ad
import numpy as np

celltype_index: pd.Series = pd.read_csv("benchmark_data/celltype_mapping.csv").set_index("original_label")["mapped_label"]
celltype_mapping = celltype_index.to_dict()


def filter_adata_celltypes(adata, mapping, cell_type_col="Cell_type (HSC)"):
    mask = adata.obs[cell_type_col].isin(mapping.keys())
    adata = adata[mask].copy()
    adata.obs[cell_type_col] = adata.obs[cell_type_col].map(mapping)
    return adata


def to_pyranges(var_names):
    coords = var_names.str.extract(r'^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$')
    valid = coords['start'].notna()
    if (~valid).any():
        print(f"  Dropping {(~valid).sum()} var_names that didn't match chrom:start-end "
              f"(e.g. var_names_make_unique suffixes)")
    coords = coords[valid]
    kept_names = var_names[valid.to_numpy()]
    return pr.PyRanges(
        chromosomes=coords['chrom'],
        starts=coords['start'].astype(int),
        ends=coords['end'].astype(int),
    ), kept_names


def filter_peaks_by_pyranges(adata, merged_pr):
    gr, kept_names = to_pyranges(adata.var_names)

    merged_df = merged_pr.df.copy()
    merged_df['MergedName'] = (merged_df['Chromosome'].astype(str) + ":"
                                + merged_df['Start'].astype(str) + "-" + merged_df['End'].astype(str))
    merged_named = pr.PyRanges(merged_df)

    gr_df = gr.df.copy()
    gr_df['OrigName'] = kept_names.to_numpy()
    mapping = pr.PyRanges(gr_df).join(merged_named).df.set_index('OrigName')['MergedName']

    merged_names = pd.Index(sorted(mapping.unique()))
    peak_pos = adata.var_names.get_indexer(mapping.index)
    region_pos = merged_names.get_indexer(mapping.to_numpy())

    M = sp.csr_matrix((np.ones(len(mapping)), (region_pos, peak_pos)),
                       shape=(len(merged_names), adata.n_vars))
    X_new = (M @ adata.X.T).T.tocsr()

    return type(adata)(X=X_new, obs=adata.obs.copy(), var=pd.DataFrame(index=merged_names))


def create_training_data(training_samples, peaks, cell_type_index, cell_type_col="cluster_label", out_dir="."):
    training_pb: list[pd.DataFrame] = []
    training_pb_props: list[pd.DataFrame] = []

    for sample in training_samples:
        sample = sample.to_memory() if sample.isbacked else sample
        pb, pb_props = generate_training_pseudobulks(sample, peaks, cell_type_index=cell_type_index, cell_type_col=cell_type_col, n_pseudobulks=20_000)
        training_pb.append(pb)
        training_pb_props.append(pb_props)

    X = pd.concat(training_pb)
    y = pd.concat(training_pb_props)

    X.to_csv(out_dir + "training_bulk.csv")
    y.to_csv(out_dir + "training_bulk_props.csv")


def create_benchmark(sig_exists=False, bulk_exists=False, training_data_exists=False, true_props_exist=False):

    s1 = sc.read_h5ad("benchmark_data/raw_adata/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad", backed="r+")
    s2 = sc.read_h5ad("benchmark_data/raw_adata/Granja2019-peripheral_blood_mononuclear_cells-D11T1.h5ad", backed="r+")
    s3 = sc.read_h5ad("benchmark_data/raw_adata/Granja2019-peripheral_blood_mononuclear_cells-D12T1.h5ad", backed="r+")
    s4 = sc.read_h5ad("benchmark_data/raw_adata/Granja2019-peripheral_blood_mononuclear_cells-D12T2.h5ad", backed="r+")
    s5 = sc.read_h5ad("benchmark_data/raw_adata/Granja2019-peripheral_blood_mononuclear_cells-D12T3.h5ad", backed="r+")

    for i in range(1, 6):
        sig_samples = [s1, s2, s3, s4, s5]
        bulk_sample = sig_samples.pop(i-1)

        os.makedirs(f"benchmark_data/processed_data/test{i}", exist_ok=True)

        if sig_exists:
            sig = pd.read_csv(f"benchmark_data/processed_data/test{i}/signature.tsv", sep='\t', index_col=0)
        else:
            sig: pd.DataFrame = generate_signature_matrix(
                adata_list=sig_samples,
                cell_type_col="cluster_label"
            )

            sig.to_csv(f"benchmark_data/processed_data/test{i}/signature.tsv", sep='\t')


        if bulk_exists:
            bulk = pd.read_csv(f"benchmark_data/processed_data/test{i}/eval_bulk.tsv", sep='\t', index_col=0).iloc[:, 0]
        else:
            bulk = generate_eval_pseudobulk(
                adata_list=[bulk_sample],
                peaks=sig.index,
                dest=f"benchmark_data/processed_data/test{i}/eval_bulk.tsv",
                sample_col="Donor (HSC)",
            )

        if not training_data_exists:
            celltype_index = sig.columns
            create_training_data(sig_samples, bulk.index, celltype_index, out_dir=f"benchmark_data/processed_data/test{i}/")

        if not true_props_exist:
            true_props = get_true_proportions(
                fragments_dir=f"eval_data/granja/pbmc/sample{i}/",
                cell_type_col="BioClassification",
                unknown_label="Unk",
            )

            true_props = true_props.rename(index=celltype_mapping)
            true_props = true_props.rename_axis("cell_type")
            true_props = true_props.groupby(level=0).sum()
            true_props.to_csv(f"benchmark_data/benchmark1/test{i}/true_proportions.csv")


if __name__ == "__main__":
    create_benchmark(sig_exists=False, bulk_exists=False, training_data_exists=True, true_props_exist=True)

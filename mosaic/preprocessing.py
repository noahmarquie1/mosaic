import numpy as np
import pandas as pd
from scipy.sparse import issparse
import scanpy as sc


def generate_signature_matrix(adata_list, dest, cell_type_col='cluster_label',
                              min_cells=10, top_n_variable=2_000):

    common_peaks = adata_list[0].var_names
    for adata in adata_list[1:]:
        common_peaks = common_peaks.intersection(adata.var_names)
    print(f"Found {len(common_peaks)} common peaks.")

    ct_sums = {}
    ct_counts = {}

    for i, adata in enumerate(adata_list):
        print(f"Processing dataset {i+1}/{len(adata_list)}...")
        for ct in adata.obs[cell_type_col].dropna().unique():
            if ct in ("Unk", "UNK", "Unknown"):
                continue
            mask = adata.obs[cell_type_col] == ct
            subset = adata[mask, common_peaks]
            X_sum = np.asarray(subset.X.sum(axis=0)).flatten()
            ct_sums[ct] = ct_sums.get(ct, np.zeros(len(common_peaks))) + X_sum
            ct_counts[ct] = ct_counts.get(ct, 0) + subset.n_obs

    signature = pd.DataFrame(index=common_peaks)
    for ct, counts_sum in ct_sums.items():
        if ct_counts[ct] < min_cells:
            print(f"Skipping '{ct}': {ct_counts[ct]} cells < {min_cells} minimum.")
            continue
        total = counts_sum.sum()
        cpm = (counts_sum / total) * 1e6
        signature[ct] = np.log1p(cpm)

    # Feature selection: keep top variable peaks across cell types
    peak_vars = signature.var(axis=1)
    signature = signature.loc[peak_vars.nlargest(top_n_variable).index]
    print(f"Retained {len(signature)} variable peaks.")

    signature.to_csv(dest, sep='\t')
    return signature


def generate_eval_pseudobulk(adata_list, signature, dest=None, sample_col='sample_id',
                             dataset_prefix=False):

    print("Loading signature reference...")
    if isinstance(signature, str):
        sig_df = pd.read_csv(signature, sep='\t', index_col=0)
    else:
        sig_df = signature.copy()

    sig_peaks = sig_df.index
    bulk_sums = {}
    seen_samples = {}

    for i, adata in enumerate(adata_list):
        print(f"Aggregating bulk for dataset {i+1}/{len(adata_list)}...")

        common_peaks = adata.var_names.intersection(sig_peaks)
        missing = len(sig_peaks) - len(common_peaks)
        if missing > 0:
            print(f"  Warning: {missing} signature peaks absent from dataset "
                  f"{i+1} — these will be treated as 0 after normalization.")

        if sample_col in adata.obs.columns:
            unique_samples = adata.obs[sample_col].dropna().unique()
            for sample in unique_samples:
                key = f"dataset{i+1}_{sample}" if dataset_prefix else sample

                if key in seen_samples and seen_samples[key] != i:
                    print(f"  Warning: sample '{sample}' already seen in dataset "
                          f"{seen_samples[key]+1}. Summing — verify this is intended.")
                seen_samples[key] = i

                mask = adata.obs[sample_col] == sample
                subset = adata[mask, common_peaks]
                raw_sum = np.asarray(subset.X.sum(axis=0)).flatten()

                # Align to full signature peak set
                series = pd.Series(0.0, index=sig_peaks)
                series.loc[common_peaks] = raw_sum

                bulk_sums[key] = bulk_sums.get(key, pd.Series(0.0, index=sig_peaks)) + series
        else:
            key = f"sample_{i+1}"
            subset = adata[:, common_peaks]
            raw_sum = np.asarray(subset.X.sum(axis=0)).flatten()
            series = pd.Series(0.0, index=sig_peaks)
            series.loc[common_peaks] = raw_sum
            bulk_sums[key] = bulk_sums.get(key, pd.Series(0.0, index=sig_peaks)) + series

    print("Normalizing bulk samples...")
    bulk = pd.DataFrame(bulk_sums)  # peaks x samples

    col_totals = bulk.sum(axis=0)
    zero_samples = col_totals[col_totals == 0].index.tolist()
    if zero_samples:
        print(f"  Warning: samples with zero total counts (dropping): {zero_samples}")
        bulk = bulk.drop(columns=zero_samples)
        col_totals = col_totals.drop(zero_samples)

    bulk_norm = np.log1p((bulk.div(col_totals, axis=1)) * 1e6)
    bulk_norm = bulk_norm.loc[sig_peaks]

    print(f"Bulk matrix shape: {bulk_norm.shape} (peaks x samples)")
    if dest:
        bulk_norm.to_csv(dest, sep='\t')

    return bulk_norm


def generate_training_pseudobulks(adata, signature, cell_type_col="cluster_label", n_pseudobulks=1000):
    rng = np.random.default_rng()

    groups = np.empty(adata.n_obs, dtype=int)
    for g, idx in enumerate(np.array_split(rng.permutation(adata.n_obs), n_pseudobulks)):
        groups[idx] = g

    adata.obs['pb_group'] = groups
    bulk = generate_eval_pseudobulk([adata], signature, sample_col='pb_group')

    sig_df = pd.read_csv(signature, sep='\t', index_col=0)
    counts = pd.crosstab(adata.obs['pb_group'], adata.obs[cell_type_col])
    props = counts.div(counts.sum(axis=1), axis=0)
    #props = props.reindex(index=X.index, columns=sig_df.columns, fill_value=0.0)

    return bulk.T, props



if __name__ == "__main__":

    sig_path = "benchmark_data/benchmark1/test1/signature.tsv"
    s2_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad")
    #s3_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad")
    #s4_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad")
    #s5_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad")

    X, y = generate_training_pseudobulks(s2_adata, sig_path, n_pseudobulks=2000)

    X.to_csv("benchmark_data/benchmark1/test1/training_bulk.csv")
    y.to_csv("benchmark_data/benchmark1/test1/training_bulk_props.csv")

    print(y)
    #pb_3 = generate_training_pseudobulks(s3_adata, sig_path, 25)
    #pb_4 = generate_training_pseudobulks(s4_adata, sig_path, 25)
    #pb_5 = generate_training_pseudobulks(s5_adata, sig_path, 25)

    #training_pb = pd.concat([pb_2, pb_3, pb_4, pb_5])

    #print(training_pb)
    #training_pb.to_csv("benchmark_data/benchmark")
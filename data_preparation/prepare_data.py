import pandas as pd
import numpy as np


def construct_signature(adata_list, dest, cell_type_col='cluster_label',
                        min_cells=50, top_n_variable=5000):

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
                print("Skipping unknown column")
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


def construct_pseudobulk(adata_list, signature, dest, sample_col='sample_id',
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
    bulk_norm.to_csv(dest, sep='\t')
    return bulk_norm

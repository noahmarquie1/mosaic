import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


def filter_zero_variance_peaks(adata: ad.AnnData, batch_col: str) -> ad.AnnData:
    """Drop peaks with zero variance in ANY batch -- these break ComBat's
    per-batch delta estimation."""
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    keep = np.ones(X.shape[1], dtype=bool)
    for batch in adata.obs[batch_col].unique():
        batch_mask = (adata.obs[batch_col] == batch).values
        batch_var = X[batch_mask].var(axis=0)
        keep &= (batch_var > 0)
    print(f"Dropping {(~keep).sum()} peaks with zero variance in at least one batch "
          f"({keep.sum()} retained of {len(keep)}).")
    return adata[:, keep].copy()


def build_metacells(adata: ad.AnnData, cell_type_col: str, cells_per_metacell: int = 30,
                    min_cells_for_clustering: int = 60, max_metacells_per_type: int = 50,
                    random_state: int = 0) -> ad.AnnData:

    X = adata.X.tocsr() if sp.issparse(adata.X) else adata.X

    metacell_rows, metacell_meta = [], []

    for ct in adata.obs[cell_type_col].unique():
        mask = (adata.obs[cell_type_col] == ct).values
        n_cells = mask.sum()
        if n_cells == 0:
            continue

        X_ct = X[mask]
        n_metacells = max(1, min(max_metacells_per_type, n_cells // cells_per_metacell))

        if n_cells < min_cells_for_clustering or n_metacells <= 1:
            summed = np.asarray(X_ct.sum(axis=0)).ravel()
            metacell_rows.append(summed)
            metacell_meta.append({cell_type_col: ct, "n_cells": int(n_cells), "metacell_id": f"{ct}_0"})
            continue

        # low-dim embedding for clustering only -- does not touch the raw
        # counts that actually get summed below
        depth = np.asarray(X_ct.sum(axis=1)).ravel()
        depth[depth == 0] = 1
        X_norm = X_ct.multiply(1.0 / depth[:, None]).tocsr() if sp.issparse(X_ct) else X_ct / depth[:, None]
        if sp.issparse(X_norm):
            X_norm = X_norm.copy()
            X_norm.data = np.log1p(X_norm.data * 1e4)
        else:
            X_norm = np.log1p(X_norm * 1e4)

        n_components = min(30, n_cells - 1, X_norm.shape[1] - 1)
        emb = TruncatedSVD(n_components=n_components, random_state=random_state).fit_transform(X_norm)

        labels = KMeans(n_clusters=n_metacells, random_state=random_state, n_init=10).fit_predict(emb)

        for c in range(n_metacells):
            c_mask = labels == c
            if c_mask.sum() == 0:
                continue
            summed = np.asarray(X_ct[c_mask].sum(axis=0)).ravel()
            metacell_rows.append(summed)
            metacell_meta.append({cell_type_col: ct, "n_cells": int(c_mask.sum()), "metacell_id": f"{ct}_{c}"})

    obs_meta = pd.DataFrame(metacell_meta).set_index("metacell_id", drop=False)
    return ad.AnnData(X=np.vstack(metacell_rows), obs=obs_meta, var=adata.var.copy())


def generate_signature_matrix(adata_list, cell_type_col='cluster_label', min_cells=10,
                              top_n_variable=2_000, dest=None):

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

        signature[ct] = counts_sum / ct_counts[ct]


    peak_vars = signature.var(axis=1)
    signature = signature.loc[peak_vars.nlargest(top_n_variable).index]
    print(f"Retained {len(signature)} variable peaks.")

    if dest:
        signature.to_csv(dest, sep='\t')
    return signature


def generate_eval_pseudobulk(adata_list, peaks, dest=None, sample_col='sample_id',
                             dataset_prefix=False):

    bulk_sums = {}
    bulk_counts = {}

    for i, adata in enumerate(adata_list):
        print(f"Aggregating bulk for dataset {i+1}/{len(adata_list)}...")

        common_peaks = adata.var_names.intersection(peaks)
        missing = len(peaks) - len(common_peaks)
        if missing > 0:
            print(f"  Warning: {missing} signature peaks absent from dataset "
                  f"{i+1} — these will be treated as 0 after normalization.")

        if sample_col in adata.obs.columns:
            codes, groups = pd.factorize(adata.obs[sample_col])
            valid = codes >= 0
            G = sp.csr_matrix((np.ones(valid.sum()), (codes[valid], np.nonzero(valid)[0])),
                shape=(len(groups), adata.n_obs))
            sums = np.asarray((G @ adata[:, common_peaks].X).todense())
            counts = np.asarray(G.sum(axis=1)).flatten()

            for j, sample in enumerate(groups):
                key = f"dataset{i+1}_{sample}" if dataset_prefix else sample
                series = pd.Series(0.0, index=peaks)
                series.loc[common_peaks] = sums[j]
                bulk_sums[key] = bulk_sums.get(key, pd.Series(0.0, index=peaks)) + series
                bulk_counts[key] = bulk_counts.get(key, 0) + counts[j]

        else:
            key = f"sample_{i+1}"
            subset = adata[:, common_peaks]
            raw_sum = np.asarray(subset.X.sum(axis=0)).flatten()
            series = pd.Series(0.0, index=peaks)
            series.loc[common_peaks] = raw_sum
            bulk_sums[key] = bulk_sums.get(key, pd.Series(0.0, index=peaks)) + series
            bulk_counts[key] = bulk_counts.get(key, 0) + adata.n_obs

    bulk = pd.DataFrame(bulk_sums)  # peaks x samples
    col_totals = bulk.sum(axis=0)
    zero_samples = col_totals[col_totals == 0].index.tolist()
    if zero_samples:
        print(f"  Warning: samples with zero total counts (dropping): {zero_samples}")
        bulk = bulk.drop(columns=zero_samples)
        col_totals = col_totals.drop(zero_samples)
        for s in zero_samples:
            bulk_counts.pop(s, None)

    counts_series = pd.Series(bulk_counts)[bulk.columns]
    result = bulk.div(counts_series, axis=1)

    print(f"Bulk matrix shape: {bulk.shape} (peaks x samples)")
    if dest:
        result.to_csv(dest, sep='\t')

    return result


def generate_training_pseudobulks(adata, peaks, cell_type_index, cell_type_col="cluster_label",
                                   n_pseudobulks=1000, cells_per_pseudobulk=300,
                                   alpha=1.0, sparse_alpha=0.1, sparse_frac=0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng()
    common_peaks = adata.var_names.intersection(peaks)
    X = sp.csr_matrix(adata[:, common_peaks].X)

    cell_type_index = pd.Index(cell_type_index)
    obs_types = adata.obs[cell_type_col].values
    type_to_indices = {ct: np.where(obs_types == ct)[0] for ct in cell_type_index
                       if (obs_types == ct).any()}
    available_types = list(type_to_indices.keys())
    k = len(available_types)

    n_sparse = int(round(n_pseudobulks * sparse_frac))
    n_dense = n_pseudobulks - n_sparse
    regime_alphas = np.concatenate([np.full(n_dense, alpha), np.full(n_sparse, sparse_alpha)])
    rng.shuffle(regime_alphas)

    rows, cols = [], []
    sampled_labels = []
    for pb in range(n_pseudobulks):
        target = rng.dirichlet(np.full(k, regime_alphas[pb]))
        counts = np.floor(target * cells_per_pseudobulk).astype(int)
        remainder = cells_per_pseudobulk - counts.sum()
        if remainder > 0:
            counts[np.argsort(-(target * cells_per_pseudobulk - counts))[:remainder]] += 1

        chosen = np.concatenate([
            rng.choice(type_to_indices[ct], size=counts[ti], replace=True)
            for ti, ct in enumerate(available_types) if counts[ti] > 0
        ])
        rows.append(np.full(len(chosen), pb))
        cols.append(chosen)
        sampled_labels.append(adata.obs[cell_type_col].iloc[chosen].values)

    G = sp.coo_matrix(
        (np.ones(len(np.concatenate(rows))), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_pseudobulks, adata.n_obs)
    ).tocsr()
    sums = np.asarray((G @ X).todense())

    bulk = pd.DataFrame(0.0, index=range(n_pseudobulks), columns=peaks)
    bulk.loc[:, common_peaks] = sums
    col_totals = bulk.sum(axis=1)
    bulk_norm = np.log1p(bulk.div(col_totals, axis=0) * 1e6)

    props = pd.DataFrame([
        pd.Series(labels).value_counts(normalize=True).reindex(cell_type_index, fill_value=0.0)
        for labels in sampled_labels
    ])
    props.index = bulk_norm.index

    print(f"Generated training pseudobulks, shape: {bulk_norm.shape}")

    return bulk_norm, props


if __name__ == "__main__":

    cell_type_index = pd.read_csv("benchmark_data/benchmark1/test1/signature.tsv", sep='\t', index_col=0).columns

    s1_peaks = pd.read_csv("benchmark_data/benchmark1/test1/eval_bulk.tsv", sep='\t', index_col=0).index
    s2_peaks = pd.read_csv("benchmark_data/benchmark1/test2/eval_bulk.tsv", sep='\t', index_col=0).index
    s3_peaks = pd.read_csv("benchmark_data/benchmark1/test3/eval_bulk.tsv", sep='\t', index_col=0).index
    s4_peaks = pd.read_csv("benchmark_data/benchmark1/test4/eval_bulk.tsv", sep='\t', index_col=0).index
    s5_peaks = pd.read_csv("benchmark_data/benchmark1/test5/eval_bulk.tsv", sep='\t', index_col=0).index

    s1_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad")
    s2_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D11T1.h5ad")
    s3_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T1.h5ad")
    s4_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T2.h5ad")
    s5_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T3.h5ad")

    peaks_list = [s1_peaks, s2_peaks, s3_peaks, s4_peaks, s5_peaks]

    for i in range(5):

        peaks = peaks_list[i]

        pb_1, pb_1_props = generate_training_pseudobulks(s1_adata, peaks, cell_type_index, n_pseudobulks=20)
        pb_2, pb_2_props = generate_training_pseudobulks(s2_adata, peaks, cell_type_index, n_pseudobulks=20)
        pb_3, pb_3_props = generate_training_pseudobulks(s3_adata, peaks, cell_type_index, n_pseudobulks=20)
        pb_4, pb_4_props = generate_training_pseudobulks(s4_adata, peaks, cell_type_index, n_pseudobulks=20)
        pb_5, pb_5_props = generate_training_pseudobulks(s5_adata, peaks, cell_type_index, n_pseudobulks=20)

        pseudobulks = [pb_1, pb_2, pb_3, pb_4, pb_5]
        pseudobulk_props = [pb_1_props, pb_2_props, pb_3_props, pb_4_props, pb_5_props]

        pseudobulks.pop(i)
        pseudobulk_props.pop(i)

        X = pd.concat(pseudobulks)
        y = pd.concat(pseudobulk_props)

        X.to_csv(f"benchmark_data/benchmark1/test{i+1}/training_bulk.csv")
        y.to_csv(f"benchmark_data/benchmark1/test{i+1}/training_bulk_props.csv")

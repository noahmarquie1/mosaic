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
    """Build a reference signature matrix from annotated single-cell ATAC data.

    Counts are pooled across every dataset in ``adata_list`` for each cell type,
    then divided by the number of contributing cells, so each column is a mean
    per-cell accessibility profile rather than a raw sum. Only peaks shared by
    all datasets are considered, and the matrix is finally restricted to the
    most variable peaks -- the ones that actually carry cell-type signal.

    Cell types labelled ``Unk``, ``UNK`` or ``Unknown`` are skipped, as are cell
    types with fewer than ``min_cells`` cells pooled across all datasets.

    Parameters
    ----------
    adata_list : list of anndata.AnnData
        Reference datasets with cells in ``obs`` and peaks in ``var``. Peaks are
        intersected across datasets, so ``var_names`` must use a consistent
        naming scheme (e.g. ``chr1:1000-1500``).
    cell_type_col : str, default 'cluster_label'
        Column in ``adata.obs`` holding the cell-type label of each cell.
    min_cells : int, default 10
        Minimum number of cells a cell type needs -- summed over all datasets --
        to get a column in the signature matrix.
    top_n_variable : int, default 2000
        Number of peaks to retain, ranked by variance across cell-type columns.
    dest : str or None, default None
        If given, the signature matrix is also written to this path as TSV.

    Returns
    -------
    pandas.DataFrame
        Signature matrix of shape ``(top_n_variable, n_cell_types)``, indexed by
        peak and with one column per retained cell type.

    Notes
    -----
    The returned values are on a raw count scale. Callers typically normalize
    per column (e.g. CPM followed by ``log1p``) before deconvolution, since the
    matrix itself carries no library-size correction.

    Examples
    --------
    >>> sig = generate_signature_matrix(
    ...     adata_list=[s1, s2, s3],
    ...     cell_type_col="cluster_label",
    ...     top_n_variable=2_000,
    ... )
    >>> sig.shape
    (2000, 12)
    """
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
    """Collapse single-cell data into bulk mixtures with known composition.

    This is the evaluation half of the benchmark. Real bulk ATAC-seq comes with
    no ground truth, so there is no way to tell a good deconvolution from a bad
    one. Instead we take annotated single-cell data and sum it back down into a
    bulk-like profile: the result *is* a bulk mixture as far as any deconvolution
    model can tell, but because every contributing cell carries a label, the true
    proportions are known exactly (recovered separately by
    ``mosaic.evaluate.get_true_proportions`` or
    ``mosaic.evaluate.get_adata_proportions``). Estimated proportions can then be
    scored against them.

    One mixture is produced per group in ``sample_col`` -- typically per donor --
    so a single dataset yields several independent test cases. Cells are pooled
    per group, restricted to ``peaks``, and divided by the group's cell count to
    give a mean per-cell profile. Peaks in ``peaks`` that the dataset lacks become
    explicit zero rows, keeping every output on the signature matrix's peak index.

    Parameters
    ----------
    adata_list : list of anndata.AnnData
        Held-out datasets to collapse. These must be disjoint from the data used
        to build the signature matrix, or the evaluation is circular.
    peaks : pandas.Index
        Peak index the output must conform to -- normally ``signature.index``.
    dest : str or None, default None
        If given, the bulk matrix is also written to this path as TSV.
    sample_col : str, default 'sample_id'
        Column in ``adata.obs`` defining the mixture grouping. If absent from a
        dataset, that whole dataset collapses into one mixture named
        ``sample_{i}``.
    dataset_prefix : bool, default False
        Prefix mixture names with ``dataset{i}_``. Needed when two datasets reuse
        the same sample IDs, which would otherwise be summed together.

    Returns
    -------
    pandas.DataFrame
        Mixtures of shape ``(len(peaks), n_samples)``, indexed by peak. Samples
        with zero total counts are dropped with a warning.

    Notes
    -----
    Nothing is randomized: every cell in a group contributes exactly once. That
    makes the mixture a deterministic function of the input, so evaluation
    variance comes from the models, not the test data.

    Values are mean raw counts -- no log transform. Whatever normalization was
    applied to the signature matrix must be applied here too, or the two live on
    different scales and the fit is meaningless.
    """
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
                                   alpha=1.0, sparse_alpha=0.1, sparse_frac=0.3, random_state=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build randomized training mixtures with known proportions for supervised models.

    The supervised models in this benchmark cannot be fit against a signature
    matrix -- they need labelled examples of the mapping they are supposed to
    learn. This function manufactures them: it repeatedly draws a random
    composition, samples that many cells of each type out of real scATAC-seq data,
    and sums them into a bulk-like profile. Each output row is a synthetic mixture
    and each corresponding row of ``props`` is the composition that actually went
    into it, giving ``(X, y)`` pairs for
    [`rf_deconvolve`][mosaic.deconvolve.rf_deconvolve] and
    [`xgb_deconvolve`][mosaic.deconvolve.xgb_deconvolve].

    **This is the most experimental and least settled part of the experiment.**
    Every choice about how mixtures are simulated -- what compositions are
    plausible, how many cells to pool, whether to model depth or noise or batch --
    is a prior the supervised models will inherit and then be credited or blamed
    for. Elaborate simulators risk measuring the simulator instead of the model.
    The response here is deliberate minimalism: assume as little as possible and
    keep the function basic and modular, so that what the supervised models learn
    is attributable to the data rather than to simulation machinery. Concretely,
    the only assumptions made are that compositions are Dirichlet-distributed, that
    a fixed number of cells is pooled per mixture, and that cells of a type are
    interchangeable. Nothing about sequencing depth, doublets, batch effects, or
    donor structure is modelled. Every knob is exposed as a parameter rather than
    baked in, so any of these assumptions can be swapped out and its effect
    measured, instead of being hidden inside the function.

    Compositions come from a Dirichlet in two regimes, mixed by ``sparse_frac``:
    ``alpha=1.0`` is uniform over the simplex -- the uninformative choice, no cell
    type favoured -- while ``sparse_alpha=0.1`` concentrates mass on a few types.
    Both are needed because real samples are sometimes balanced and sometimes
    dominated by one population, and a model trained only on balanced mixtures
    never learns the dominated case.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell reference to sample from. Must be in memory, not
        backed.
    peaks : pandas.Index
        Peak index the output must conform to -- normally the evaluation bulk's
        index, so training and target features line up.
    cell_type_index : pandas.Index
        Cell types to model, defining the columns of ``props``. Types absent from
        ``adata`` are skipped when sampling but still appear as all-zero columns.
    cell_type_col : str, default 'cluster_label'
        Column in ``adata.obs`` holding cell-type labels.
    n_pseudobulks : int, default 1000
        Number of mixtures to generate.
    cells_per_pseudobulk : int, default 300
        Cells pooled per mixture. Low values make the realized composition noisier
        relative to the drawn target; high values make mixtures smoother than real
        samples.
    alpha : float, default 1.0
        Dirichlet concentration for the dense regime. 1.0 is uniform over the
        simplex.
    sparse_alpha : float, default 0.1
        Dirichlet concentration for the sparse regime, where a few cell types
        dominate.
    sparse_frac : float, default 0.3
        Fraction of mixtures drawn from the sparse regime.
    random_state : int, default 0
        Seed for composition draws and cell sampling, making the whole set
        reproducible.

    Returns
    -------
    bulk : pandas.DataFrame
        Mixtures of shape ``(n_pseudobulks, len(peaks))``, ``log1p`` of mean counts
        per cell.
    props : pandas.DataFrame
        Realized proportions of shape ``(n_pseudobulks, len(cell_type_index))``,
        row-aligned with ``bulk``.

    Notes
    -----
    ``props`` is counted from the cells actually drawn, not copied from the
    Dirichlet target, so it reflects the label composition the mixture really has
    after integer rounding and sampling noise. Targets are converted to integer
    cell counts by flooring plus largest-remainder allocation, so each mixture
    contains exactly ``cells_per_pseudobulk`` cells.

    Cells are drawn **with replacement**, both within and across mixtures. Rare
    cell types therefore get resampled heavily, and a mixture asking for more cells
    of a type than exist will contain duplicates -- the alternative, capping at the
    available count, would silently distort the composition being trained on.

    Unlike
    [`generate_signature_matrix`][mosaic.preprocessing.generate_signature_matrix],
    this function applies ``log1p`` itself. The target bulk passed to the
    supervised models must be transformed the same way.
    """
    rng = np.random.default_rng(random_state)
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
    bulk = np.log1p(bulk / cells_per_pseudobulk)

    props = pd.DataFrame([
        pd.Series(labels).value_counts(normalize=True).reindex(cell_type_index, fill_value=0.0)
        for labels in sampled_labels
    ])
    props.index = bulk.index

    print(f"Generated training pseudobulks, shape: {bulk.shape}")

    return bulk, props

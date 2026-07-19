from mosaic.evaluate import get_true_proportions, get_adata_proportions
import scanpy as sc
import pandas as pd
from mosaic.preprocessing import generate_signature_matrix, generate_eval_pseudobulk, generate_training_pseudobulks, build_metacells, filter_zero_variance_peaks
import os
import pyranges as pr
import scipy.sparse as sp
import anndata as ad
import numpy as np

celltype_index: pd.Series = pd.read_csv("eval_data/celltype_mapping.csv").set_index("original_label")["mapped_label"]
celltype_mapping = celltype_index.to_dict()

bone_marrow_labels = {
    "HSPC": "Hematopoietic Stem Cell",
    "01_HSC": "Hematopoietic Stem Cell",

    "Mono-1": "Classical Monocyte",
    "11_CD14.Mono.1": "Classical Monocyte",

    "Mono-2": "Non-Classical Monocyte",
    "12_CD14.Mono.2": "Non-Classical Monocyte",

    "pDC": "Plasmacytoid Dendritic Cell",
    "09_pDC": "Plasmacytoid Dendritic Cell",

    "NK": "Natural Killer Cell",
    "25_NK": "Natural Killer Cell",

    "B": "B Cell",
    "17_B": "B Cell",

    "preB": "Pre-B Cell",
    "16_Pre.B": "Pre-B Cell",

    "Ery-late": "Late Erythroid Progenitor",
    "03_Late.Eryth": "Late Erythroid Progenitor",

    "Ery-early": "Early Erythroid Progenitor",
    "02_Early.Eryth": "Early Erythroid Progenitor",

    "CD4": "CD4+ T Cell",
    "20_CD4.N1": "CD4+ T Cell",
    "21_CD4.N2": "CD4+ T Cell",
    "22_CD4.M": "CD4+ T Cell",

    "CD8": "CD8+ T Cell",
    "19_CD8.N": "CD8+ T Cell",
    "23_CD8.EM": "CD8+ T Cell",
    "24_CD8.CM": "CD8+ T Cell",

    "CLP": "Common Lymphoid Progenitor",
    "06_CLP.1": "Common Lymphoid Progenitor",
    "15_CLP.2": "Common Lymphoid Progenitor",
}


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


def create_training_data(training_samples, peaks, celltype_index, out_dir):
    training_pb: list[pd.DataFrame] = []
    training_pb_props: list[pd.DataFrame] = []

    for sample in training_samples:
        sample = sample.to_memory() if sample.isbacked else sample
        pb, pb_props = generate_training_pseudobulks(sample, peaks, celltype_index, n_pseudobulks=2000)
        training_pb.append(pb)
        training_pb_props.append(pb_props)

    X = pd.concat(training_pb)
    y = pd.concat(training_pb_props)

    X.to_csv(out_dir + "training_bulk.csv")
    y.to_csv(out_dir + "training_bulk_props.csv")


def create_benchmark1(sig_exists=False, bulk_exists=False, training_data_exists=False):

    s1 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad", backed="r+")
    s2 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D11T1.h5ad", backed="r+")
    s3 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T1.h5ad", backed="r+")
    s4 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T2.h5ad", backed="r+")
    s5 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T3.h5ad", backed="r+")

    for i in range(1, 6):
        sig_samples = [s1, s2, s3, s4, s5]
        bulk_sample = sig_samples.pop(i-1)

        os.makedirs(f"benchmark_data/benchmark1/test{i}", exist_ok=True)

        if sig_exists:
            sig = pd.read_csv(f"benchmark_data/benchmark1/test{i}/signature.tsv", sep='\t', index_col=0)
        else:
            sig: pd.DataFrame = generate_signature_matrix(
                adata_list=sig_samples,
                cell_type_col="cluster_label"
            )

            sig = pd.DataFrame(
                np.log1p(sig.div(sig.sum(axis=0), axis=1) * 1e6),
                index=sig.index,
                columns=sig.columns,
            )

            sig.to_csv(f"benchmark_data/benchmark1/test{i}/signature.tsv", sep='\t')


        if bulk_exists:
            bulk = pd.read_csv(f"benchmark_data/benchmark1/test{i}/eval_bulk.tsv", sep='\t', index_col=0).iloc[:, 0]
        else:
            bulk = generate_eval_pseudobulk(
                adata_list=[bulk_sample],
                peaks=sig.index,
                dest=f"benchmark_data/benchmark1/test{i}/eval_bulk.tsv",
                sample_col="Donor (HSC)",
            )

        if not training_data_exists:
            celltype_index = sig.columns
            create_training_data(sig_samples, bulk.index, celltype_index, f"benchmark_data/benchmark1/test{i}/")

        true_props = get_true_proportions(
            fragments_dir=f"eval_data/granja/pbmc/sample{i}/",
            cell_type_col="BioClassification",
            unknown_label="Unk",
        )

        true_props = true_props.rename(index=celltype_mapping)
        true_props = true_props.rename_axis("cell_type")
        true_props = true_props.groupby(level=0).sum()
        true_props.to_csv(f"benchmark_data/benchmark1/test{i}/true_proportions.csv")


def filter_adata_celltypes(adata, mapping, cell_type_col="Cell_type (HSC)"):
    mask = adata.obs[cell_type_col].isin(mapping.keys())
    adata = adata[mask].copy()
    adata.obs[cell_type_col] = adata.obs[cell_type_col].map(mapping)
    return adata


def create_benchmark2(sig_exists=False, bulk_exists=False, training_data_exists=False):
    granja_sample = sc.read_h5ad("data/Bone_marrow_tissue-cell_by_peak/Granja2019/Granja2019-bone_marrow_tissue/Granja2019-bone_marrow_tissue-cell_by_peak.h5ad")
    lareau_sample = sc.read_h5ad("data/Bone_marrow_tissue-cell_by_peak/Lareau2019/Lareau2019-bone_marrow_tissue/Lareau2019-bone_marrow_tissue-cell_by_peak.h5ad")

    granja_sample = granja_sample.to_memory()
    lareau_sample = lareau_sample.to_memory()

    granja_sample.var_names_make_unique()
    lareau_sample.var_names_make_unique()

    granja_sample = filter_adata_celltypes(granja_sample, bone_marrow_labels)
    lareau_sample = filter_adata_celltypes(lareau_sample, bone_marrow_labels)

    granja_gr, _ = to_pyranges(granja_sample.var_names)
    lareau_gr, _ = to_pyranges(lareau_sample.var_names)

    merged = pr.concat([granja_gr, lareau_gr]).merge()

    granja_sample = filter_peaks_by_pyranges(granja_sample, merged)
    lareau_sample = filter_peaks_by_pyranges(lareau_sample, merged)

    common_peaks = granja_sample.var_names.intersection(lareau_sample.var_names)
    granja_sample = granja_sample[:, common_peaks].copy()
    lareau_sample = lareau_sample[:, common_peaks].copy()

    # Normalize with Scanpy Combat
    granja_meta = build_metacells(granja_sample, cell_type_col="Cell_type (HSC)")
    lareau_meta = build_metacells(lareau_sample, cell_type_col="Cell_type (HSC)")

    granja_meta.obs["study"] = "Granja"
    lareau_meta.obs["study"] = "Lareau"

    combined = ad.concat([granja_meta, lareau_meta], join="outer")
    combined.obs_names_make_unique()
    combined = filter_zero_variance_peaks(combined, batch_col="study")

    combined.layers["counts"] = combined.X.copy()
    sc.pp.normalize_total(combined, target_sum=1e6)

    # Do combat preprocessing with surrounding log transformation
    sc.pp.log1p(combined)
    sc.pp.combat(combined, key="study", covariates=["Cell_type (HSC)"])  # preserve cell-type signal
    combined.X = np.expm1(combined.X)

    combined.X[combined.X < 0] = 0 # Removes negative values

    granja_meta = combined[combined.obs["study"] == "Granja"].copy()
    lareau_meta = combined[combined.obs["study"] == "Lareau"].copy()

    if sig_exists:
        sig = pd.read_csv(f"benchmark_data/benchmark2/signature.tsv", sep='\t', index_col=0)
    else:
        sig = generate_signature_matrix(
            adata_list=[granja_meta],
            cell_type_col="Cell_type (HSC)",
            min_cells=2,
        )
        sig = np.log1p(sig)
        sig.to_csv(f"benchmark_data/benchmark2/signature.tsv", sep='\t')

    if bulk_exists:
        bulk = pd.read_csv(f"benchmark_data/benchmark2/eval_bulk.tsv", sep='\t', index_col=0).iloc[:, 0]
    else:
        bulk = generate_eval_pseudobulk(
            adata_list=[lareau_meta],
            peaks=sig.index,
            sample_col="Donor (HSC)",
        )
        bulk = np.log1p(bulk)
        bulk.to_csv(f"benchmark_data/benchmark2/eval_bulk.tsv", sep='\t')

    if not training_data_exists:
        celltype_index = sig.columns
        create_training_data(granja_meta, bulk.index, celltype_index, f"benchmark_data/benchmark2/")

    # Ground truth must come from the original per-cell Lareau labels, not the
    # metacell-collapsed object (which caps every type at max_metacells_per_type
    # and would otherwise flatten true abundance differences to near-uniform).
    true_props = get_adata_proportions(
        lareau_sample,
        "Cell_type (HSC)",
        True,
        "Unknown",
    )

    true_props.to_csv(f"benchmark_data/benchmark2/true_proportions.csv")



BENCHMARK1 = False
BENCHMARK2 = True

if __name__ == "__main__":
    if BENCHMARK1:
        create_benchmark1(sig_exists=False, bulk_exists=False, training_data_exists=True)
    if BENCHMARK2:
        create_benchmark2(sig_exists=False, bulk_exists=False, training_data_exists=True)

from mosaic.evaluate import get_true_proportions
import scanpy as sc
import pandas as pd
from data_preparation.prepare_data import construct_signature, construct_pseudobulk
from mosaic.preprocessing import generate_training_pseudobulks
import os

celltype_index: pd.Series = pd.read_csv("benchmark_data/benchmark_labels.csv").set_index("original_label")["mapped_label"]
celltype_mapping = celltype_index.to_dict()


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
            sig = construct_signature(
                adata_list=sig_samples,
                dest=f"benchmark_data/benchmark1/test{i}/signature.tsv",
                cell_type_col="cluster_label"
            )

        if bulk_exists:
            bulk = pd.read_csv(f"benchmark_data/benchmark1/test{i}/eval_bulk.tsv", sep='\t', index_col=0).iloc[:, 0]
        else:
            bulk = construct_pseudobulk(
                adata_list=[bulk_sample],
                signature=sig,
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


def create_benchmark2(sig_exists=False, bulk_exists=False):
    pass



BENCHMARK1 = True
if __name__ == "__main__":
    if BENCHMARK1:
        create_benchmark1(sig_exists=False, bulk_exists=True, training_data_exists=True)

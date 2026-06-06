from mosaic.evaluate import get_true_proportions
import scanpy as sc
import pandas as pd
from data_preparation.prepare_data import construct_signature, construct_pseudobulk

celltype_mapping: pd.Series = pd.read_csv("eval_data/celltype_mapping.csv").set_index("original_label")["mapped_label"]

s1 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad", backed="r+")
s2 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D11T1.h5ad", backed="r+")
s3 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T1.h5ad", backed="r+")
s4 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T2.h5ad", backed="r+")
s5 = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D12T3.h5ad", backed="r+")


def create_benchmark1():
    for i in range(1, 6):
        sig_samples = [s1, s2, s3, s4, s5]
        bulk_sample = sig_samples.pop(i-1)
        """

        sig = construct_signature(
            adata_list=sig_samples,
            dest=f"benchmark_data/benchmark1/test{i}/signature.tsv",
            cell_type_col="cluster_label"
        )

        construct_pseudobulk(
            adata_list=[bulk_sample],
            signature=sig,
            dest=f"benchmark_data/benchmark1/test{i}/bulk.tsv",
            sample_col="Donor (HSC)",
        )
        """

        true_props = get_true_proportions(
            fragments_dir=f"eval_data/granja/pbmc/sample{i}/",
            cell_type_col="BioClassification",
            unknown_label="Unk",
        )

        true_props = true_props.rename(index=celltype_mapping)
        true_props = true_props.rename_axis("cell_type")
        true_props = true_props.groupby(level=0).sum()
        true_props.to_csv(f"benchmark_data/benchmark1/test{i}/true_proportions.csv")


BENCHMARK1 = True
if __name__ == "__main__":
    if BENCHMARK1:
        create_benchmark1()

import os
import snapatac2 as snap
from pathlib import Path
from snapatac2 import genome
import pandas as pd
import warnings
import anndata as ad

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Align cell-type labels in advance (using satpathy format)
barcode_mappings = pd.read_csv(Path("eval_data/benchmark_barcode_mappings.csv")).set_index('original_label')['original_label']

# Ensure that a directory name ends with a slash
def finish_dirname(dirname: str) -> str:
    if len(dirname) > 0 and dirname[-1] != "/":
        dirname += "/"
    return dirname


# Add tiled h5 files for each fragment file in a directory
#  - assumes that the fragment files are in tsv or tsv.gz format
def fragments_to_h5ad(fragments_dir, h5ad_dir):
    fragments_dir = finish_dirname(fragments_dir)
    h5ad_dir = finish_dirname(h5ad_dir)

    if not os.path.exists(h5ad_dir):
        os.makedirs(h5ad_dir)

    for file in os.listdir(fragments_dir):
        if file.endswith(".tsv") or file.endswith(".tsv.gz"):
            filename = os.path.splitext(file)[0].split(".")[0]
            print("Processing ", filename)
            data = snap.pp.import_fragments(
               Path(fragments_dir + file),
               chrom_sizes=genome.hg38,
               sorted_by_barcode=False,
               file=Path(h5ad_dir + filename + ".h5ad"),
            )
            print("Finished importing fragments for ", filename)
            snap.pp.add_tile_matrix(
               data,
               bin_size=500,
               inplace=True
            )
            print("Finished adding tile matrix for ", filename)


# adds cluster labels to h5 files in a given directory
def add_corpus_cluster_labels(label_dir, h5ad_dir):
    label_dir = finish_dirname(label_dir)
    h5ad_dir = finish_dirname(h5ad_dir)

    # compile cluster labels into a single pd.Dataframe
    label_series = pd.Series()
    for file in os.listdir(label_dir):
        if file.endswith(".csv") and file != "barcode_mapping.csv":
            label_section = pd.read_csv(label_dir + file, index_col=0)
            label_section = pd.Series(
                label_section["cell_type"].values,
                index=label_section["Cell_type (HSC)"].index
            )
            label_series = pd.concat([label_series, label_section], axis=0)
    label_series = label_series[~label_series.index.duplicated(keep='first')]
    label_series.index.name = 'barcode'
    label_series.name = 'cluster_label'
    label_series.to_csv(label_dir + "barcode_mapping.csv")
    print("Finished compiling cluster labels:")

    # add cluster labels to h5 files
    for file in os.listdir(h5ad_dir):
        if file.endswith(".h5ad"):
            print("Adding cluster labels to ", h5ad_dir + file)
            adata = ad.read_h5ad(Path(h5ad_dir + file))
            adata.obs["cluster_label"] = label_series.reindex(adata.obs_names).fillna("Unknown")
            print(adata.obs["cluster_label"].value_counts())
            adata.write_h5ad(Path(h5ad_dir + file))

    print("Finished adding cluster labels to h5 files.")


def corpus_fragments_to_h5ad(corpus_dir, h5ad_dir):
    # make h5ad directory if it doesn't exist
    if not os.path.exists(h5ad_dir):
        os.makedirs(h5ad_dir)

    fragments_to_h5ad(corpus_dir, h5ad_dir)
    add_corpus_cluster_labels(corpus_dir, h5ad_dir)


if __name__ == "__main__":
    # any code here
    add_corpus_cluster_labels("buenrostro/bone_marrow/", "buenrostro/bone_marrow/h5ad/")
    #add_corpus_cluster_labels("granja/bone_marrow/", "granja/bone_marrow/h5ad/")
    #add_corpus_cluster_labels("buenrostro/bone_marrow/", "buenrostro/bone_marrow/h5ad/")

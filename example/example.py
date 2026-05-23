from data_preparation.prepare_data import construct_signature, construct_pseudobulk
import pandas as pd
from mosaic.deconvolve import nnls_deconvolve
from mosaic.evaluate import get_true_proportions

"""
s1 = sc.read_h5ad("eval_data/granja/bone_marrow/h5ad/Granja2019-bone_marrow_tissue-D5T1.h5ad", backed="r+")
s2 = sc.read_h5ad("eval_data/granja/bone_marrow/h5ad/Granja2019-bone_marrow_tissue-D6T1.h5ad", backed="r+")
s3 = sc.read_h5ad("eval_data/granja/bone_marrow/h5ad/Granja2019-bone_marrow_tissue-D7T1.h5ad", backed="r+")

sig = construct_signature(
    adata_list=[s1, s2], 
    dest="temp/signature.tsv",
    cell_type_col="cluster_label" 
)

# 2. Generate the pseudo-bulk
bulk = construct_pseudobulk(
    adata_list=[s3], 
    signature=sig,
    dest="temp/bulk.tsv",
    sample_col="Donor (HSC)",
)
"""

sig = pd.read_csv("data/signatures/granja_signature.tsv", sep='\t', index_col=0)
bulk = pd.read_csv("data/bulk_mixtures/granja_bulk.tsv", sep='\t', index_col=0).iloc[:, 0]

print(sig.head())
print(bulk.head())

est_props = nnls_deconvolve(sig, bulk)
print(est_props)




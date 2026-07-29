# Data Preprocessing

Reference data can be preprocessed from AnnData (h5/h5ad) format. Statistical models require the use of a signature matrix, while Supervised-Learning models require pseudobulks for training. 

Targets for deconvolution should be bulk mixtures, with the same peaks as the signature matrices and/or pseudobulks. If performing deconvolution on ATAC-seq data at single-cell resolution — for instance for evaluation with previously known proportions, a `generate_eval_pseudobulk` function is provided to generate a valid bulk mixture.

## Generating a Signature Matrix

::: mosaic.preprocessing.generate_signature_matrix

## Generating Pseudobulks

### Supervised Model Training
::: mosaic.preprocessing.generate_training_pseudobulks

### Evaluation
::: mosaic.preprocessing.generate_eval_pseudobulk

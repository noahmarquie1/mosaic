<pre style="background: transparent; border: none; padding: 0; font-size: 1.3em;">
 __  __  ___  ____    _    ___ ____
|  \/  |/ _ \/ ___|  / \  |_ _/ ___|
| |\/| | | | \___ \ / _ \  | | |
| |  | | |_| |___) / ___ \ | | |___
|_|  |_|\___/|____/_/   \_\___\____|
</pre>
## Multi-method Open-chromatin Solver for ATAC-seq Inference of Composition

A systematic benchmark of machine learning methods for bulk ATAC-seq deconvolution. We evaluate a range of approaches — from regularized regression to gradient boosting and constrained neural networks — to ask how much model complexity chromatin accessibility deconvolution actually requires.

## Usage

XXX

## Methods Tested

1. Non-Negative Least Squares (NNLS)
2. Elastic Net
3. Support Vector Regression (SVR)
4. Random Forests
5. Gradient Boosting with XGBoost

## Evaluation


Four benchmarks were conducted to evaluate the performance of our approaches, using scATAC-seq PBMC data from studies Granja et. al. 2019 and Satpathy et. al. 2019. 

The signature matrices, bulk matrices and true proportions for each benchmark are provided in `benchmark_data/` for recreation.

1. For *Benchmark 1*, we construct a signature matrix from 4 samples in Granja, and estimate proportions of the remaining sample in Granja, repeating until the proportions of all 5 samples are estimated
2. For *Benchmark 2*, we construct a signature matrix from 16 samples in Satpathy, and estimate proportions of the remaining sample in Satpathy, repeating until the proportions of all 17 samples are estimated
3. For *Benchmark 3*, we construct a signature matrix from 5 samples in Granja, and we average results from single-sample predictions among 17 samples in Satpathy
4. For *Benchmark 4*, we construct a signature matrix from 17 samples in Satpathy, and average results from single-sample predictions among 5 samples in Granja
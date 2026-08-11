# Getting Started


Mosaic is a systematic benchmark and suite of machine learning methods for bulk ATAC-seq deconvolution. We evaluate a range of approaches — from Non-Negative Least Squares to XGBoost — to ask whether ML complexity necessarily translates to more accurate deconvolution results.

## Project Structure

Mosaic is composed of two main parts:
1. Preprocessing
2. Deconvolution

The preprocessing module provides tools for building signature matrices, bulk mixtures and pseudobulks directly from AnnData format using SciPy, while the deconvolution module provides functionality to perform deconvolution using any of the five benchmarked ML algorithms

## Benchmarks

Benchmarks were conducted using PBMC data from the study Granja et. al. 2019, provided by the Tsinghua Human scATAC-seq Corpus. We separate the data by sample (5 in the study), and perform five corresponding benchmarks, each where the ML models are trained using data from four samples, and a final sample is held out for evaluation. 

### Replicating Benchmarks

To replicate benchmarks, first unzip the `benchmark_data` zip file provided in the repository. Then call `python generate_benchmarks.py` to generate signature matrices, bulk mixtures, and pseudobulks for each benchmark using the MOSAIC's preprocessing module. Finally, run `example.py` to run deconvolution on the generated examples, which will give comparable results to those reported below.

### Overall Results


### Individual Results

#### Benchmark 1

#### Benchmark 2

#### Benchmark 3

#### Benchmark 4

#### Benchmark 5

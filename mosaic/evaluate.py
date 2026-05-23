import numpy as np
import pandas as pd
from collections import Counter
import gzip
from scipy.stats import pearsonr


def evaluate_deconvolution(estimated_proportions: pd.Series, true_proportions: pd.Series) -> dict:
    # Compute error metrics
    errors = estimated_proportions - true_proportions
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    rmse = np.sqrt(squared_errors.mean())
    mae = abs_errors.mean()
    max_error = abs_errors.max()
    pcc = pearsonr(true_proportions.values, estimated_proportions.values).pvalue

    print("─" * 70)
    print(f"\nError Metrics:")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"  Max Absolute Error:             {max_error:.4f}")
    print(f"  Pearson Correlation:            {pcc:.4f}")
    print("=" * 70)

    return {
        'true_proportions': true_proportions,
        'estimated_proportions': estimated_proportions,
        'errors': errors,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'correlation': pcc
    }


def get_true_proportions(
        fragments_file: str,
        mapping: pd.Series,
        max_fragments: int = None,
        batch_size: int = 100_000,
    ) -> pd.Series:

    counts = Counter()
    barcodes_batch: list[str] = []

    n = 0
    with gzip.open(fragments_file, "rt", encoding='utf-8') as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                barcodes_batch.append(parts[3])

            n += 1
            if max_fragments is not None and n >= max_fragments:
                break

            if len(barcodes_batch) >= batch_size:
                for bc in barcodes_batch:
                    ct = mapping.get(bc)
                    if ct is not None:
                        counts[ct] += 1
                barcodes_batch.clear()

    # Finish the last batch
    if barcodes_batch:
        for bc in barcodes_batch:
            ct = mapping.get(bc)
            if ct is not None:
                counts[ct] += 1

    total = sum(counts.values())
    if total == 0:
        print("Warning: No matched barcodes found!")
        return pd.Series(dtype=float)

    return pd.Series(dict(counts), dtype=float) / total
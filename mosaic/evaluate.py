import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr


def evaluate_deconvolution(estimated_proportions: pd.Series, true_proportions: pd.Series) -> dict:
    errors = estimated_proportions - true_proportions
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    rmse = np.sqrt(squared_errors.mean())
    mae = abs_errors.mean()
    max_error = abs_errors.max()
    pcc = pearsonr(true_proportions.values, estimated_proportions.values).statistic

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
        fragments_dir: str | Path,
        cell_type_col: str = "cell_type",
        drop_unknown: bool = True,
        unknown_label: str = "Unknown",
        metadata_glob: str = "*-metadata.csv",
    ) -> pd.Series:

    fragments_dir = Path(fragments_dir)
    files = list(fragments_dir.glob(metadata_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{metadata_glob}' found in {fragments_dir}"
        )

    meta = pd.concat([pd.read_csv(f, index_col=0) for f in files])

    if cell_type_col not in meta.columns:
        raise ValueError(
            f"Column '{cell_type_col}' not found. "
            f"Available columns: {list(meta.columns)}"
        )

    labels = meta[cell_type_col]

    if drop_unknown:
        labels = labels[~labels.str.contains(unknown_label, case=False, na=False)]

    return labels.value_counts(normalize=True).rename("proportion")

import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet
from sklearn.svm import NuSVR
from sklearn.inspection import permutation_importance

def nnls_deconvolve(signature_matrix: pd.DataFrame,
               mixture_vector: pd.Series) -> pd.Series:

    m = mixture_vector.reindex(signature_matrix.index)
    f, residual = nnls(signature_matrix.values, m.values)

    if f.sum() > 0:
        f = f / f.sum()

    proportions = pd.Series(f, index=signature_matrix.columns)
    return proportions


def elastic_net_deconvolve(signature_matrix: pd.DataFrame,
                           mixture_vector: pd.Series) -> pd.Series:

    model = ElasticNet(alpha=0.01, l1_ratio=0.5, positive=True)
    model.fit(signature_matrix, mixture_vector)
    proportions = pd.Series(model.coef_, index=signature_matrix.columns)

    total = proportions.sum()
    if total > 0:
        proportions = proportions / total

    print_proportions(proportions)
    return proportions


def nu_svr_deconvolve(signature_matrix: pd.DataFrame,
                     mixture_vector: pd.Series) -> pd.Series:
    model = NuSVR(kernel='rbf', nu=0.5, C=1.0, gamma='scale')
    model.fit(signature_matrix, mixture_vector)

    results = permutation_importance(model, signature_matrix, mixture_vector, n_repeats=10)
    proportions = pd.Series(results.importances_mean, index=signature_matrix.columns)

    total = proportions.sum()
    if total > 0:
        proportions = proportions / total

    print_proportions(proportions)
    return proportions


def print_proportions(proportions: pd.Series, top_n=None):
    print("\nEstimated cell type proportions:")
    print("─" * 35)

    n = 0
    for cell_type, proportion in proportions.sort_values(ascending=False).items():
        bar = "█" * int(proportion * 40)
        print(f"  {cell_type:<30} {proportion:.4f}  {bar}")
        n += 1
        if n == top_n:
            break

    print("─" * 35)
    print(f"  {'Total':<25} {proportions.sum():.4f}")

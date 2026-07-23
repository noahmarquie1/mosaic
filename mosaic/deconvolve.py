import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet
from sklearn.svm import NuSVR
import scanpy as sc
import numpy as np

from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

# Core Statistical and ML Models
def nnls_deconvolve(signature_matrix: pd.DataFrame,
               mixture_vector: pd.Series) -> pd.Series:

    A = (signature_matrix.to_numpy(dtype=float))
    b = (mixture_vector.to_numpy(dtype=float))
    f, _ = nnls(A, b)

    if f.sum() > 0:
        f = f / f.sum()

    proportions = pd.Series(f, index=signature_matrix.columns)
    return proportions


def elastic_net_deconvolve(signature_matrix: pd.DataFrame,
                           mixture_vector: pd.Series) -> pd.Series:

    A = (signature_matrix.to_numpy(dtype=float))
    b = (mixture_vector.to_numpy(dtype=float))

    model = ElasticNet(alpha=0.01, l1_ratio=0.5, positive=True)
    model.fit(A, b)
    proportions = pd.Series(model.coef_, index=signature_matrix.columns)

    total = proportions.sum()
    if total > 0:
        proportions = proportions / total

    return proportions


def nu_svr_deconvolve(signature_matrix: pd.DataFrame,
                     mixture_vector: pd.Series) -> pd.Series:

    b = (mixture_vector.to_numpy(dtype=float))

    model = NuSVR(kernel='rbf', nu=0.5, C=1.0, gamma='scale')
    model.fit(signature_matrix, b)

    results = permutation_importance(model, signature_matrix, mixture_vector, n_repeats=10)
    proportions = pd.Series(results.importances_mean, index=signature_matrix.columns)

    total = proportions.sum()
    if total > 0:
        proportions = proportions / total

    return proportions


def random_forests_deconvolve(training_bulks: pd.DataFrame, training_bulk_props: pd.DataFrame,
                       mixture_vector: pd.Series) -> pd.Series:

    print("Starting random forests deconvolution:\n")
    model = RandomForestRegressor(
        n_estimators=850,
        max_depth=10,
        max_features=0.3,
        max_samples=0.7,
        min_samples_leaf=6,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        training_bulks,
        training_bulk_props,
        test_size=0.2,
        random_state=0
    )

    model.fit(X_train, y_train)

    mixture_vector = mixture_vector.to_frame().T
    y_pred = model.predict(mixture_vector)
    y_pred = pd.Series(y_pred[0], index=training_bulk_props.columns)
    print("Finished random forests deconvolution.\n")

    return y_pred


def xgb_deconvolve(training_bulks: pd.DataFrame, training_bulk_props: pd.DataFrame,
                       mixture_vector: pd.Series) -> pd.Series:

    print("Starting xgboost deconvolution:\n")

    X_train, X_test, y_train, y_test = train_test_split(
        training_bulks,
        training_bulk_props,
        test_size=0.2,
        random_state=0
    )

    model = XGBRegressor(
        tree_method="hist",
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.4,
        min_child_weight=5,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=0,
        eval_metric="rmse",
        early_stopping_rounds=50,
    )

    mixture_vector = mixture_vector.to_frame().T
    model.fit(X_train, y_train, verbose=10, eval_set=[(X_test, y_test)])
    y_pred = model.predict(mixture_vector)
    y_pred = pd.Series(y_pred[0], index=training_bulk_props.columns)

    y_pred = y_pred.clip(lower=0)
    total = y_pred.sum()
    if total > 0:
        y_pred = y_pred / total

    print("Finished xgboost deconvolution.\n")
    return y_pred


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


if __name__ == "__main__":
    s1_adata = sc.read_h5ad("eval_data/granja/pbmc/h5ad/Granja2019-peripheral_blood_mononuclear_cells-D10T1.h5ad")
    print(s1_adata.var[:5])
    print(s1_adata.obs.columns)
    print(s1_adata.obs['cluster_label'][:5])

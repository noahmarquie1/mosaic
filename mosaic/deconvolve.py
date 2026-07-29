import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet
from sklearn.svm import NuSVR
from sklearn.multioutput import MultiOutputRegressor
import scanpy as sc
import numpy as np

from sklearn.inspection import permutation_importance
import xgboost as xgb
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


def rf_deconvolve(training_bulks: pd.DataFrame, training_bulk_props: pd.DataFrame,
                       mixture_vector: pd.Series) -> pd.Series:

    print("Starting random forests deconvolution:\n")
    model = RandomForestRegressor(
        n_estimators=20,
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
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    epsilon = 1e-8
    row_sums = y_pred.sum(axis=1, keepdims=True)
    y_pred = y_pred / (row_sums + epsilon)

    y_pred = pd.Series(y_pred[0], index=training_bulk_props.columns)
    print("Finished random forests deconvolution.\n")

    return y_pred


def xgb_deconvolve(X_train, y_train, X_bulk):
    print("Starting xgboost deconvolution:\n")
    params = {
        'n_estimators': 50,
        'objective': 'reg:squarederror',
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'learning_rate': 0.02,
        'random_state': 42
    }

    base_xgb = xgb.XGBRegressor(**params)
    multi_xgb = MultiOutputRegressor(base_xgb)
    multi_xgb.fit(X_train, y_train)

    X_bulk = X_bulk.to_frame().T
    raw_predictions = multi_xgb.predict(X_bulk)

    non_negative_preds = np.clip(raw_predictions, a_min=0, a_max=None)
    normalized_predictions = non_negative_preds / (non_negative_preds.sum(axis=1, keepdims=True) + 1e-8)
    y_pred = pd.Series(normalized_predictions[0], index=y_train.columns)

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

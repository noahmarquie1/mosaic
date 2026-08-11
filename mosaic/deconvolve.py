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
    """Deconvolve a bulk mixture by non-negative least squares.

    Solves ``min ||A f - b||_2`` subject to ``f >= 0``, where ``A`` is the
    signature matrix and ``b`` the bulk mixture, then rescales ``f`` to sum to
    one. This is the reference baseline of the benchmark: no hyperparameters, no
    training data, and the non-negativity constraint alone is enough to make the
    coefficients readable as cell-type proportions.

    Parameters
    ----------
    signature_matrix : pandas.DataFrame
        Peaks (rows) by cell types (columns), as produced by
        [`generate_signature_matrix`][mosaic.preprocessing.generate_signature_matrix].
    mixture_vector : pandas.Series
        Bulk accessibility profile indexed by the same peaks, in the same order,
        as ``signature_matrix``.

    Returns
    -------
    pandas.Series
        Estimated proportions indexed by ``signature_matrix.columns``, summing
        to 1 (or all zeros if the solver returns an all-zero solution).

    Notes
    -----
    Peak alignment is not checked here -- both inputs are converted straight to
    NumPy, so a mismatched index silently produces garbage. Align upstream.

    Examples
    --------
    >>> props = nnls_deconvolve(sig, bulk)
    >>> props.sum()
    1.0
    """
    A = (signature_matrix.to_numpy(dtype=float))
    b = (mixture_vector.to_numpy(dtype=float))
    f, _ = nnls(A, b)

    if f.sum() > 0:
        f = f / f.sum()

    proportions = pd.Series(f, index=signature_matrix.columns)
    return proportions


def elastic_net_deconvolve(signature_matrix: pd.DataFrame,
                           mixture_vector: pd.Series) -> pd.Series:
    """Deconvolve a bulk mixture by non-negative elastic-net regression.

    Same setup as [`nnls_deconvolve`][mosaic.deconvolve.nnls_deconvolve], with an
    L1 + L2 penalty added on top of the non-negativity constraint. The L1 term
    drives implausible cell types to exactly zero and the L2 term shares weight
    between correlated signature columns -- the failure mode NNLS has when two
    cell types have near-identical accessibility profiles. Fitted coefficients
    are rescaled to sum to one.

    Parameters
    ----------
    signature_matrix : pandas.DataFrame
        Peaks (rows) by cell types (columns).
    mixture_vector : pandas.Series
        Bulk accessibility profile indexed by the same peaks, in the same order,
        as ``signature_matrix``.

    Returns
    -------
    pandas.Series
        Estimated proportions indexed by ``signature_matrix.columns``, summing
        to 1 (or all zeros if every coefficient is shrunk to zero).

    Notes
    -----
    Hyperparameters are fixed at ``alpha=0.01`` and ``l1_ratio=0.5`` so the
    benchmark compares model classes rather than tuning budgets. Raising
    ``alpha`` sparsifies the estimate further, at the cost of dropping genuinely
    rare cell types.
    """
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
    """Deconvolve a bulk mixture with a nu-support-vector regressor.

    Fits an RBF-kernel ``NuSVR`` mapping signature columns to the bulk profile,
    in the spirit of CIBERSORT's support-vector formulation. Because the kernel
    is non-linear the model exposes no per-cell-type coefficients, so weights are
    recovered post hoc from permutation importance (10 repeats per cell type) and
    then rescaled to sum to one.

    Parameters
    ----------
    signature_matrix : pandas.DataFrame
        Peaks (rows) by cell types (columns).
    mixture_vector : pandas.Series
        Bulk accessibility profile indexed by the same peaks, in the same order,
        as ``signature_matrix``.

    Returns
    -------
    pandas.Series
        Estimated proportions indexed by ``signature_matrix.columns``, rescaled
        to sum to 1.

    Notes
    -----
    Permutation importance measures how much the fit degrades when a cell type's
    column is shuffled, which is a *proxy* for abundance, not a coefficient. It
    can come out negative for cell types the model ignores, and those negatives
    are carried into the normalization rather than clipped -- so treat the
    resulting vector as a ranking first and a proportion second. Permutation
    importance also refits nothing but re-scores 10 times per column, making this
    the slowest of the statistical models.
    """
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
                       mixture_vector: pd.Series, depth: int = 1) -> pd.Series:
    """Deconvolve a bulk mixture with a multi-output random forest.

    The first of the supervised models: instead of solving against a signature
    matrix, it learns the mixture-to-proportion map directly from simulated
    pseudobulks whose composition is known
    ([`generate_training_pseudobulks`][mosaic.preprocessing.generate_training_pseudobulks]).
    A single forest predicts all cell-type fractions at once; predictions are
    clipped at zero and renormalized, since nothing in the objective enforces a
    simplex.

    Parameters
    ----------
    training_bulks : pandas.DataFrame
        Training pseudobulks, one row per mixture, columns being peaks.
    training_bulk_props : pandas.DataFrame
        Ground-truth proportions aligned row-wise with ``training_bulks``; its
        columns define the cell types of the output.
    mixture_vector : pandas.Series
        The bulk profile to deconvolve, indexed by the same peaks as
        ``training_bulks``.

    Returns
    -------
    pandas.Series
        Estimated proportions indexed by ``training_bulk_props.columns``, summing
        to 1.

    Notes
    -----
    Trees are deliberately constrained (``max_depth=10``, ``max_features=0.3``,
    ``min_samples_leaf=6``) because peaks vastly outnumber training mixtures and
    an unconstrained forest memorizes the simulator instead of the biology.

    The 80/20 ``train_test_split`` currently fits on ``X_train`` only; the held
    out split is not scored, so the reported estimate comes from a model trained
    on 80% of the pseudobulks.
    """
    print("Starting random forests deconvolution:\n")
    if depth == 0:
        model = RandomForestRegressor(
            n_estimators=2,
            max_depth=10,
            max_features=0.3,
            max_samples=0.7,
            min_samples_leaf=6,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            max_features=int(np.sqrt(training_bulks.shape[1])),
            min_samples_leaf=2,
            min_samples_split=5,
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
        )

    model.fit(training_bulks, training_bulk_props)

    mixture_vector = mixture_vector.to_frame().T
    y_pred = model.predict(mixture_vector)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    epsilon = 1e-8
    row_sums = y_pred.sum(axis=1, keepdims=True)
    y_pred = y_pred / (row_sums + epsilon)

    y_pred = pd.Series(y_pred[0], index=training_bulk_props.columns)
    print("Finished random forests deconvolution.\n")

    return y_pred


def xgb_deconvolve(X_train, y_train, X_bulk, depth=1):
    """Deconvolve a bulk mixture with gradient-boosted trees.

    Like [`rf_deconvolve`][mosaic.deconvolve.rf_deconvolve], this learns from
    simulated pseudobulks rather than a signature matrix, but replaces bagging
    with boosting: an ``XGBRegressor`` wrapped in ``MultiOutputRegressor``, so
    one independent booster is fit per cell type. Predictions are clipped at zero
    and renormalized, as the per-cell-type boosters know nothing about each other
    and have no reason to produce a vector that sums to one.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training pseudobulks, one row per mixture, columns being peaks.
    y_train : pandas.DataFrame
        Ground-truth proportions aligned row-wise with ``X_train``; its columns
        define the cell types of the output.
    X_bulk : pandas.Series
        The bulk profile to deconvolve, indexed by the same peaks as ``X_train``.

    Returns
    -------
    pandas.Series
        Estimated proportions indexed by ``y_train.columns``, summing to 1.

    Notes
    -----
    Regularization is doing most of the work here: a low ``learning_rate`` (0.01)
    over 200 rounds, row and column subsampling (0.8 / 0.7), and both L1 and L2
    penalties, all to keep the boosters from fitting pseudobulk simulation
    artifacts.

    Fitting is ``n_cell_types`` separate models, so cost scales linearly with the
    number of columns in ``y_train`` -- this is the most expensive model in the
    benchmark to train.
    """
    print("Starting xgboost deconvolution:\n")
    if depth == 0:
        params = {
            'n_estimators': 2,
            'objective': 'reg:squarederror',
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'learning_rate': 0.01,
            'random_state': 42
        }
    else:
        params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'min_child_weight': 3,
            'reg_lambda': 1.0,
            'reg_alpha': 0.1,
            'tree_method': "hist",
            'n_jobs': -1,
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

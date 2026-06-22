import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet
from sklearn.svm import NuSVR
import scanpy as sc
import anndata as ad

from sklearn.inspection import permutation_importance
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Core Statistical and ML Models
def nnls_deconvolve(signature_matrix: pd.DataFrame,
               mixture_vector: pd.Series) -> pd.Series:

    m = mixture_vector.reindex(signature_matrix.index)
    f, _ = nnls(signature_matrix.values, m.values)

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

    return proportions


def xgb_deconvolve(training_bulks: pd.DataFrame, training_bulk_props: pd.DataFrame,
                       mixture_vector: pd.Series) -> pd.Series:

    training_bulks = training_bulks.drop(columns=["Unnamed: 0"])
    training_bulk_props = training_bulk_props.drop(columns=["Unknown"])
    X_train, X_test, y_train, y_test = train_test_split(
        training_bulks, 
        training_bulk_props, 
        test_size=0.2
    )

    xgb = XGBRegressor(
        tree_method="hist",
        n_estimators=2,
        n_jobs=16,
        max_depth=4,
        multi_strategy="multi_output_tree",
        subsample=0.6,
        eval_metric="rmse"
    )

    #print(y_train.head)
    #print(y_test.head)

    mixture_vector = mixture_vector.to_frame().T
    print(mixture_vector)

    xgb.fit(X_train, y_train, verbose=10, eval_set=[(X_train, y_train)])
    y_pred = xgb.predict(mixture_vector)
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

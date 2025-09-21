import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier


def load_data(npz_file):
    """
    Load embeddings, labels, and grouping info (e.g., gene names).
    """
    npz = np.load(npz_file)
    return npz["X"], npz["y"], npz["hgnc"], npz["idx"]


def run_xgb(X, y, groups, n_splits=10, save_path="xgb_kinase_pathogenicity.json"):
    """
    Train and evaluate tuned XGBoost using GroupKFold cross-validation.
    Also trains on the full dataset and saves the model.

    Parameters
    ----------
    X : np.ndarray
        Embeddings, shape (n_samples, n_features).
    y : np.ndarray
        Labels, shape (n_samples,).
    groups : np.ndarray
        Group identifiers (e.g., gene names), shape (n_samples,).
    n_splits : int
        Number of GroupKFold splits (default=10).
    save_path : str
        Path to save the final trained model.
    """
    params = {
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 1,
        "reg_lambda": 2,
        "reg_alpha": 0.1,
        "scale_pos_weight": (np.sum(y == 0) / np.sum(y == 1)),
        "eval_metric": "logloss",
        "random_state": 42,
        "tree_method": "hist"
    }

    gkf = GroupKFold(n_splits=n_splits)
    scores = {"f1": [], "roc_auc": []}

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]

        scores["f1"].append(f1_score(y[test_idx], y_pred))
        scores["roc_auc"].append(roc_auc_score(y[test_idx], y_prob))

    results = {k: np.mean(v) for k, v in scores.items()}
    print("Cross-validation results:", results)

    final_model = XGBClassifier(**params)
    final_model.fit(X, y)
    final_model.save_model(save_path)
    print(f"Final model trained and saved to {save_path}")

    return results, final_model
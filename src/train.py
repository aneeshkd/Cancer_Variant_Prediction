import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from xgboost import XGBClassifier
import joblib
import os


def load_data(npz_path):
    """
    Load ΔESM embeddings and labels from NPZ file.

    Returns:
        X (np.ndarray): feature matrix
        y (np.ndarray): labels (0/1)
        genes (np.ndarray): gene identifiers for grouping
    """
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["y"], data["genes"]


def build_classifier(pos_weight=1.0):
    """
    Build an XGBoost classifier with preset hyperparameters.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist"
    )


def cross_validate(X, y, genes, n_splits=5, save_models=False, model_dir="models"):
    """
    Perform GroupKFold cross-validation with XGBoost.

    Args:
        X (np.ndarray): features
        y (np.ndarray): labels
        genes (np.ndarray): grouping array (e.g. gene symbols)
        n_splits (int): number of folds
        save_models (bool): whether to save fold models
        model_dir (str): directory to save models if save_models=True

    Returns:
        pd.DataFrame: per-fold metrics
    """
    cv = GroupKFold(n_splits=n_splits)
    results = []

    pos_weight = (sum(y == 0) / sum(y == 1)) if sum(y == 1) > 0 else 1.0

    if save_models:
        os.makedirs(model_dir, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=genes)):
        clf = build_classifier(pos_weight=pos_weight)
        clf.fit(X[train_idx], y[train_idx])

        y_pred = clf.predict_proba(X[test_idx])[:, 1]
        y_label = clf.predict(X[test_idx])

        metrics = {
            "Fold": fold,
            "AUC": roc_auc_score(y[test_idx], y_pred),
            "PR_AUC": average_precision_score(y[test_idx], y_pred),
            "Accuracy": accuracy_score(y[test_idx], y_label)
        }
        results.append(metrics)

        print(f"Fold {fold}: AUC={metrics['AUC']:.3f}, "
              f"PR-AUC={metrics['PR_AUC']:.3f}, "
              f"ACC={metrics['Accuracy']:.3f}")

        if save_models:
            joblib.dump(clf, os.path.join(model_dir, f"xgb_fold{fold}.pkl"))

    return pd.DataFrame(results)


def train_final_model(X, y, out_model="models/xgb_full.pkl"):
    """
    Train final XGBoost model on the full dataset and save to disk.
    """
    pos_weight = (sum(y == 0) / sum(y == 1)) if sum(y == 1) > 0 else 1.0
    clf = build_classifier(pos_weight=pos_weight)
    clf.fit(X, y)

    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(clf, out_model)
    print(f"Final model saved to {out_model}")
    return clf


def main(npz_path="data/esm_embeddings/kinase_variants_delta_pca.npz", save_models=False):
    """
    Run cross-validation and print mean metrics.
    """
    X, y, genes = load_data(npz_path)
    results = cross_validate(X, y, genes, save_models=save_models)
    print("\n=== Cross-validation summary ===")
    print(results.mean(numeric_only=True))
    return results

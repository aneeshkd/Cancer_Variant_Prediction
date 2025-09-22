import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def load_data(npz_file):
    npz = np.load(npz_file)
    return npz["X"], npz["y"], npz["hgnc"], npz["idx"]


def xgb_objective(trial, X, y, groups, n_splits=10):
    params = dict(
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 200, 1500),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        scale_pos_weight=float(np.sum(y == 0) / np.sum(y == 1)),
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        use_label_encoder=False,
    )

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        model = XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        fold_scores.append(roc_auc_score(y[val_idx], preds))

    return float(np.mean(fold_scores))


def run_xgb_optuna(X, y, groups, n_splits=10, n_trials=30, model_path="models/xgb_kinase_pathogenicity.json"):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: xgb_objective(trial, X, y, groups, n_splits), n_trials=n_trials)

    best_params = study.best_trial.params
    print("Best ROC-AUC:", study.best_trial.value)
    print("Best parameters:", best_params)

    final_model = XGBClassifier(**best_params)
    final_model.fit(X, y)
    final_model.save_model(model_path)
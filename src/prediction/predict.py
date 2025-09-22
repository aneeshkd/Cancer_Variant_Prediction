import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
from src.training.train import load_data


def predict(csv_path, npz_file, out_csv):
    """
    Run the trained XGBoost classifier on embedded variants and save predictions.
    """
    X, y, hgnc, idx = load_data(npz_file)
    df = pd.read_csv(csv_path)
    df["prob_pathogenic"] = np.nan
    df["pathogenic_label"] = np.nan

    model = XGBClassifier()
    model.load_model("models/xgb_kinase_pathogenicity.json")
    if not hasattr(model, "n_classes_"):
        model.n_classes_ = 2
        model.classes_ = np.array([0, 1])

    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)

    df.loc[idx, "prob_pathogenic"] = probs
    df.loc[idx, "pathogenic_label"] = labels

    df.to_csv(out_csv, index=False)
    print(f"=====Predictions Complete, Data Saved To: {out_csv}=====")


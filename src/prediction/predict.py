import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
from src.training.train import load_data

def predict(csv_path, npz_file, outdir):
    """
    Run trained XGB classifier 
    """
    print("=====Predicting Labels from XGB Classifier=====")
    X, y, hgnc, idx = load_data(npz_file)
    df = pd.read_csv(csv_path)

    model = XGBClassifier()
    model.load_model("models/xgb_kinase_pathogenicity.json")
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)

    df["prob_pathogenic"] = np.nan
    df["pathogenic_label"] = np.nan

    df.loc[idx, "prob_pathogenic"] = probs
    df.loc[idx, "pathogenic_label"] = labels

    df.to_csv(outdir, index=False)
    print(f"=====Predictions Complete, Data Saved To: {outdir}=====")
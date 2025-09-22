# Cancer Variant Prediction

A machine learning pipeline for predicting the pathogenicity of protein kinase variants using ESM embeddings and XGBoost.

### Features
- Fetches UniProt FASTA sequences for given variants
- Generates embeddings with ESM-2 
- Trains and applies an XGBoost classifier for pathogenicity prediction, hyperparameter tuning using Optuna 
- Outputs probability scores and binary pathogenic labels

### Installation
Clone the repo:
git clone https://github.com/aneeshkd/Cancer_Variant_Prediction.git
cd Cancer_Variant_Prediction

Create environment and install dependencies:
conda create -n cancer_variant_prediction python=3.12
conda activate cancer_variant_prediction
pip install -r requirements.txt

### Usage
Run the full pipeline:
python main.py --input data/intermediate/test_df.csv --outdir results/

Arguments:
--input CSV with variant info (columns: gene, uniprot, wt_aa, pos_protein, mut_aa)
--outdir Output folder for FASTA, embeddings, and predictions

Prediction Output:
A CSV will be created with added columns:
prob_pathogenic : Probability (0–1) that the variant is pathogenic
pathogenic_label : Binary classification (0 = benign, 1 = pathogenic)

### Credit:
ESM2: FASTA embeddings 
ClinVar: Benign variant labels 
COSMIC: Cancer variant labels
UniProt: FASTA sequences 

import os
import argparse

from src.preprocessing.fetch_uniprot import fetch_uniprot_fasta
from src.embeddings.esm_embeddings import featurize_delta
from src.prediction.predict import predict


def main():
    parser = argparse.ArgumentParser(description="Kinase variant pathogenicity prediction pipeline")
    parser.add_argument("--input", required=True, help="Input CSV with variants")
    parser.add_argument("--outdir", default="results", help="Output directory for pipeline results")
    args = parser.parse_args()

    csv_path = args.input
    outdir = args.outdir

    fasta_path = os.path.join(outdir, "fasta")
    npz_path = os.path.join(outdir, "npz")
    npz_file = os.path.join(npz_path, "embedding_deltas.npz")
    out_csv = os.path.join(outdir, "cancer_var_pred.csv")

    os.makedirs(fasta_path, exist_ok=True)
    os.makedirs(npz_path, exist_ok=True)

    fetch_uniprot_fasta(csv_path, fasta_path)
    featurize_delta(csv_path, fasta_path, npz_file)
    predict(csv_path, npz_file, out_csv)


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import torch
import esm
from Bio import SeqIO
from sklearn.decomposition import PCA


def load_fasta_sequence(gene, uniprot, seq_dir="data/fasta"):
    """Load amino acid sequence from FASTA file."""
    path = os.path.join(seq_dir, f"{gene}_{uniprot}.fasta")
    rec = next(SeqIO.parse(path, "fasta"))
    return str(rec.seq)


@torch.no_grad()
def embed_sequence(seq, model, batch_converter, device="cuda", layer=33):
    """Embed the full protein and return per-residue embeddings."""
    labels, strs, toks = batch_converter([("protein", seq)])
    toks = toks.to(device)
    out = model(toks, repr_layers=[layer], return_contacts=False)
    reps = out["representations"][layer][0]
    return reps[1: len(seq)+1].detach().cpu().numpy()


def featurize_delta(
    harmonized_csv="data/intermediate/harmonized.csv",
    seq_dir="data/fasta",
    out_npz="data/esm_embeddings/kinase_variants_delta_pca.npz",
    layer=33,
    pca_components=50
):
    """Compute PCA-reduced ΔESM embeddings (mean-pooled) for variants in harmonized dataset."""
    df = pd.read_csv(harmonized_csv)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    wt_cache = {}
    X, y, genes, kept_idx = [], [], [], []

    for i, row in df.iterrows():
        gene, uniprot = row["gene"], row["uniprot"]
        wt_aa, pos, mut_aa = row["wt_aa"], int(row["protein_pos"]), row["mut_aa"]

        key = (gene, uniprot)
        if key not in wt_cache:
            seq = load_fasta_sequence(gene, uniprot, seq_dir)
            wt_cache[key] = embed_sequence(seq, model, batch_converter, device, layer)

        wt_embs = wt_cache[key]
        if pos < 1 or pos > len(wt_embs):
            continue

        seq = load_fasta_sequence(gene, uniprot, seq_dir)

        if seq[pos-1] != wt_aa:
            continue

        m_seq = seq[:pos-1] + mut_aa + seq[pos:]
        mut_embs = embed_sequence(m_seq, model, batch_converter, device, layer)

        delta_matrix = wt_embs - mut_embs
        features = delta_matrix.mean(axis=0)  

        X.append(features.astype(np.float32))
        y.append(int(row["pathogenic"]))
        genes.append(gene)
        kept_idx.append(i)

    X = np.vstack(X)  
    y = np.array(y, dtype=np.int64)
    genes = np.array(genes)
    kept_idx = np.array(kept_idx)

    pca = PCA(n_components=pca_components, random_state=0)
    X = pca.fit_transform(X)

    np.savez(out_npz, X=X, y=y, genes=genes, rows=kept_idx)
    return out_npz
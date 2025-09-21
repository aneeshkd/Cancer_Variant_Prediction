import os
import pandas as pd
from cyvcf2 import VCF
from src.preprocessing.variant_helpers import parse_hgvs, process_batch

def extract_missense_variants(clinvar_path, genes, batch_size=200, save_csv=True, outdir="../results"):
    """
    Extract missense variants for a target gene from a VCF file
    Returns a pandas DataFrame with annotations
    """
    records = []
    batch = []

    vcf = VCF(clinvar_path)
    for var in vcf:
        gene = (var.INFO.get("GENEINFO") or "").split(":")[0]
        if gene not in genes.keys():
            continue
        if "SO:0001583" not in (var.INFO.get("MC") or ""):
            continue
        if var.INFO.get("CLNSIG") not in ["Benign", "Likely_benign", "Benign/Likely_benign"]:
            continue 

        for alt in var.ALT:
            variant_query = f"{var.CHROM} {var.POS} {var.ID} {var.REF} {alt}"
            uniprot = genes[gene]
            batch.append((variant_query, gene, uniprot))

        if len(batch) >= batch_size:
            records.extend(process_batch(batch))
            batch = []

    if batch:
        records.extend(process_batch(batch))

    df = pd.DataFrame(records)

    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "clinvar_missense.csv")
        df.to_csv(outfile, index=False)
        print(f"Saved results to {outfile}")

    return df.reset_index(drop=True)


def extract_cosmic_variants(cosmic_path, genes, save_csv=True, outdir="../results"):
    """
    Extract missense driver variants for given target genes from COSMIC
    Returns a pandas DataFrame with annotations
    """
    df = pd.read_csv(cosmic_path, sep="\t")

    df = df[df["GENE_SYMBOL"].isin(genes.keys())]
    df = df[df["MUTATION_DESCRIPTION"].str.contains("missense", case=False, na=False)]
    df = df.dropna(subset=["HGVSP"])
    df[["wt_aa", "protein_pos", "mut_aa"]] = df["HGVSP"].apply(parse_hgvs).apply(pd.Series)
    df = df[df["MUTATION_SOMATIC_STATUS"] == "Confirmed somatic variant"]
    df = df.groupby(["GENE_SYMBOL", "wt_aa", "protein_pos", "mut_aa"]).filter(lambda x: len(x) >= 3)

    df_out = pd.DataFrame({
        "hgnc": df["GENE_SYMBOL"],
        "uniprot": df["GENE_SYMBOL"].map(genes),
        "wt_aa": df['wt_aa'],
        "protein_pos": df['protein_pos'],
        "mut_aa": df['mut_aa'],
        "pathogenic": 1
    })

    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "cosmic_missense.csv")
        df_out.to_csv(outfile, index=False)
        print(f"Saved COSMIC driver results to {outfile}")

    return df_out.reset_index(drop=True)


def harmonize_data(clinvar_df, cosmic_df, save_csv=True, outdir="data/intermediate"):
    """
    Merge ClinVar benign and COSMIC driver variants into a harmonized dataset
    Conflict resolution: if the same variant is in both, keep COSMIC (pathogenic=1)
    """
    df = pd.concat([clinvar_df, cosmic_df], ignore_index=True)
    df = df.sort_values(by="pathogenic", ascending=True)
    df = df.drop_duplicates(
        subset=["hgnc", "uniprot", "wt_aa", "protein_pos", "mut_aa"],
        keep="last"
    )

    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "protein_kinase_variants.csv")
        df.to_csv(outfile, index=False)
        print(f"Saved harmonized dataset to {outfile}")

    return df.reset_index(drop=True)
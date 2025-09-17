import os
import sys
import re
import time
import random
import json
import pandas as pd
import requests
from cyvcf2 import VCF
from Bio.Data import IUPACData

THREE_TO_ONE = IUPACData.protein_letters_3to1

def __parse_hgvs(hgvs_p):
    """
    Parse HGVS protein notation like ENSP00000275493.2:p.Arg2Gln.
    """
    if hgvs_p is None:
        return None, None, None
    match = re.match(r'.*?:p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', hgvs_p)
    if not match:
        return None, None, None

    wt_aa, protein_pos, mut_aa = match.groups()
    wt_aa = THREE_TO_ONE.get(wt_aa)
    mut_aa = THREE_TO_ONE.get(mut_aa)

    return wt_aa, protein_pos, mut_aa


def __annotate_variants_batch(variant_strings, max_retries=5):
    """Query Ensembl VEP REST API for a batch of variants."""
    server = "https://rest.ensembl.org"
    ext = "/vep/homo_sapiens/region?hgvs=1"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = json.dumps({"variants": variant_strings})

    for attempt in range(max_retries):
        r = requests.post(server + ext, headers=headers, data=payload)
        if r.ok:
            return r.json()
        elif r.status_code == 429:  
            wait = 2 ** attempt + random.random()
            print(f"Rate limit hit. Waiting {wait:.2f}s...")
            time.sleep(wait)
        else:
            print(f"Error {r.status_code}: {r.text}")
            r.raise_for_status()
            sys.exit()
    return []


def __process_batch(batch):
    """
    Process a batch of variant tuples and annotate with Ensembl VEP.
    """
    variant_strings = [b[0] for b in batch]
    results = __annotate_variants_batch(variant_strings)
    hgvsp = None
    wt_aa, protein_pos, mut_aa = None, None, None 

    records = []
    for res, (variant_query, gene, uniprot) in zip(results, batch):
        for tx in res.get("transcript_consequences", []):
            if "missense_variant" in tx.get("consequence_terms", []):
                hgvsp = tx.get("hgvsp")
                wt_aa, protein_pos, mut_aa = __parse_hgvs(hgvsp)
                break

        record = {
            "gene": gene,
            "uniprot": uniprot,
            "wt_aa": wt_aa,
            "protein_pos": protein_pos,
            "mut_aa": mut_aa,
            "pathogenic": 0
        }
        records.append(record)

    return records


def map_gene_symbols(genes_path):
    """
    Pair HGNC and UniprotID together from a csv
    """
    genes = pd.read_csv(genes_path)
    return dict(zip(genes['HGNC'], genes['UniprotID']))


def extract_missense_variants(clinvar_path, genes, batch_size=200, save_csv=True, outdir="../results"):
    """
    Extract missense variants for a target gene from a VCF file.
    Returns a pandas DataFrame with annotations.
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
            records.extend(__process_batch(batch))
            batch = []

    if batch:
        records.extend(__process_batch(batch))

    df = pd.DataFrame(records)

    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "clinvar_missense.csv")
        df.to_csv(outfile, index=False)
        print(f"Saved results to {outfile}")

    return df.reset_index(drop=True)


def extract_cosmic_variants(cosmic_path, genes, save_csv=True, outdir="../results"):
    """
    Extract missense driver variants for given target genes from COSMIC.
    Returns a pandas DataFrame with annotations.
    """
    df = pd.read_csv(cosmic_path, sep="\t")

    df = df[df["GENE_SYMBOL"].isin(genes.keys())]
    df = df[df["MUTATION_DESCRIPTION"].str.contains("missense", case=False, na=False)]
    df = df.dropna(subset=["HGVSP"])
    df[["wt_aa", "protein_pos", "mut_aa"]] = df["HGVSP"].apply(__parse_hgvs).apply(pd.Series)
    df = df[df["POSITIVE_SCREEN"] == "y"]
    df = df.groupby(["GENE_SYMBOL", "wt_aa", "protein_pos", "mut_aa"]).filter(lambda x: len(x) >= 3)

    df_out = pd.DataFrame({
        "gene": df["GENE_SYMBOL"],
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
    Merge ClinVar benign and COSMIC driver variants into a harmonized dataset.
    Conflict resolution: if the same variant is in both, keep COSMIC (pathogenic=1).
    """
    df = pd.concat([clinvar_df, cosmic_df], ignore_index=True)
    df = df.sort_values(by="pathogenic", ascending=True)
    df = df.drop_duplicates(
        subset=["gene", "uniprot", "wt_aa", "protein_pos", "mut_aa"],
        keep="last"
    )

    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "harmonized.csv")
        df.to_csv(outfile, index=False)
        print(f"Saved harmonized dataset to {outfile}")

    return df.reset_index(drop=True)
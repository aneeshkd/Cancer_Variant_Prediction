import re 
import json 
import requests
import sys 
import time 
import random
import pandas as pd 
from Bio.Data import IUPACData

THREE_TO_ONE = IUPACData.protein_letters_3to1

def parse_hgvs(hgvs_p):
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
    """
    Query Ensembl VEP REST API for a batch of variants
    """
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


def process_batch(batch):
    """
    Process a batch of variant tuples and annotate with Ensembl VEP
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
                wt_aa, protein_pos, mut_aa = parse_hgvs(hgvsp)
                break

        record = {
            "hgnc": gene,
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
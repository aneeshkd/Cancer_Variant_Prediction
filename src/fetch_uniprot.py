import os 
import requests 
import pandas as pd 

def fetch_uniprot_fasta(genes, outdir="data/fasta", overwrite=False):
    """
    Fetch UniProt FASTA by ID and save 
    """
    for hgnc, uniprot in genes.items():
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
        r = requests.get(url)
        outfile = os.path.join(outdir, f"{hgnc}_{uniprot}.fasta")

        if os.path.exists(outfile) and not overwrite:
            print(f"Skipping {uniprot}, already exists.")
            continue

        if r.status_code == 200:
            with open(outfile, "w") as f:
                f.write(r.text)
            print(f"Saved {uniprot} to {outfile}")
        else:
            print(f"Failed to fetch {uniprot} ({r.status_code})")
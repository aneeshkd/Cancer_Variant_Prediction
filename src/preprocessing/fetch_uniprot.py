import pandas as pd 
import os 
import requests 

def fetch_uniprot_fasta(harmonized_csv, outdir, overwrite=False):
    """
    Fetch UniProt FASTA by ID and save 
    """
    df = pd.read_csv(harmonized_csv)
    print("=====Downloading UniProt FASTA=====")

    for _, row in df.iterrows():
        hgnc = row['hgnc']
        uniprot = row['uniprot']
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
        r = requests.get(url)
        outfile = os.path.join(outdir, f"{hgnc}_{uniprot}.fasta")

        if os.path.exists(outfile) and not overwrite:
            continue

        if r.status_code == 200:
            with open(outfile, "w") as f:
                f.write(r.text)
            print(f"Saved {uniprot} to {outfile}")
        else:
            print(f"Failed to fetch {uniprot} ({r.status_code})")
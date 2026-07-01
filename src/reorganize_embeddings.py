"""
Convert the embedding tensors to more easily usable format,
so that the right embedding can be directly found with the key
"""

import sys
import torch
import pandas as pd
from utils import load_lookup_from_file
import utils

def reorganize(embeddings: torch.Tensor, lookup: dict[str, int], strings: pd.Series):
    out = torch.zeros((len(strings), embeddings.shape[1]), dtype=embeddings.dtype)
    for i, string in enumerate(strings):
        if string in lookup:
            index = lookup[string]
            out[i] = embeddings[index]
    return out

def reorganize_kinase_proteins(df: pd.DataFrame | None = None):
    embeddings = torch.load('./../data/proteins/representations.pt')
    if df is None:
        df = pd.read_csv('./../data/master.csv')
    strings = df['sequence']
    lookup = load_lookup_from_file('./../data/proteins/sequences.txt')
    return reorganize(embeddings, lookup, strings)

def reorganize_kinase_drugs(df: pd.DataFrame | None = None):
    embeddings = torch.load('./../data/drugs/schulman_representations.pt')
    if df is None:
        df = pd.read_csv('./../data/master.csv')
    strings = df['smiles']
    lookup = load_lookup_from_file('./../data/drugs/schulman_smiles_processed.txt')
    return reorganize(embeddings, lookup, strings)

def reorganize_kinase_prot_drug():
    df = pd.read_csv('./../data/master.csv')
    prot = reorganize_kinase_proteins(df)
    drug = reorganize_kinase_drugs(df)
    return prot, drug

def main(args: list[str]):
    embeddings = torch.load(utils.get_from_args(args, '--in', None))
    df = pd.read_csv(utils.get_from_args(args, '--df', None))
    strings = df[utils.get_from_args(args, '--column', None)]
    lookup = load_lookup_from_file(utils.get_from_args(args, '--lookup', None))
    embeddings = reorganize(embeddings, lookup, strings)
    torch.save(embeddings, utils.get_from_args(args, '--out', None))

if __name__ == '__main__':
    main(sys.argv)

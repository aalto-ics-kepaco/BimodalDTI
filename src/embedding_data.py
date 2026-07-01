import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils import load_lookup_from_file
from collections.abc import Sequence
from utils import key_file_to_key_list
from get_key_files import get_key_files

class EmbeddingDataset(Sequence):
    def __init__(
        self,
        keys: Sequence[int],
        embeddings: Sequence[(str, str | None, pd.Series | None)],
        labels: pd.Series
    ):
        self._embeddings = torch.cat([
            self._load_embeddings(keys, strings, embedding_file, lookup_file)
            for embedding_file, lookup_file, strings in embeddings
        ], dim=1)
        self._labels = self._build_label_tensor(keys, labels)

    def __len__(self):
        return self._embeddings.shape[0]

    def __getitem__(self, idx: int):
        return (self._embeddings[idx], self._labels[idx])
    
    def _load_embeddings(
        self,
        keys: Sequence[int],
        strings: pd.Series | None,
        embedding_file: str,
        lookup_file: str | None
    ):
        lookup = None if lookup_file is None else load_lookup_from_file(lookup_file)
        initial_embeddings = torch.load(embedding_file)
        embeddings = torch.zeros((len(keys), initial_embeddings.shape[1]))
        for i, key in enumerate(keys):
            index = key if lookup is None else lookup[strings.iloc[key]]
            embeddings[i] = initial_embeddings[index]
        return embeddings

    def _build_label_tensor(self, keys: Sequence[int], labels: pd.Series):
        t = torch.zeros(len(keys), dtype=torch.float32)
        for i, k in enumerate(keys):
            t[i] = labels[k]
        return t

def get_kinase_dataloader(keys: str | list[int], batch_size = 64, shuffle = True, use_gnn_embeddings = True):
    if type(keys) == str:
        keys = key_file_to_key_list(keys)

    drug_embedding_file = './../data/drugs/schulman_representations.pt'
    protein_embedding_file = './../data/proteins/representations.pt'
    gign_embedding_file = './../data/complex/11_9_gign_embeddings.pt'
    hetero_gign_embedding_file = './../data/complex/10_9_hetero_gign_embeddings.pt'

    drug_lookup_file = './../data/drugs/schulman_smiles_processed.txt'
    protein_lookup_file = './../data/proteins/sequences.txt'
    gign_lookup_file = None
    hetero_gign_lookup_file = None

    df = pd.read_csv('./../data/master.csv')

    labels = df['pchembl_value']

    drug_strings = df['smiles']
    protein_strings = df['sequence']
    gign_strings = None
    hetero_gign_strings = None

    embeddings = [
        (drug_embedding_file, drug_lookup_file, drug_strings),
        (protein_embedding_file, protein_lookup_file, protein_strings)
    ]

    if use_gnn_embeddings:
        #embeddings.insert(0, (gign_embedding_file, gign_lookup_file, gign_strings))
        embeddings.insert(0, (hetero_gign_embedding_file, hetero_gign_lookup_file, hetero_gign_strings))

    dataset = EmbeddingDataset(keys, embeddings, labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_kinase_train_dataloader(use_gnn_embeddings = True, split: int | None = None, new_drug = False):
    return get_kinase_dataloader(
        get_key_files(split, new_drug)[0],
        use_gnn_embeddings=use_gnn_embeddings
    )

def get_kinase_validation_dataloader(use_gnn_embeddings = True, split: int | None = None, new_drug = False):
    return get_kinase_dataloader(
        get_key_files(split, new_drug)[1],
        use_gnn_embeddings=use_gnn_embeddings,
        shuffle=False
    )

def get_kinase_test_dataloader(use_gnn_embeddings = True, split: int | None = None, new_drug = False):
    return get_kinase_dataloader(
        get_key_files(split, new_drug)[2],
        use_gnn_embeddings=use_gnn_embeddings,
        shuffle=False
    )

def get_kinase_dataloaders(use_gnn_embeddings = True, split: int | None = None, new_drug = False):
    train_loader = get_kinase_train_dataloader(use_gnn_embeddings, split, new_drug)
    validation_loader = get_kinase_validation_dataloader(use_gnn_embeddings, split, new_drug)
    return train_loader, validation_loader

def get_kinase_full_dataset_dataloader():
    train_keys = key_file_to_key_list('./../data/keys/train_keys_clean.txt')
    validation_keys = key_file_to_key_list('./../data/keys/validation_keys_clean.txt')
    test_keys = key_file_to_key_list('./../data/keys/test_keys_clean.txt')
    keys = train_keys + validation_keys + test_keys
    return get_kinase_dataloader(keys, batch_size=128, shuffle=False, use_gnn_embeddings=False)

def get_davis_dataloaders(use_gnn_embeddings = True):
    raise Exception('davis data is not supported')

def get_davis_dataloader(keys: str | list[int] | None = None, batch_size = 64, shuffle = False, use_gnn_embeddings = False):
    if keys is None:
        #keys = key_file_to_key_list('./../data/davis/davis_keys/train_keys.txt') + \
        #key_file_to_key_list('./../data/davis/davis_keys/validation_keys.txt') + \
        #key_file_to_key_list('./../data/davis/davis_keys/test_keys.txt')
        keys = key_file_to_key_list('./../data/davis/davis_keys/only_active/active_keys.txt')
    elif type(keys) == str:
        keys = key_file_to_key_list(keys)

    drug_embedding_file = './../data/davis/drug_representations.pt'
    protein_embedding_file = './../data/davis/protein_representations.pt'
    #gign_embedding_file = './../data/complex/11_9_gign_embeddings.pt'
    #hetero_gign_embedding_file = './../data/complex/10_9_hetero_gign_embeddings.pt'

    drug_lookup_file = './../data/davis/drugs.txt'
    protein_lookup_file = './../data/davis/proteins.txt'
    #gign_lookup_file = None
    #hetero_gign_lookup_file = None

    df = pd.read_csv('./../data/davis/davis_dataset_processed.csv')

    labels = df['Kd']

    drug_strings = df['SMILES']
    protein_strings = df['Sequence']
    #gign_strings = None
    #hetero_gign_strings = None

    embeddings = [
        (drug_embedding_file, drug_lookup_file, drug_strings),
        (protein_embedding_file, protein_lookup_file, protein_strings)
    ]

    if use_gnn_embeddings:
        #embeddings.insert(0, (gign_embedding_file, gign_lookup_file, gign_strings))
        #embeddings.insert(0, (hetero_gign_embedding_file, hetero_gign_lookup_file, hetero_gign_strings))
        raise Exception('gnn embeddings not supported for davis data')

    dataset = EmbeddingDataset(keys, embeddings, labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

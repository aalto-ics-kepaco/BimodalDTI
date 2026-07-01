import os
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from reorganize_embeddings import reorganize_kinase_prot_drug
from utils import key_file_to_key_list
from get_key_files import get_key_files

class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    def __init__(self, keys: list[int], data_dir: str, include_embeddings = False):
        self._pre_process(keys, data_dir, include_embeddings)

    def _pre_process(self, keys: list[int], data_dir: str, include_embeddings = False):
        if include_embeddings:
            prot, drug = reorganize_kinase_prot_drug()
        if not data_dir.endswith('/'):
            data_dir += '/'
        self._graphs: list[Data] = []
        for k in keys:
            data = torch.load(data_dir + str(k), weights_only=False)
            if include_embeddings:
                data.embedding = torch.cat((drug[k], prot[k])).unsqueeze(0)
            self._graphs.append(data)

    def __getitem__(self, idx):
        return self._graphs[idx]

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self._graphs)
    
def get_kinase_dataloader(keys: str | list[int], shuffle = True, batch_size = 128, include_embeddings = True):
    data_dir = './../../dataset-construction/kinase_binary_cheapnet/'
    if type(keys) == str:
        keys = key_file_to_key_list(keys)
    dataset = GraphDataset(keys, data_dir, include_embeddings)
    dataloader = PLIDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

def get_kinase_train_dataloader(include_embeddings = False, split: int | None = None, new_drug = False):
    return get_kinase_dataloader(
        get_key_files(split, new_drug)[0],
        include_embeddings=include_embeddings
    )

def get_kinase_validation_dataloader(include_embeddings = False, split: int | None = None, new_drug = False):
    return get_kinase_dataloader(
        get_key_files(split, new_drug)[1],
        include_embeddings=include_embeddings
    )

def get_kinase_test_dataloader(include_embeddings = False, split: int | None = None, new_drug = False):
    return get_kinase_dataloader(
        get_key_files(split, new_drug)[2],
        include_embeddings=include_embeddings,
        shuffle=False
    )

def get_kinase_dataloaders(include_embeddings = False, split: int | None = None, new_drug = False):
    train_loader = get_kinase_train_dataloader(include_embeddings=include_embeddings, split=split, new_drug=new_drug)
    validation_loader = get_kinase_validation_dataloader(include_embeddings=include_embeddings, split=split, new_drug=new_drug)
    return train_loader, validation_loader

def get_kinase_full_dataset_dataloader():
    train_keys = key_file_to_key_list('./../data/keys/train_keys_clean.txt')
    validation_keys = key_file_to_key_list('./../data/keys/validation_keys_clean.txt')
    test_keys = key_file_to_key_list('./../data/keys/test_keys_clean.txt')
    keys = train_keys + validation_keys + test_keys
    return get_kinase_dataloader(keys, shuffle=False, include_embeddings=False)

def get_davis_dataloaders(include_embeddings = False):
    raise Exception('davis data is not supported')

def get_davis_dataloader(keys: str | list[int] | None = None, shuffle = False, batch_size = 128, include_embeddings = False):
    data_dir = './../../dataset-construction/davis_binary_cheapnet/'
    if keys is None:
        #keys = key_file_to_key_list('./../data/davis/davis_keys/train_keys.txt') + \
        #key_file_to_key_list('./../data/davis/davis_keys/validation_keys.txt') + \
        #key_file_to_key_list('./../data/davis/davis_keys/test_keys.txt')
        keys = key_file_to_key_list('./../data/davis/davis_keys/only_active/active_keys.txt')
    elif type(keys) == str:
        keys = key_file_to_key_list(keys)
    dataset = GraphDataset(keys, data_dir, include_embeddings)
    dataloader = PLIDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

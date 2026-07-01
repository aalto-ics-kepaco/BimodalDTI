import sys
from collections.abc import Sequence
import os
import numpy as np
from numpy.typing import NDArray
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import pandas as pd
from get_key_files import get_key_files

import time

feature_vector_size = 35

class ResidueDataset(Sequence):
    def __init__(
            self,
            data_folder: str,
            file_names: list[str] | None = None,
            labels: None | dict[str, float] = None,
            rbf_settings: dict[str, float | int] = {
                'min_dist': 0.0,
                'max_dist': 6.0,
                'centroid_count': 9
            }
        ):
        self._rbf_centroids = np.linspace(
            rbf_settings['min_dist'], rbf_settings['max_dist'], rbf_settings['centroid_count']
        )
        self._rbf_multiplier = -(rbf_settings['centroid_count'] / (rbf_settings['max_dist'] - rbf_settings['min_dist']))**2
        self._data_folder = data_folder if data_folder.endswith('/') else data_folder + '/'
        self._entries = [
            self._process_entry(e, (None if labels is None else labels[e]))
            for e in (os.listdir(self._data_folder) if file_names is None else file_names)
            #for e in os.listdir(self._data_folder)[0:128]
        ]

    def __len__(self):
        return len(self._entries)
    
    def __getitem__(self, i):
        return self._entries[i]

    def _read_entry(self, entry: str):
        return np.load(self._data_folder + entry)

    def _unpack_data(self, data) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.uint16], NDArray[np.float32], NDArray[np.uint16], NDArray[np.float32], NDArray[np.uint16], NDArray[np.float32]]:
        ligand_atoms = np.unpackbits(data['ligand_atoms'], axis=1, count=feature_vector_size)
        residue_nodes = data['residue_nodes']
        lig_edges = data['lig_edges']
        lig_edge_features = data['lig_edge_features']
        prot_edges = data['prot_edges']
        prot_edge_features = data['prot_edge_features']
        lig_prot_edges = data['lig_prot_edges']
        lig_prot_edge_features = data['lig_prot_edge_features']
        return ligand_atoms, residue_nodes, lig_edges, lig_edge_features, prot_edges, \
            prot_edge_features, lig_prot_edges, lig_prot_edge_features
    
    def _add_reverse_edges(
            self,
            lig_edges: NDArray[np.uint16],
            lig_edge_features: NDArray[np.float32],
            prot_edges: NDArray[np.uint8],
            prot_edge_features: NDArray[np.float32],
            lig_prot_edges: NDArray[np.uint16]
        ):
        lig_edges = np.concatenate((lig_edges, np.flip(lig_edges, 0)), 1)
        lig_edge_features = np.concatenate((lig_edge_features, lig_edge_features), 0)
        prot_edges = np.concatenate((prot_edges, np.flip(prot_edges, 0)), 1)
        prot_edge_features = np.concatenate((prot_edge_features, prot_edge_features), 0)
        prot_lig_edges = np.flip(lig_prot_edges, 0)
        return lig_edges, lig_edge_features, prot_edges, prot_edge_features, lig_prot_edges, prot_lig_edges
    
    def _process_edge_features(self, edge_features: NDArray[np.float32]):
        if edge_features.shape[0] == 0:
            return torch.tensor(edge_features, dtype=torch.float32)
        return torch.tensor(
            np.exp(np.pow(edge_features - self._rbf_centroids, 2) * self._rbf_multiplier),
            dtype=torch.float32
        )
    
    def _build_data_object(
        self,
        ligand_atoms: NDArray[np.uint8],
        residue_nodes: NDArray[np.float32],
        lig_edges: NDArray[np.uint16],
        lig_edge_features: NDArray[np.float32],
        prot_edges: NDArray[np.uint16],
        prot_edge_features: NDArray[np.float32],
        lig_prot_edges: NDArray[np.uint16],
        prot_lig_edges: NDArray[np.uint16],
        lig_prot_edge_features: NDArray[np.float32],
        label: None | float = None
    ):
        data = HeteroData()
        data['atom'].x = torch.tensor(ligand_atoms, dtype=torch.float32)
        data['residue'].x = torch.tensor(residue_nodes, dtype=torch.float32)
        data['atom', 'bond', 'atom'].edge_index = torch.tensor(lig_edges, dtype=torch.int32)
        data['atom', 'bond', 'atom'].edge_attr = self._process_edge_features(lig_edge_features)
        data['residue', 'bond', 'residue'].edge_index = torch.tensor(prot_edges, dtype=torch.int32)
        data['residue', 'bond', 'residue'].edge_attr = self._process_edge_features(prot_edge_features)
        cross_edge_features = self._process_edge_features(lig_prot_edge_features)
        data['atom', 'near', 'residue'].edge_index = torch.tensor(lig_prot_edges, dtype=torch.int32)
        data['atom', 'near', 'residue'].edge_attr = cross_edge_features
        data['residue', 'near', 'atom'].edge_index = torch.tensor(prot_lig_edges.copy(), dtype=torch.int32)
        data['residue', 'near', 'atom'].edge_attr = cross_edge_features
        if label is not None:
            data.y = torch.tensor(label, dtype=torch.float32)
        return data
    
    def _process_entry(self, entry: str, label: None | float = None):
        ligand_atoms, residue_nodes, lig_edges, lig_edge_features, prot_edges, \
            prot_edge_features, lig_prot_edges, lig_prot_edge_features = \
            self._unpack_data(self._read_entry(entry))
        lig_edges, lig_edge_features, prot_edges, prot_edge_features, lig_prot_edges, prot_lig_edges = \
            self._add_reverse_edges(
                lig_edges, lig_edge_features, prot_edges, prot_edge_features, lig_prot_edges
            )
        data = self._build_data_object(
            ligand_atoms, residue_nodes, lig_edges, lig_edge_features, prot_edges,
            prot_edge_features, lig_prot_edges, prot_lig_edges, lig_prot_edge_features, label
        )
        return data

def get_data_loader(
        data_folder: str,
        keys: list[int],
        labels: dict[str, float] | None = None,
        batch_size = 128,
        shuffle = True
):
    file_names = [f'{k}.npz' for k in keys]
    dataset = ResidueDataset(data_folder, file_names, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_label_mapping_from_df(df: pd.DataFrame, label_column: str):
    mapping: dict[str, float] = dict()
    for i, label in enumerate(df[label_column]):
        mapping[f'{i}.npz'] = label
    return mapping

def get_kinase_label_mapping_from_df(df: pd.DataFrame):
    return get_label_mapping_from_df(df, 'pchembl_value')

def key_file_to_key_list(key_file: str):
    with open(key_file, 'r') as f:
        return [int(k) for k in f.read().split('\n')]

def get_davis_dataloaders():
    raise Exception('Davis data is not currently supported')

def get_kinase_dataloader(keys: list[int], labels: dict[str, float] | None = None, batch_size=128, shuffle=True):
    if labels is None:
        labels = get_kinase_label_mapping_from_df(pd.read_csv('./../data/master.csv'))
    folder = './../../dataset-construction/kinase_binary_residue/'
    return get_data_loader(folder, keys, labels, batch_size=batch_size, shuffle=shuffle)

def get_kinase_train_dataloader(split: int | None = None, new_drug = False):
    keys = key_file_to_key_list(get_key_files(split, new_drug)[0])
    return get_kinase_dataloader(keys, shuffle=False)

def get_kinase_validation_dataloader(split: int | None = None, new_drug = False):
    keys = key_file_to_key_list(get_key_files(split, new_drug)[1])
    return get_kinase_dataloader(keys, shuffle=False)

def get_kinase_test_dataloader(split: int | None = None, new_drug = False):
    keys = key_file_to_key_list(get_key_files(split, new_drug)[2])
    return get_kinase_dataloader(keys, shuffle=False)

def get_kinase_dataloaders(split: int | None = None, new_drug = False):
    labels = get_kinase_label_mapping_from_df(pd.read_csv('./../data/master.csv'))
    train_keys, validation_keys, _ = get_key_files(split, new_drug)
    train_keys = key_file_to_key_list(train_keys)
    validation_keys = key_file_to_key_list(validation_keys)
    folder = './../../dataset-construction/kinase_binary_residue/'
    train_loader = get_data_loader(folder, train_keys, labels, batch_size=128, shuffle=True)
    validation_loader = get_data_loader(folder, validation_keys, labels, batch_size=128, shuffle=False)
    return train_loader, validation_loader

def get_kinase_full_dataset_dataloader():
    labels = get_kinase_label_mapping_from_df(pd.read_csv('./../data/master.csv'))
    train_keys = key_file_to_key_list('./../data/keys/train_keys_clean.txt')
    validation_keys = key_file_to_key_list('./../data/keys/validation_keys_clean.txt')
    test_keys = key_file_to_key_list('./../data/keys/test_keys_clean.txt')
    keys = train_keys + validation_keys + test_keys
    folder = './../../dataset-construction/kinase_binary_residue/'
    dataloader = get_data_loader(folder, keys, labels, batch_size=128, shuffle=False)
    return dataloader

def get_davis_dataloader(keys: str | list[int] | None = None, labels: dict[str, float] | None = None, batch_size=128, shuffle=False):
    if keys is None:
        #keys = key_file_to_key_list('./../data/davis/davis_keys/train_keys.txt') + \
        #key_file_to_key_list('./../data/davis/davis_keys/validation_keys.txt') + \
        #key_file_to_key_list('./../data/davis/davis_keys/test_keys.txt')
        keys = key_file_to_key_list('./../data/davis/davis_keys/only_active/active_keys.txt')
    elif type(keys) == str:
        keys = key_file_to_key_list(keys)
    if labels is None:
        labels = get_label_mapping_from_df(pd.read_csv('./../data/davis/davis_dataset_processed.csv'), 'Kd')
    folder = './../../dataset-construction/davis_binary_residue/'
    return get_data_loader(folder, keys, labels, batch_size=batch_size, shuffle=shuffle)

def main(args: list[str]):
    start = time.time()
    dataset = ResidueDataset('./../../dataset-construction/kinase_binary_residue/')
    end = time.time()
    print('dataset creation:', end - start)
    start = time.time()
    min_connections = min(e['residue'].x.shape[0] for e in dataset)
    end = time.time()
    print('dataset iteration:', end - start)
    max_connections = max(e['residue'].x.shape[0] for e in dataset)
    print(min_connections, max_connections)
    print(dataset[0]['residue'].x)

if __name__ == '__main__':
    main(sys.argv)

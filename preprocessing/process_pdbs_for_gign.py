import pymol
from rdkit import Chem
import pickle
import utils
import os
import sys
import time
import traceback
import torch
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
import pandas as pd

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        results += one_of_k_encoding(atom.GetPDBResidueInfo().GetResidueName(), [
            'MET', 'LYS', 'ALA', 'PRO', 'VAL', 'LEU', 'GLY',
            'ILE', 'PHE', 'THR', 'GLN', 'ARG', 'SER', 'ASN',
            'GLU', 'CYS', 'TYR', 'HIS', 'ASP', 'TRP', 'LIG'
        ])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    #atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

def mols2graphs(ligand: Chem.Mol, pocket: Chem.Mol, label: float, dis_threshold=5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p = mol2graph(pocket)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
    y = torch.FloatTensor([label])
    pos = torch.concat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
    
    data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)

    #torch.save(data, save_path)
    return data

def sanitize_pdb(input_file: str, output_file: str):
    molecule: Chem.Mol = Chem.MolFromPDBFile(input_file, sanitize=False)
    Chem.SanitizeMol(molecule, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)
    with Chem.PDBWriter(output_file) as w:
        w.write(molecule)

def extract_protein_and_ligand(input_file: str, output_folder: str):
    complex = 'complex'
    pymol.cmd.load(input_file, complex)
    ligand = 'ligand'
    pymol.cmd.select(ligand, 'chain B')
    pymol.cmd.save(output_folder + 'ligand.pdb', ligand)
    protein = 'protein_pocket'
    pymol.cmd.select(protein, 'chain A within 5 of chain B')
    pymol.cmd.save(output_folder + 'protein.pdb', protein)
    pymol.cmd.delete('all')

def protein_and_ligand_to_binary(input_folder: str, output_file: str, label: float):
    ligand = Chem.MolFromPDBFile(input_folder + 'ligand.pdb', sanitize=False, removeHs=True)
    Chem.SanitizeMol(ligand, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)
    protein = Chem.MolFromPDBFile(input_folder + 'protein.pdb', sanitize=False, removeHs=True)
    Chem.SanitizeMol(protein, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)
    data = mols2graphs(ligand, protein, label)
    torch.save(data, output_file)
    #with open(output_file, 'wb') as f:
    #    pickle.dump((ligand, protein), f)

def delete_partial_pdbs(folder: str):
    os.remove(folder + 'ligand.pdb')
    os.remove(folder + 'protein.pdb')

def main(args: list[str]):
    df_file = utils.get_from_args(args, '--df', 'master.csv')
    labels = pd.read_csv(df_file)['pchembl_value']
    input_folder = utils.get_from_args(args, '--input-folder', './')
    output_folder = utils.get_from_args(args, '--output-folder', None)
    entries = os.listdir(input_folder)
    start, end = utils.get_start_and_end(args, len(entries))
    start_time = time.time()
    i = start
    while i < end:
        entry = entries[i]
        folder = f'{input_folder}{entry}/'
        try:
            sanitize_pdb(f'{folder}{entry}.pdb', f'{folder}{entry}_s.pdb')
            extract_protein_and_ligand(f'{folder}{entry}_s.pdb', folder)
            protein_and_ligand_to_binary(
                folder,
                (folder if output_folder is None else output_folder) + entry,
                labels[int(entry)]
            )
            delete_partial_pdbs(folder)
            os.remove(f'{folder}{entry}_s.pdb')
        except Exception:
            print(i, entry)
            traceback.print_exc()
        i += 1
        elapsed = time.time() - start_time
        print(f'{i}/{end} {elapsed:.2f} s', flush=True)
    pymol.cmd.quit()

if __name__ == '__main__':
    main(sys.argv)

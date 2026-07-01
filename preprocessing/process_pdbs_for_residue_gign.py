import sys
import os
import utils
from rdkit import Chem
import rdkit.Geometry.rdGeometry
from rdkit.Chem.rdchem import ChiralType, HybridizationType, BondType
import numpy as np
import time
import torch
import pickle
import pandas as pd
import traceback

cutoff = 5.0

feature_vector_size = 35

def read_file(file: str):
    molecule = Chem.MolFromPDBFile(file, sanitize=False)
    r = Chem.SanitizeMol(molecule, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)
    if r != 0:
        print('SanitizeMol failed', file, r)
    return molecule

def get_atoms(molecule: Chem.Mol) -> list[Chem.Atom]:
    return list(molecule.GetAtoms())

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):
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

    atom_feats = np.array(results).astype(np.float32)

    return atom_feats

def atom_to_vector(atom: Chem.Atom):
    return atom_features(atom)

def atoms_to_matrix(atoms: list[Chem.Atom]):
    matrix = np.zeros((len(atoms), feature_vector_size), dtype=np.uint8)
    for i, atom in enumerate(atoms):
        matrix[i] = atom_to_vector(atom)
    return matrix

def get_covalent_edges(molecule: Chem.Mol, atoms: list[Chem.Atom]):
    conformer = molecule.GetConformers()[0]
    atom_ids = [a.GetIdx() for a in atoms]
    atom_map = {id: index for index, id in enumerate(atom_ids)}
    edges: list[list[int]] = [[], []]
    edge_features: list[list[float]] = []
    for bond in molecule.GetBonds():
        a: int = bond.GetBeginAtomIdx()
        b: int = bond.GetEndAtomIdx()
        if a in atom_map and b in atom_map:
            edges[0].append(atom_map[a])
            edges[1].append(atom_map[b])
            edge_features.append([conformer.GetAtomPosition(a).Distance(conformer.GetAtomPosition(b))])
    return np.array(edges, dtype=np.uint16), np.array(edge_features, dtype=np.float32)

def get_ligand_atoms(atoms: list[Chem.Atom]):
    ligand_atoms: list[Chem.Atom] = []
    for atom in atoms:
        chain_id = atom.GetPDBResidueInfo().GetChainId()
        assert(chain_id == 'A' or chain_id == 'B')
        if chain_id == 'B':
            ligand_atoms.append(atom)
    return ligand_atoms

def group_residue_atoms(atoms: list[Chem.Atom], residue_count: int):
    residues: list[list[Chem.Atom]] = [[] for _ in range(residue_count)]
    for atom in atoms:
        info = atom.GetPDBResidueInfo()
        if info.GetChainId() == 'B':
            continue
        index = info.GetResidueNumber() - 1
        assert(index >= 0) # only the ligand atoms should end up with index 0, and we are skipping them
        residues[index].append(atom)
    return residues

def calc_residue_distance(a: list[rdkit.Geometry.rdGeometry.Point3D], b: list[rdkit.Geometry.rdGeometry.Point3D]):
    min_dist = np.inf
    for p1 in a:
        for p2 in b:
            dist = p1.Distance(p2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def extract_nodes(molecule: Chem.Mol, emb_tensor: np.typing.NDArray):
    residue_count = emb_tensor.shape[0]

    atoms = get_atoms(molecule)

    ligand_atoms = get_ligand_atoms(atoms)
    residues = group_residue_atoms(atoms, residue_count)

    conformers = molecule.GetConformers()
    assert(len(conformers) == 1)
    conformer = conformers[0]

    residue_nodes: list[np.typing.NDArray] = []
    # the coordinate list will be None for residues that are not included
    residue_coordinates: list[None | list[rdkit.Geometry.rdGeometry.Point3D]] = [None for _ in range(residue_count)]
    prot_edges: list[list[int]] = [[], []]
    prot_edge_features: list[list[float]] = []
    lig_prot_edges: list[list[int]] = [[], []]
    lig_prot_edge_features: list[list[float]] = []

    index = 0
    for i, residue in enumerate(residues):
        connected: dict[int, float] = dict()
        coordinates = []
        for atom in residue:
            position: rdkit.Geometry.rdGeometry.Point3D = conformer.GetAtomPosition(atom.GetIdx())
            coordinates.append(position)
            for j, other in enumerate(ligand_atoms):
                dist = position.Distance(conformer.GetAtomPosition(other.GetIdx()))
                if dist <= cutoff and (j not in connected or connected[j] > dist):
                    # the same ligand atom could be connected to multiple protein atoms in the same residue
                    # hence, it is better to first gather a set of connected atoms
                    connected[j] = dist
        if len(connected) > 0:
            residue_coordinates[i] = coordinates
            residue_nodes.append(emb_tensor[i])
            for j in connected:
                lig_prot_edges[0].append(j)
                lig_prot_edges[1].append(index)
                lig_prot_edge_features.append([connected[j]])
            if index > 0 and residue_coordinates[i - 1] is not None:
                prot_edges[0].append(index - 1)
                prot_edges[1].append(index)
                prot_edge_features.append([calc_residue_distance(residue_coordinates[i - 1], coordinates)])
            index += 1
    
    return (
        ligand_atoms,
        np.stack(residue_nodes, axis=0, dtype=np.float32),
        np.array(lig_prot_edges, dtype=np.uint16),
        np.array(lig_prot_edge_features, dtype=np.float32),
        np.array(prot_edges, dtype=np.uint16),
        np.array(prot_edge_features, dtype=np.float32)
    )

def molecule_to_numpy(molecule: Chem.Mol, emb_tensor: torch.Tensor):
    ligand_atoms, residue_nodes, lig_prot_edges, lig_prot_edge_features, prot_edges, prot_edge_features = extract_nodes(molecule, emb_tensor)
    lig_edges, lig_edge_features = get_covalent_edges(molecule, ligand_atoms)
    ligand_atoms = atoms_to_matrix(ligand_atoms)
    return ligand_atoms, residue_nodes, lig_edges, lig_edge_features, prot_edges, prot_edge_features, lig_prot_edges, lig_prot_edge_features

def save_numpy(
    file: str,
    ligand_atoms: np.typing.NDArray[np.uint8],
    residue_nodes: np.typing.NDArray[np.float32],
    lig_edges: np.typing.NDArray[np.uint16],
    lig_edge_features: np.typing.NDArray[np.float32],
    prot_edges: np.typing.NDArray[np.uint16],
    prot_edge_features: np.typing.NDArray[np.float32],
    lig_prot_edges: np.typing.NDArray[np.uint16],
    lig_prot_edge_features: np.typing.NDArray[np.float32]
):
    ligand_atoms = np.packbits(ligand_atoms, axis=1)
    np.savez(
        file,
        ligand_atoms=ligand_atoms,
        residue_nodes=residue_nodes,
        lig_edges=lig_edges,
        lig_edge_features=lig_edge_features,
        prot_edges=prot_edges,
        prot_edge_features=prot_edge_features,
        lig_prot_edges=lig_prot_edges,
        lig_prot_edge_features=lig_prot_edge_features
    )

def process_entry_risky(entry: str, input_folder: str, output_folder: str, emb_tensor: np.typing.NDArray):
    molecule = read_file(f'{input_folder}{entry}/{entry}.pdb')
    ligand_atoms, residue_nodes, lig_edges, lig_edge_features, prot_edges, prot_edge_features, lig_prot_edges, lig_prot_edge_features = molecule_to_numpy(molecule, emb_tensor)
    save_numpy(f'{output_folder}{entry}.npz', ligand_atoms, residue_nodes, lig_edges, lig_edge_features, prot_edges, prot_edge_features, lig_prot_edges, lig_prot_edge_features)

def process_entry(entry: str, input_folder: str, output_folder: str, emb_tensor: np.typing.NDArray):
    try:
        process_entry_risky(entry, input_folder, output_folder, emb_tensor)
    except Exception as err:
        print('failed to process an entry')
        print(entry)
        traceback.print_exc()
        print('******************************')

def load_residue_embeddings(file: str):
    with open(file, 'rb') as f:
        tensors: list[torch.Tensor] = pickle.load(f)
        return list(t.numpy() for t in tensors)
    
def load_file_lines_to_list(file: str):
    with open(file, 'r') as f:
        return list(map(lambda line: line.strip(), f.readlines()))

def load_lookup_from_file(file: str):
    lines = load_file_lines_to_list(file)
    return {value: index for index, value in enumerate(lines)}
    
def entry_to_embedding_tensor(entry: str, residue_embeddings: list[torch.Tensor], df: pd.DataFrame, sequence_lookup: dict[str, int]):
    sequence = df.loc[int(entry), 'sequence']
    return residue_embeddings[sequence_lookup[sequence]]

def main(args: list[str]):
    input_folder = utils.get_from_args(args, '--input-folder', './')
    output_folder = utils.get_from_args(args, '--output-folder', None)
    entries = os.listdir(input_folder)
    embedding_file = utils.get_from_args(args, '--residue-embeddings', './residue_representations')
    residue_embeddings = load_residue_embeddings(embedding_file)
    df_file = utils.get_from_args(args, '--df', './master.csv')
    df = pd.read_csv(df_file)
    sequence_file = utils.get_from_args(args, '--sequences', './sequences.txt')
    sequence_lookup = load_lookup_from_file(sequence_file)
    start, end = utils.get_start_and_end(args, len(entries))
    start_time = time.time()
    i = start
    while i < end:
        entry = entries[i]
        emb_tensor = entry_to_embedding_tensor(entry, residue_embeddings, df, sequence_lookup)
        process_entry(entry, input_folder, output_folder, emb_tensor)
        i += 1
        elapsed = time.time() - start_time
        print(f'{i}/{len(entries)} {elapsed:.2f} s', flush=True)

if __name__ == '__main__':
    main(sys.argv)

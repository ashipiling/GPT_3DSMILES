import warnings


from rdkit import Chem
import torch.nn.functional as F
import numpy as np
import torch
import math
import torch_geometric
from torch_geometric.data import Data
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import NeighborSearch, Selection

import copy, os, hashlib, time, random
import shutil
import logging
logging.getLogger('prody').setLevel(logging.WARNING)

from scripts.gen_pocket_smiles import gen_3dsmiles
from scripts.gen_pocket_smiles_3 import get_pocket_vertice, get_pocket_vertice_2
from scipy.spatial import distance_matrix


import warnings
warnings.simplefilter('ignore', BiopythonWarning)

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


def read_sdf(file):
    supp = Chem.SDMolSupplier(file)
    return [i for i in supp]


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance[key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if
                                                  instance.ligand_bond_index[0, k].item() == i] for i in
                                       instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        elif key == 'ligand_context_bond_index':
            return self['ligand_context_element'].size(0)

        elif key == 'mask_ctx_edge_index_0':
            return self['ligand_masked_element'].size(0)
        elif key == 'mask_ctx_edge_index_1':
            return self['ligand_context_element'].size(0)
        elif key == 'mask_compose_edge_index_0':
            return self['ligand_masked_element'].size(0)
        elif key == 'mask_compose_edge_index_1':
            return self['compose_pos'].size(0)

        elif key == 'compose_knn_edge_index':  # edges for message passing of encoder
            return self['compose_pos'].size(0)

        elif key == 'real_ctx_edge_index_0':
            return self['pos_real'].size(0)
        elif key == 'real_ctx_edge_index_1':
            return self['ligand_context_element'].size(0)
        elif key == 'real_compose_edge_index_0':  # edges for edge type prediction
            return self['pos_real'].size(0)
        elif key == 'real_compose_edge_index_1':
            return self['compose_pos'].size(0)

        elif key == 'real_compose_knn_edge_index_0':  # edges for message passing of  field
            return self['pos_real'].size(0)
        elif key == 'fake_compose_knn_edge_index_0':
            return self['pos_fake'].size(0)
        elif (key == 'real_compose_knn_edge_index_1') or (key == 'fake_compose_knn_edge_index_1'):
            return self['compose_pos'].size(0)

        elif (key == 'idx_protein_in_compose') or (key == 'idx_ligand_ctx_in_compose'):
            return self['compose_pos'].size(0)

        elif key == 'index_real_cps_edge_for_atten':
            return self['real_compose_edge_index_0'].size(0)
        elif key == 'tri_edge_index':
            return self['compose_pos'].size(0)

        elif key == 'idx_generated_in_ligand_masked':
            return self['ligand_masked_element'].size(0)
        elif key == 'idx_focal_in_compose':
            return self['compose_pos'].size(0)
        elif key == 'idx_protein_all_mask':
            return self['compose_pos'].size(0)
        else:
            return super().__inc__(key, value)


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def compute_backbone_dihedrals(X, eps=1e-7):
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def compute_backbone_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def compute_sidechains_vec(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


class ProteinGraphDataset_v2():

    def __init__(self, 
                 data_list,
                 num_positional_embeddings=16,
                 top_k=8, 
                 num_rbf=16, 
                 device="cpu"
            ):
        super(ProteinGraphDataset_v2, self).__init__()

        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e['seq']) for e in data_list]

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        self.residue_numbers = torch.tensor(np.array(list(self.letter_to_num.values())))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self._featurize_as_graph(self.data_list[i])

    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'],
                                     device=self.device, dtype=torch.float32)
            
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']], device=self.device, dtype=torch.long)
            len_seq = len(seq)
            seq = seq.view(-1, 1) == self.residue_numbers
            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf
            X_ca = coords[:, 1]
            dihedrals = compute_backbone_dihedrals(coords)
            orientations = compute_backbone_orientations(X_ca)
            sidechains = compute_sidechains_vec(coords)
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            is_mol_atom = torch.zeros([len_seq, 1], dtype=torch.long)

        node_s = torch.cat([node_s, seq, is_mol_atom], dim=-1)
        data = torch_geometric.data.Data(x=X_ca, 
                                         seq=seq, 
                                         name=name,
                                         node_s=node_s, 
                                         mask=mask,
                                         node_v=node_v, 
                                         res_seq=protein['seq'])
        return data


class ProteinGraphDataset_v3():

    def __init__(self, 
                 data_list,
                 num_positional_embeddings=16,
                 top_k=8, 
                 num_rbf=16, 
                 device="cpu"
            ):
        super(ProteinGraphDataset_v3, self).__init__()

        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e['seq']) for e in data_list]

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        self.residue_numbers = torch.tensor(np.array(list(self.letter_to_num.values())))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self._featurize_as_graph(self.data_list[i])

    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'],
                                     device=self.device, dtype=torch.float32)
            # repeat扩充coords纬度(n, 3) -> (n, 4, 3)
            coords = coords.unsqueeze(1).repeat(1, 4, 1)

            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']], device=self.device, dtype=torch.long)
            len_seq = len(seq)
            seq = seq.view(-1, 1) == self.residue_numbers
            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf
            X_ca = coords[:, 1]
            dihedrals = compute_backbone_dihedrals(coords)
            orientations = compute_backbone_orientations(X_ca)
            sidechains = compute_sidechains_vec(coords)
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            is_mol_atom = torch.zeros([len_seq, 1], dtype=torch.long)

        node_s = torch.cat([node_s, seq, is_mol_atom], dim=-1)
        data = torch_geometric.data.Data(x=X_ca, 
                                         seq=seq, 
                                         name=name,
                                         node_s=node_s, 
                                         mask=mask,
                                         node_v=node_v, 
                                         res_seq=protein['seq'])
        return data




def get_backbone_coords(res_list, coord_offset=np.array([0.,0.,0.]), rot_matrix=np.eye(3)):
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:     #Size=(4,3)
            coord = copy.deepcopy(atom.coord)
            coord -= coord_offset
            coord = np.dot(coord, rot_matrix)
            res_coords.append(list(coord))
        
        coords.append(res_coords)

    return coords


def get_backbone_coords_2(res_list, coord_offset=np.array([0.,0.,0.]), rot_matrix=np.eye(3)):
    coords = []
    for res in res_list:
        coord = copy.deepcopy(res)
        coord -= coord_offset
        coord = np.dot(coord, rot_matrix)

        coords.append(coord)

    return coords


def get_protein_feature_v2(res_list, coord_offset=np.array([0.,0.,0.]), rot_matrix=np.eye(3)):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = ''
    for res in res_list:
        if res.resname in three_to_one:
            structure['seq'] += three_to_one.get(res.resname)

    coords = get_backbone_coords(res_list, coord_offset=coord_offset, rot_matrix=rot_matrix)
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = ProteinGraphDataset_v2([structure])
    protein = dataset[0]
    x = {}
    x['pkt_node_xyz'] = protein.x
    x['pkt_node_vec'] = protein.node_v
    x['pkt_seq'] = protein.seq
    x['pkt_node_sca'] = protein.node_s
    x['res_seq'] = protein.res_seq

    return x


def get_protein_feature_v3(res_list, coord_offset=np.array([0.,0.,0.]), rot_matrix=np.eye(3)):
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = ''
    for res in res_list:
        structure['seq'] += three_to_one.get('ALA')

    coords = get_backbone_coords_2(res_list, coord_offset=coord_offset, rot_matrix=rot_matrix)
    structure['coords'] = coords
    torch.set_num_threads(1)
    dataset = ProteinGraphDataset_v3([structure])
    protein = dataset[0]
    x = {}
    x['pkt_node_xyz'] = protein.x
    x['pkt_node_vec'] = protein.node_v
    x['pkt_seq'] = protein.seq
    x['pkt_node_sca'] = protein.node_s
    x['res_seq'] = protein.res_seq

    return x


def pdb_to_pocket_data(pdb_file, bbox_size=10.0, mol_file=None, perturb=True, center=None):
    '''
    use the sdf_file as the center
    '''
    if mol_file is not None:
        prefix = mol_file.split('.')[-1]
        if prefix == 'mol2':
            mol = Chem.MolFromMol2File(mol_file, sanitize=False)
            mol = Chem.RemoveHs(mol)
            center = mol.GetConformer().GetPositions()
            center = np.array(center)
        elif prefix == 'sdf':
            supp = Chem.SDMolSupplier(mol_file, sanitize=False)
            mol = supp[0]
            mol = Chem.RemoveHs(mol)
            center = mol.GetConformer().GetPositions()
        else:
            print('The File type of Molecule is not support')
        smiles = Chem.MolToSmiles(mol)
    elif center is not None:
        center = [center]
        smiles = None
    else:
        print('You must specify the original ligand file or center')
        smiles = None
    center_xyz = np.mean(center, axis=0)
    if perturb:
        center_xyz = center_xyz + np.random.normal(0, 1, center_xyz.shape)
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('target', pdb_file)[0]
    atoms  = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atoms)
    close_residues= []
    dist_threshold = bbox_size
    # for a in center:
    #     close_residues.extend(ns.search(a, dist_threshold, level='R'))
    close_residues.extend(ns.search(center_xyz, dist_threshold, level='R'))
    close_residues = Selection.uniqueify(close_residues)
    protein_dict = get_protein_feature_v2(close_residues)
    
    return smiles, protein_dict


def pdb_to_pocket_data2(pdb_file, bbox_size=10.0, mol_file=None, perturb=True, center=None, id=0):
    '''
    use the sdf_file as the center
    '''
    if mol_file is not None:
        prefix = mol_file.split('.')[-1]
        if prefix == 'mol2':
            mol = Chem.MolFromMol2File(mol_file, sanitize=False)
            mol = Chem.RemoveHs(mol)
            center = mol.GetConformer().GetPositions()
            center = np.array(center)
        elif prefix == 'sdf':
            supp = Chem.SDMolSupplier(mol_file, sanitize=False)
            mol = supp[id]
            mol = Chem.RemoveHs(mol)
            center = mol.GetConformer().GetPositions()
        else:
            print('The File type of Molecule is not support')
    elif center is not None:
        center = center
    else:
        print('You must specify the original ligand file or center')
    center_xyz = np.mean(center, axis=0)
    if perturb:
        center_xyz = center_xyz + np.random.normal(0, 1, center_xyz.shape)
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('target', pdb_file)[0]
    atoms  = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atoms)
    close_residues= []
    dist_threshold = bbox_size

    close_residues.extend(ns.search(center_xyz, dist_threshold, level='R'))
    close_residues = Selection.uniqueify(close_residues)
    protein_dict = get_protein_feature_v2(close_residues)
    smiles = Chem.MolToSmiles(mol)
    return smiles, protein_dict


def pdb_to_pocket_data3(pocket_path, ligand_path, bbox_size=5, perturb=True, center=None, id=0):
    ligand_mol, property_str, ligand_str, coord_offset, rot_matrix = gen_3dsmiles(
            ligand_path, 
            if_train=True
        )
    if not ligand_mol:
        return None, None, None, None

    ligand_coords = ligand_mol.GetConformer().GetPositions()

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('', pocket_path)[0]
        atoms = Selection.unfold_entities(structure, 'A')

        ns = NeighborSearch(atoms)
        close_residues= []
        dist_threshold = bbox_size
        for coord in ligand_coords:
            close_residues.extend(ns.search(coord, dist_threshold, level='R'))
        
        close_residues = Selection.uniqueify(close_residues)

        protein_dict = get_protein_feature_v2(close_residues, coord_offset=coord_offset, rot_matrix=rot_matrix)
    except:
        return None, None, None, None
    
    smiles_3d = property_str + '|' + ligand_str

    return smiles_3d, protein_dict, coord_offset, rot_matrix


def pdb_to_pocket_data4(pocket_path, ligand_path, threading_id=0, bbox_size=5, perturb=True, center=None, id=0, if_gen=False):
    ligand_mol, property_str, ligand_str, coord_offset, rot_matrix = gen_3dsmiles(
            ligand_path, 
            if_train=False
        )
    if not ligand_mol:
        return None, None, None, None

    ligand_coords = ligand_mol.GetConformer().GetPositions()
    # 随机文件夹
    random_name = hashlib.md5((str(time.time()) + str(random.random())).encode('utf-8')).hexdigest()

    config = {
        'surface_msms_bin': '/home/luohao/molGen/msms/msms',
        'surface_save_dir': 'data/surface/' + random_name,
        'mesh_threshold': 2
    }

    os.makedirs(config['surface_save_dir'], exist_ok=True)
    
    vertice, errno = get_pocket_vertice_2(pocket_path, ligand_coords, dist_t=6, config=config)
    if errno:
        return None, None, None, None

    dist_mat = distance_matrix(ligand_coords, vertice)

    min_dist_list_set = []
    for dist_list in dist_mat:
        tmp = np.argsort(dist_list)
        if len(tmp) > 0 and not tmp[0] in min_dist_list_set:
            min_dist_list_set.append(tmp[0])
        if len(tmp) > 1 and not tmp[1] in min_dist_list_set:
            min_dist_list_set.append(tmp[1])
        if len(tmp) > 2 and not tmp[2] in min_dist_list_set:
            min_dist_list_set.append(tmp[2])
        if len(tmp) > 3 and not tmp[3] in min_dist_list_set:
            min_dist_list_set.append(tmp[3])
        if len(tmp) > 4 and not tmp[4] in min_dist_list_set:
            min_dist_list_set.append(tmp[4])
        if len(tmp) > 5 and not tmp[5] in min_dist_list_set:
            min_dist_list_set.append(tmp[5])

    final_index_list = min_dist_list_set[:64]
    final_coord_list = []
    for index in final_index_list:
        final_coord_list.append(list(vertice[index]))

    protein_dict = get_protein_feature_v3(final_coord_list, coord_offset=coord_offset, rot_matrix=rot_matrix)

    if errno:
        return None, None, None, None, None
    
    smiles_3d = property_str + '|' + ligand_str

    print(protein_dict, smiles_3d)

    os.remove(config['surface_save_dir'] + '/../' + random_name + '.vert')
    os.remove(config['surface_save_dir'] + '/../' + random_name + '.face')
    os.remove(config['surface_save_dir'] + '/../' + random_name + '.area')
    os.system('rm -rf ' + config['surface_save_dir'])

    min_dist_list_set_2 = []
    for dist_list in dist_mat:
        tmp = np.argsort(dist_list)
        if len(tmp) > 0 and not tmp[0] in min_dist_list_set_2:
            min_dist_list_set_2.append(tmp[0])
        if len(tmp) > 1 and not tmp[1] in min_dist_list_set_2:
            min_dist_list_set_2.append(tmp[1])
        if len(tmp) > 2 and not tmp[2] in min_dist_list_set_2 and ligand_mol.GetNumAtoms() < 26:
            min_dist_list_set_2.append(tmp[2])
    
    final_list = min_dist_list_set_2
    ver_list = []
    pocket_res_str = ''
    pocket_coord_str = ''
    for id in final_list:
        ver_list.append(vertice[id])

    for f_vert in ver_list:
        # pocket_res_str += 'X'
        pocket_coord_str += transform_coord_2(f_vert, coord_offset=coord_offset, rot_matrix=rot_matrix)

    ret_str = pocket_res_str + '&' + pocket_coord_str + '|1|'

    return smiles_3d, protein_dict, coord_offset, rot_matrix, ret_str


def transform_coord_2(coord, coord_offset=np.array([0,0,0]), rot_matrix=np.eye(3)):
    coord -= coord_offset
    coord = np.dot(coord, rot_matrix)
    coord *= 10
    coord = list(map(int, list(coord)))

    # ret_str = '{'
    # for c in coord:
    #     ret_str += str(c) + ','
    ret_str = '{' + str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2])
    return ret_str



if __name__ == '__main__':
    pdb_file = '/mnt/e/tangui/SMILES_NEW/data/PDBbind/10gs/10gs_protein.pdb'
    ligand_path   = '/mnt/e/tangui/SMILES_NEW/data/PDBbind/10gs/10gs_ligand.sdf'
    cluster_threshold = 0.4
    bbox_size = 20

    pdb_to_pocket_data(pdb_file, 20, ligand_path)

# 该脚本将crossdocked数据集中的蛋白质口袋和配体提取出来，生成3dsmiles
import os, copy
from multiprocessing import Pool
import traceback, pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit import RDLogger
import torch
from prody import parsePDB, writePDB
import logging
from Bio import BiopythonWarning
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import NeighborSearch, Selection, PDBIO
import random

logging.getLogger('prody').setLevel(logging.WARNING)


import warnings

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


MAX_ATOM_NUM = 48
DIST_THRESHOLD = 5


def check_atom_symbol(mol):
    atom_symbol_list = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B']
    # 去除 H
    try:
        mol = Chem.RemoveHs(mol)
    except:
        print('RemoveHs failed')
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in atom_symbol_list:
            print('smiles中包含了不支持的元素: {}'.format(atom.GetSymbol()))
            return False
    if mol.GetNumAtoms() > MAX_ATOM_NUM:
        # print('smiles中原子数超过了最大限制: {}'.format(mol.GetNumAtoms()))
        return False
    return True


def calc_qed(mol):
    qed_v = 0.0
    try:
        qed_v = QED.qed(mol)
    except:
        pass
    if qed_v >= 0.6:
        return 1
    return 0


def calc_logp(mol):
    logp = 5
    try:
        logp = Chem.Crippen.MolLogP(mol)
    except:
        pass

    if logp <= 3 and logp >= -1:
        return 1
    return 0


def encode_number(num):
    if num > 199.9 or num < -199.9:
        print('coords 太大')
        return -1, None

    num = int(round(int(num * 10), 0))
    num_str = encode(num)

    return 0, num_str


def encode(inp):
    num = int(inp)
    prefix = ''
    if abs(num) != num:
        prefix = '-'
    num_str = prefix + '0' * (4 - len(str(abs(num)))) + str(abs(num))
    return num_str


def fix_coords(mol):
    cp_mol = copy.deepcopy(mol)
    smi = Chem.MolToSmiles(cp_mol, isomericSmiles=True)
    core = Chem.MolFromSmarts(smi)
    match = cp_mol.GetSubstructMatch(core)

    if len(match) != cp_mol.GetNumAtoms():
        print('match长度不一致', match)
        return -1, None

    coords = copy.deepcopy(mol).GetConformer().GetPositions()
    ret_coords = []
    # 根据align_dict对coords进行重排
    for index_cp in match:
        ret_coords.append(list(coords[index_cp]))
    
    return 0, np.array(ret_coords)


def coord_2_str(coords):
    three_dimension_smiles = ''
    for idx in range(len(coords)):
        ret, x = encode_number(coords[idx][0])
        if ret:
            return None
        ret, y = encode_number(coords[idx][1])
        if ret:
            return None
        ret, z = encode_number(coords[idx][2])
        if ret:
            return None

        three_dimension_smiles += '{' + str(x) + ',' + str(y) + ',' + str(z)

    return three_dimension_smiles


def out_3dsmiles(mol, if_train=False):
    
    err_no, coords = fix_coords(mol)
    if err_no:
        return None, None, None

    # 找到坐标的中心点
    coord_offset = np.array([0., 0., 0.])
    for c in coords:
        coord_offset += c
    coord_offset /= len(coords)

    # 随机在xyz轴上平移2a以内坐标
    if random.random() > 0.8:
        coord_offset += np.random.rand(3) * 22 - 11

    coords -= coord_offset

    # 随机旋转坐标
    # 旋转轴
    rot_matrix = np.eye(3)
    if if_train:
        axis = np.random.rand(3)
        axis /= np.sqrt(np.sum(axis * axis))
        # 旋转角度
        angle = np.random.rand() * 2 * np.pi
        # 旋转矩阵
        rot_matrix = np.array([
            [np.cos(angle) + axis[0] * axis[0] * (1 - np.cos(angle)),
            axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
            axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
            [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
            np.cos(angle) + axis[1] * axis[1] * (1 - np.cos(angle)),
            axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
            [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
            axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
            np.cos(angle) + axis[2] * axis[2] * (1 - np.cos(angle))]
        ])
        coords = np.dot(coords, rot_matrix)


    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    # 生成3dsmiles
    three_dimension_smiles = canonical_smiles

    three_dimension_smiles += '&'

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        ret, x = encode_number(coords[idx][0])
        if ret:
            return None, None, None
        ret, y = encode_number(coords[idx][1])
        if ret:
            return None, None, None
        ret, z = encode_number(coords[idx][2])
        if ret:
            return None, None, None

        three_dimension_smiles += '{' + str(x) + ',' + str(y) + ',' + str(z)

    return three_dimension_smiles, coord_offset, rot_matrix


def gen_3dsmiles(ligand_path, if_train=True):
    # smiles 过滤
    # 如果包含C O N S P F Cl Br I以外的元素，过滤掉
    # return smiles3d
    mol = Chem.SDMolSupplier(ligand_path)[0]
    if not mol:
        print('read sdf failed: {}'.format(ligand_path))
        return None, None, None, None, None

    if check_atom_symbol(mol):
        mol = Chem.RemoveHs(mol)
        # Chem.RemoveStereochemistry(mol)
        qed_v = calc_qed(mol)
        logp_v = calc_logp(mol)
        property_str = 'qed'+ str(qed_v) + '|' + 'logp' + str(logp_v)
        tdsmiles, coord_offset, rot_matrix = out_3dsmiles(mol, if_train=if_train)
        if not tdsmiles:
            return None, None, None, None, None

        return mol, property_str, tdsmiles, coord_offset, rot_matrix
    else:
        return None, None, None, None, None


def get_backbone_coords(res_list):
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:     #Size=(4,3)
            res_coords.append(list(atom.coord))
        coords.append(list(np.mean(np.array(res_coords), 0)))
    return coords


def transform_coords(coords, coord_offset=np.array([0,0,0]), rot_matrix=np.eye(3)):
    ret_list = []
    for i in coords:
        ret = transform_coord(i, coord_offset, rot_matrix)
        ret_list.append(ret)

    return ret_list


def transform_coord(coord, coord_offset=np.array([0,0,0]), rot_matrix=np.eye(3)):
    coord -= coord_offset
    coord = np.dot(coord, rot_matrix)
    coord = list(coord)

    return coord


def extract_pocket_mol(pocket_path, ligand_mol=None):
    # 读取pdb文件
    ligand_coords = ligand_mol.GetConformer().GetPositions()

    structure = parsePDB(pocket_path)
    protein = structure.select('protein')  # remove water and other useless
    selected = protein.select('same residue as within %s of ligand' % 8, ligand=ligand_coords)
    writePDB('data/tmp.pdb', selected)
    pocket_mol = Chem.MolFromPDBFile('data/tmp.pdb')
    if not pocket_mol:
        return None
    print('extract pocket mol success')

    return pocket_mol


def get_pocket_seq_and_coords(pocket_path, ligand_path, coord_offset, rot_matrix, box_size=5.0):
    supp = Chem.SDMolSupplier(ligand_path, sanitize=False)
    center = supp[0].GetConformer().GetPositions()
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('target', pocket_path)[0]
    atoms = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atoms)
    close_residues = []
    dist_threshold = box_size
    for a in center:
        close_residues.extend(ns.search(a, dist_threshold, level='R'))

    if len(close_residues) < 8:
        print('The structures studied are not well defined, maybe the box center is invalid.')
        return None, None
    
    close_residues = Selection.uniqueify(close_residues)
    
    res_list = [res for res in close_residues if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    seq = []
    res_coords = []
    for res in res_list:
        for atom in res.get_atoms():
            seq.append(res.resname)
            res_coords.append(list(atom.coord))
    # seq = [res.resname for res in res_list]
    # coords = get_backbone_coords(res_list)
    ret_coords = transform_coords(res_coords, coord_offset, rot_matrix)

    return seq, ret_coords


def gen_3dsmiles_list(pocket_ligand_dict):
    pocket_ligand_dict_list = pocket_ligand_dict['lst']

    smiles_3d_list = []
    for index, pocket_ligand_dict in enumerate(pocket_ligand_dict_list):
        pocket_path = pocket_ligand_dict['pocket_path']
        ligand_path = pocket_ligand_dict['ligand_path']

        try:
            mol = Chem.SDMolSupplier(ligand_path)[0]
            if not mol:
                print('read sdf failed: {}'.format(ligand_path))
                continue

            for dist_t in range(6):
                res_dict = {}
                ligand_mol, property_str, ligand_str, coord_offset, rot_matrix = gen_3dsmiles(
                    ligand_path, 
                    if_train=True
                )
                
                if not ligand_mol or not property_str or not ligand_str:
                    continue

                res_seq, res_coords = get_pocket_seq_and_coords(pocket_path, ligand_path, coord_offset, rot_matrix)
                res_coords_str = coord_2_str(res_coords)
                if res_seq is None:
                    continue

                res_dict['smiles'] = '{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000' + '|' + property_str + '|' + ligand_str
                # res_dict['res_seq'] = res_seq
                # res_dict['res_coords'] = res_coords
                res_dict['res_coords_str'] = res_coords_str

                smiles_3d_list.append(res_dict)
        except:
            pass
              
        if index % 1000 == 0:
            print('已处理{}个smiles'.format(index))
    return smiles_3d_list


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    res = []
    for i in range(n):
        l = lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        res.append({'lst': l, 'process_id': i})
    return res


def main(pdb_split_path, out_3dsmiles_path, n_worker=48):
    path_dict_list = []
    path_dict_list = pickle.load(open(pdb_split_path, 'rb'))

    print('path_dict_list', len(path_dict_list))
    path_dict_train_list = path_dict_list[: -100]
    # path_dict_train_list = path_dict_train_list[:2]

    # 启用多进程
    with Pool(processes=n_worker) as pool:
        results_train = pool.map(gen_3dsmiles_list, split_list(path_dict_train_list, n_worker))
    
    smiles_3d_train_list = [item for sublist in results_train for item in sublist]

    path_dict_val_list = path_dict_list[-100: ]
    # path_dict_val_list = path_dict_list[: 2]
    # 启用多进程
    with Pool(processes=n_worker) as pool:
        results_val = pool.map(gen_3dsmiles_list, split_list(path_dict_val_list, n_worker))
    
    smiles_3d_val_list = [item for sublist in results_val for item in sublist]

    ret_dict = {
        'train': smiles_3d_train_list,
        'test': smiles_3d_val_list
    }

    pickle.dump(ret_dict, open(out_3dsmiles_path, 'wb'))


if __name__ == '__main__':
    pdb_split_path = 'data/dock_results_list.pkl'
    out_3dsmiles_path = 'data/pocketsmiles_atom.pickle'
    print('start', out_3dsmiles_path)

    main(pdb_split_path, out_3dsmiles_path)


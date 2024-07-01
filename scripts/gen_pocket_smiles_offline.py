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

logging.getLogger('prody').setLevel(logging.WARNING)

from scipy.spatial import distance_matrix
try:
    from surface import Surface
except:
    from .surface import Surface

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)



MAX_ATOM_NUM = 38
TRY_TIME = 2  # 4
DIST_THRESHOLD = 5

AA_DICT = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

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
    if num >= 192.2 or num <= -192.2:
        return -1, None
    # num = -192.2~192.2
    num = int(num * 10)
    return 0, num


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


def calc_coord_delta(coord1, coord2):
    return sum([abs(coord1[i] - coord2[i]) for i in range(3)])


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
    if if_train:
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
        Chem.RemoveStereochemistry(mol)
        qed_v = calc_qed(mol)
        logp_v = calc_logp(mol)
        property_str = str(qed_v) + '|' + str(logp_v)
        tdsmiles, coord_offset, rot_matrix = out_3dsmiles(mol, if_train=if_train)
        if not tdsmiles:
            return None, None, None, None, None

        return mol, property_str, tdsmiles, coord_offset, rot_matrix
    else:
        return None, None, None, None, None


def transform_coord(coord, coord_offset=np.array([0,0,0]), rot_matrix=np.eye(3)):
    coord -= coord_offset
    coord = np.dot(coord, rot_matrix)
    coord = list(coord)

    return coord


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


def calc_dist(coord1, coord2):
    # 计算两个坐标的直线距离
    tmp = coord1 - coord2
    return np.sqrt(np.sum(tmp * tmp))


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


def gen_pocket_str(
        vertice, 
        ligand_mol=None, 
        coord_offset=np.array([0.,0.,0.]), 
        dist_t=4.5,
        rot_matrix=np.eye(3)
    ):
    ret_str = ''
    pocket_res_str = ''
    pocket_coord_str = ''

    ligand_coords = ligand_mol.GetConformer().GetPositions()

    dist_mat = distance_matrix(ligand_coords, vertice)
    # 找到最近的64个原子

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
        
    final_list = min_dist_list_set[:64]

    ret_coord_list = []
    for index in final_list:
        ret_coord_list.append(transform_coord(vertice[index], coord_offset=coord_offset, rot_matrix=rot_matrix))


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
    for id in final_list:
        ver_list.append(vertice[id])

    for f_vert in ver_list:
        # pocket_res_str += 'X'
        pocket_coord_str += transform_coord_2(f_vert, coord_offset=coord_offset, rot_matrix=rot_matrix)

    ret_str = pocket_res_str + '&' + pocket_coord_str

    return ret_str, ret_coord_list


def get_pocket_vertice_2(pocket_path, ligand_coords, dist_t=8, config={}):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('', pocket_path)[0]
        atoms = Selection.unfold_entities(structure, 'A')

        ns = NeighborSearch(atoms)
        close_residues= []

        new_structure = PDB.Structure.Structure('new')
        new_model = PDB.Model.Model(new_structure)
        new_chain = PDB.Chain.Chain('A')
        new_model.add(new_chain)  
        new_structure.add(new_model)

        for coord in ligand_coords:
            close_residues.extend(ns.search(coord, 6, level='R'))
        
        close_residues = Selection.uniqueify(close_residues)
        for residue in close_residues:
            try:
                new_chain.add(residue.copy())
            except:
                pass

        io = PDBIO()
        io.set_structure(new_structure)
        ori_pocket_path = os.path.join(config['surface_save_dir'], 'ori_pocket.pdb')
        io.save(ori_pocket_path)

        sur = Surface(config)
        ret_dict = sur.calc_pocket_vertice(ori_pocket_path)
        if ret_dict['error_no']:
            return None, -1
        
        return ret_dict['vertices'], 0
    
    except Exception as e:
        return None, -1


def get_pocket_vertice(pocket_path, ligand_coords, dist_t=8, config={}):
    try:
        structure = parsePDB(pocket_path)
        protein = structure.select('protein')  # remove water and other useless
        selected = protein.select('within %s of ligand' % dist_t, ligand=ligand_coords)
        if len(selected) == 0:
            print('no pocket')
            return None, -1
        ori_pocket_path = os.path.join(config['surface_save_dir'], 'ori_pocket.pdb')
        writePDB(ori_pocket_path, selected)
        sur = Surface(config)
        ret_dict = sur.calc_pocket_vertice(ori_pocket_path)
        if ret_dict['error_no']:
            return None, -1
        
        return ret_dict['vertices'], 0
    except Exception as e:
        return None, -1


def gen_3dsmiles_list(pocket_ligand_dict):
    pocket_ligand_dict_list = pocket_ligand_dict['lst']
    process_id = pocket_ligand_dict['process_id']
    config = {
        'surface_msms_bin': '/home/luohao/molGen/msms/msms',
        'surface_save_dir': 'data/surface/' + str(process_id),
        'mesh_threshold': 2
    }
    if not os.path.exists(config['surface_save_dir']):
        os.makedirs(config['surface_save_dir'])
    smiles_3d_list = []
    for index, pocket_ligand_dict in enumerate(pocket_ligand_dict_list):
        pocket_path = pocket_ligand_dict['pocket_path']
        ligand_path = pocket_ligand_dict['ligand_path']
        xscore = pocket_ligand_dict['xscore']
        if xscore < -8:
            xscore_flag = 1
        else:
            xscore_flag = 0

        mol = Chem.SDMolSupplier(ligand_path)[0]
        if not mol:
            print('read sdf failed: {}'.format(ligand_path))
            continue

        # 获取蛋白表面点
        vertice, errno = get_pocket_vertice(pocket_path, mol.GetConformer().GetPositions(), config=config)
        if errno:
            continue

        for dist_t in range(6):
            res_dict = {}
            ligand_mol, property_str, ligand_str, coord_offset, rot_matrix = gen_3dsmiles(
                ligand_path, 
                if_train=True
            )
            
            if not ligand_mol or not property_str or not ligand_str:
                continue

            pocket_str, pocket_vertice = gen_pocket_str(
                vertice, 
                ligand_mol=ligand_mol, 
                coord_offset=coord_offset, 
                dist_t=dist_t,
                rot_matrix=rot_matrix
            )

            res_dict['smiles_3d'] = pocket_str + '|' + str(xscore_flag) + '|' + property_str + '|' + ligand_str
            res_dict['res_coords'] = pocket_vertice
            res_dict['res_seq'] = 'A' * len(pocket_vertice)

            smiles_3d_list.append(res_dict)
              
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


def main(pdb_split_path, out_3dsmiles_path, n_worker=32):
    path_dict_list = []
    path_dict_list = pickle.load(open(pdb_split_path, 'rb'))

    print('path_dict_list', len(path_dict_list))
    path_dict_train_list = path_dict_list[: -100]

    # 启用多进程
    with Pool(processes=n_worker) as pool:
        results_train = pool.map(gen_3dsmiles_list, split_list(path_dict_train_list, n_worker))
    
    smiles_3d_train_list = [item for sublist in results_train for item in sublist]

    path_dict_val_list = path_dict_list[-100: ]
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
    out_3dsmiles_path = 'data/pocketsmiles_crossdocked_with_property_offline_1.pickle'

    main(pdb_split_path, out_3dsmiles_path)


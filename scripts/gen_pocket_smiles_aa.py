# 该脚本将crossdocked数据集中的蛋白质口袋和配体提取出来，生成3dsmiles
import os, copy, math
from multiprocessing import Pool
import traceback
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit import RDLogger
import torch
from prody import parsePDB, writePDB
import logging
import pickle
from Bio import BiopythonWarning
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import NeighborSearch, Selection, PDBIO
import random
from scipy.spatial import distance
import copy
from functools import partial


logging.getLogger('prody').setLevel(logging.WARNING)

from scipy.spatial import distance_matrix
try:
    from surface import Surface
except:
    from .surface import Surface


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)



MAX_ATOM_NUM = 48  # 38
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
            # print('smiles中包含了不支持的元素: {}'.format(atom.GetSymbol()))
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
    ligand_coords_copy = copy.deepcopy(coords)
    if err_no:
        return None, None, None, None

    # 找到坐标的中心点
    coord_offset = np.array([0., 0., 0.])
    for c in coords:
        coord_offset += c
    coord_offset /= len(coords)

    # 随机在xyz轴上平移2a以内坐标
    # if if_train:
    if random.random() > 0.2:
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
            return None, None, None, None
        ret, y = encode_number(coords[idx][1])
        if ret:
            return None, None, None, None
        ret, z = encode_number(coords[idx][2])
        if ret:
            return None, None, None, None

        three_dimension_smiles += '{' + str(x) + ',' + str(y) + ',' + str(z)

    return three_dimension_smiles, coord_offset, rot_matrix, ligand_coords_copy


def gen_3dsmiles(ligand_path, if_train=True):
    # smiles 过滤
    # 如果包含C O N S P F Cl Br I以外的元素，过滤掉
    # return smiles3d
    if type(ligand_path) is str:
        mol = Chem.SDMolSupplier(ligand_path)[0]
    else:
        mol = ligand_path

    if not mol:
        print('read sdf failed: {}'.format(ligand_path))
        return None, None, None, None, None, None

    if check_atom_symbol(mol):
        mol = Chem.RemoveHs(mol)
        # Chem.RemoveStereochemistry(mol)
        qed_v = calc_qed(mol)
        logp_v = calc_logp(mol)
        property_str = 'qed' + str(qed_v) + '|' + 'logp' + str(logp_v)
        tdsmiles, coord_offset, rot_matrix, ligand_coords_copy = out_3dsmiles(mol, if_train=if_train)
        if not tdsmiles:
            print('not tdsmiles')
            return None, None, None, None, None, None

        return mol, property_str, tdsmiles, coord_offset, rot_matrix, ligand_coords_copy
    else:
        # print('check_atom_symbol failed')
        return None, None, None, None, None, None


def transform_coord(coord, coord_offset=np.array([0,0,0]), rot_matrix=np.eye(3)):
    coord -= coord_offset
    coord = np.dot(coord, rot_matrix)
    coord *= 10
    f_coord = [int(round(i, 0)) for i in coord]

    if max(f_coord) >= 2000 or min(f_coord) <= -2000:
        print('pocket coord too big!!!!!')
        print(f_coord)
        return -1, None
    # ret_str = '{'
    # for c in coord:
    #     ret_str += str(c) + ','
    ret_str = '{' + encode(f_coord[0]) + ',' + encode(f_coord[1]) + ',' + encode(f_coord[2])
    return 0, ret_str


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
    if selected is None or len(selected) == 0:
        print('selected is None')
        return None
    writePDB('data/tmp.pdb', selected)
    pocket_mol = Chem.MolFromPDBFile('data/tmp.pdb')
    if not pocket_mol:
        return None
    print('extract pocket mol success')

    return pocket_mol


def save_vertice(ligand_coords, vertice, threshold=5):
    # 计算ligand_coords的重心
    ligand_center = np.mean(ligand_coords, axis=0)
    # 将vertice中的点与ligand_center的距离进行比较，去除掉与ligand_center距离大于threshold的点，最大的排在最前面
    vertice = sorted(vertice, key=lambda x: np.linalg.norm(x - ligand_center), reverse=True)
    if len(vertice) > 100:
        vertice = vertice[(len(vertice) -  100) : ]
    ret = np.array([list(i) for i in vertice])
    return ret


def encode_number(num, size_factor=10):
    if num > 199.9 or num < -199.9:
        print('coords 太大', num)
        return -1, None

    num = int(round(int(num * size_factor), 0))
    num_str = encode(num)

    return 0, num_str


def encode(inp):
    num = int(inp)
    prefix = ''
    if abs(num) != num:
        prefix = '-'
    num_str = prefix + '0' * (4 - len(str(abs(num)))) + str(abs(num))
    return num_str


def calc_min_dist(ligand_coords, vertice):
    dist = distance_matrix(ligand_coords, vertice)
    # 找到ligand_coords中每个点到vertice中最近的点的距离
    min_dist = np.min(dist, axis=1)
    return min_dist


def transform_coord_dist(dist_list):
    ret_dist_str = ''
    for dist in dist_list:
        errno, dist_str = encode_number(dist, size_factor=100)
        ret_dist_str += dist_str + ','
    return ret_dist_str[: -1]


def gen_pocket_str(
        vertice, 
        ligand_coords,
        ligand_mol=None, 
        coord_offset=np.array([0.,0.,0.]), 
        dist_t=4.5,
        rot_matrix=np.eye(3),
        use_expand=False,
        ori_xscore=-7,
        use_msms=True
    ):

    ret_str = ''

    pocket_res_str = ''
    pocket_coord_str = ''

    ver_list = save_vertice(ligand_coords, vertice)
    # print('ver_list', len(ver_list))
    min_list = calc_min_dist(ligand_coords, ver_list)

    for f_vert in ver_list:
        # pocket_res_str += 'X'
        ret_no, coord_str = transform_coord(f_vert, coord_offset=coord_offset, rot_matrix=rot_matrix)
        if ret_no == 0:
            pocket_coord_str += coord_str
        else:
            return -1, None, None
    
    pocket_dist_str = transform_coord_dist(min_list)

    ret_str = pocket_res_str + '&' + pocket_coord_str

    return 0, ret_str, pocket_dist_str


def find_nearest_coord(vertice, ligand_coords):
    min_dist = 1e8
    min_coord = ligand_coords[0]
    for coord in ligand_coords:
        dist = calc_dist(vertice, coord)
        if dist < min_dist:
            min_dist = dist
            min_coord = coord

    return min_coord



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


def get_pocket_aa_vertice(pocket_path, ligand_coords, dist_t=10, config={}, cube_size=1):
    try:
        structure = parsePDB(pocket_path)
        protein = structure.select('protein')  # remove water and other useless
        selected = protein.select('within %s of ligand' % dist_t, ligand=ligand_coords)
        if len(selected) == 0:
            print('no pocket')
            return None, -1
        ori_pocket_path = os.path.join(config['surface_save_dir'], 'ori_pocket.pdb')
        writePDB(ori_pocket_path, selected)
        pocket_mol = Chem.MolFromPDBFile(ori_pocket_path)
        return pocket_mol.GetConformer().GetPositions(), 0
    except Exception as e:
        return None, -1


def gen_3dsmiles_list(pocket_ligand_dict, add_xscore=True, use_msms=True):
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
    failed_pocket_num = 0
    for index, pocket_ligand_dict in enumerate(pocket_ligand_dict_list):
        pocket_path = pocket_ligand_dict['pocket_path']
        ligand_path = pocket_ligand_dict['ligand_path']
        xscore = pocket_ligand_dict['xscore']

        mol = Chem.SDMolSupplier(ligand_path)[0]
        if not mol:
            # print('read sdf failed: {}'.format(ligand_path))
            continue

        vertice, errno = get_pocket_aa_vertice(pocket_path, mol.GetConformer().GetPositions(), dist_t=6, config=config)

        if errno:
            # print('get pocket vertice failed: {}'.format(pocket_path))
            continue
    
        for dist_t in [3.2, 3.7, 4.2, 4.7, 4.7, 4.7]:
            ligand_mol, property_str, ligand_str, coord_offset, rot_matrix, ligand_coords_copy = gen_3dsmiles(
                ligand_path, 
                if_train=True
            )
            
            if not ligand_mol or not property_str or not ligand_str:
                # print('gen 3dsmiles failed: {}'.format(ligand_path))
                continue

            ret_no, pocket_str, dist_str = gen_pocket_str(
                vertice, 
                ligand_coords_copy,
                ligand_mol=ligand_mol, 
                coord_offset=coord_offset, 
                dist_t=dist_t,
                rot_matrix=rot_matrix,
                use_msms=use_msms
            )
            if ret_no != 0:
                # print('gen pocket str failed: {}'.format(pocket_path))
                failed_pocket_num += 1
                continue

            tdsmiles = pocket_str + '|' + property_str + '|' + ligand_str + '|' + dist_str
            smiles_3d_list.append(tdsmiles)
            # print('smiles', tdsmiles)
        if index % 100 == 0:
            print('已处理{}个smiles'.format(index))
            print('failed_pocket_num: ', failed_pocket_num)

    return smiles_3d_list


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    res = []
    for i in range(n):
        l = lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        res.append({'lst': l, 'process_id': i})
    return res


def main(pdb_split_path, out_3dsmiles_path, pdb_more_path='', n_worker=64, use_msms=True):
    path_dict_list = []
    path_dict_list = pickle.load(open(pdb_split_path, 'rb'))
    pdb_more_list = []
    if pdb_more_path:
        pdb_more_list = pickle.load(open(pdb_more_path, 'rb'))['data_list']
        print('pdb_more_list', len(pdb_more_list))
    path_dict_list.extend(pdb_more_list)

    print('path_dict_list', len(path_dict_list))

    path_dict_list = path_dict_list

    max_smiles_string_num = 0
    smiles_string_num_list = []

    # 生成3dsmiles
    smiles_3d_list = []
    # 启用多进程
    
    gen_3dsmiles_list_p = partial(gen_3dsmiles_list, use_msms=use_msms)
    with Pool(processes=n_worker) as pool:
        results = pool.map(gen_3dsmiles_list_p, split_list(path_dict_list, n_worker))
    
    smiles_3d_list = [item for sublist in results for item in sublist]

    for smiles in smiles_3d_list:
        smiles_string_num_list.append(len(smiles))
        if len(smiles) > max_smiles_string_num:
            max_smiles_string_num = len(smiles)

    # 打印统计信息
    print('最大smiles字符串长度: {}'.format(max_smiles_string_num))
    print('平均smiles字符串长度: {}'.format(sum(smiles_string_num_list) / len(smiles_string_num_list)))
    print('3dsmiles数量: {}'.format(len(smiles_3d_list)))

    # 保存3dsmiles
    with open(out_3dsmiles_path, 'w') as f:
        for smiles in smiles_3d_list:
            f.write(smiles + '\n')


if __name__ == '__main__':
    pdb_split_path = 'data/dock_results_list.pkl'
    # pdb_more_path = 'data/for_train_gened_mmff_fixed_xscore.pkl'
    pdb_more_path = None
    # 部分平移，只剩下旋转
    out_3dsmiles_path = 'data/pocket_aa_dist.txt'
    use_msms = False
    n_worker = 48

    print('start', out_3dsmiles_path)
    print('use_msms', use_msms)

    main(pdb_split_path, out_3dsmiles_path, pdb_more_path=pdb_more_path, n_worker=n_worker, use_msms=use_msms)

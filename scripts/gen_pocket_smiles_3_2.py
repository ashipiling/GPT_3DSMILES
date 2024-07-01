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
    from scripts.surface import Surface


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)



MAX_ATOM_NUM = 48  # 38
TRY_TIME = 2  # 4
DIST_THRESHOLD = 5
CUBE_SIZE = 2
ALIGN_XYZ = False
RADII = {
    "N": "1.850000", 
    "O": "1.800000", 
    "F": "1.750000", 
    "C": "1.90000", 
    "H": "1.200000", 
    "S": "1.950000", 
    "P": "1.900000", 
    "Z": "1.39", 
    "X": "0.770000", 
    "B": "2.00", 
    "I": "1.95",
    "Cl": "1.90"
}

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


def encode_number(num):
    if num > 39.9 or num < -39.9:
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


def calc_coord_delta(coord1, coord2):
    return sum([abs(coord1[i] - coord2[i]) for i in range(3)])


def find_farthest_atoms(coords):
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    a, b = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    return a, b


def find_nearest_atom_to_a(coords, a):
    dist_to_a = np.linalg.norm(coords - coords[a], axis=1)
    dist_to_a[a] = np.inf  # exclude a itself
    c = np.argmin(dist_to_a)
    return c


def calculate_rotation_matrix(coords, a, b, c):
    # calculate translation vector
    translation_vector = (coords[a] + coords[b]) / 2
    # translate coordinates
    coords -= translation_vector
    # calculate first rotation matrix
    ab = coords[b] - coords[a]
    ab /= np.linalg.norm(ab)  # normalize
    c1 = np.cross(ab, [1, 0, 0])
    c1 /= np.linalg.norm(c1)  # normalize
    v1 = np.cross(c1, ab)
    rotation_matrix_1 = np.array([ab, v1, c1])
    # apply first rotation
    coords = np.dot(coords, rotation_matrix_1.T)
    # calculate second rotation matrix
    oc = coords[c]
    angle = -np.arctan2(oc[2], oc[1])
    if oc[1] < 0:
        angle += np.pi

    rotation_matrix_2 = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    # apply second rotation
    coords = np.dot(coords, rotation_matrix_2.T)
    # combine rotation matrices
    rotation_matrix = np.dot(rotation_matrix_2, rotation_matrix_1)
    return coords, rotation_matrix, translation_vector


def out_3dsmiles_alignxyz(mol, c):
    coords = copy.deepcopy(c)
    a, b = find_farthest_atoms(coords)
    if a > b:  # make sure a is the atom with smaller index
        a, b = b, a
    c = find_nearest_atom_to_a(coords, a)
    coords, rot_matrix, coord_offset = calculate_rotation_matrix(coords, a, b, c)
    
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

    return three_dimension_smiles, rot_matrix, coord_offset


def out_3dsmiles_randomxyz(mol, ligand_coords_copy, if_train=False):
    # err_no, coords = fix_coords(mol)
    coords = copy.deepcopy(ligand_coords_copy)

    # 找到坐标的中心点
    coord_offset = np.array([0., 0., 0.])
    for c in coords:
        coord_offset += c
    coord_offset /= len(coords)

    # 随机在xyz轴上平移2a以内坐标
    # if if_train:
    # if random.random() > 0.2:
    #     coord_offset += np.random.rand(3) * 22 - 11
    # rd1 = random.random() + random.choice(np.concatenate([np.arange(-20, -10), np.arange(10, 20)]))
    # rd2 = random.random() + random.choice(np.concatenate([np.arange(-20, -10), np.arange(10, 20)]))
    # rd3 = random.random() + random.choice(np.concatenate([np.arange(-20, -10), np.arange(10, 20)]))
    # trans = np.array([rd1, rd2, rd3])
    # coord_offset += trans

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
        coords = np.dot(coords, rot_matrix.T)


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

    return three_dimension_smiles, rot_matrix, coord_offset


def gen_3dsmiles(ligand_path, ligand_coords_copy, if_train=True):
    # smiles 过滤
    # 如果包含C O N S P F Cl Br I以外的元素，过滤掉
    # return smiles3d
    if type(ligand_path) is str:
        mol = Chem.SDMolSupplier(ligand_path)[0]
    else:
        mol = ligand_path

    if not mol:
        print('read sdf failed: {}'.format(ligand_path))
        return None, None, None, None, None

    if check_atom_symbol(mol):
        mol = Chem.RemoveHs(mol)
        # Chem.RemoveStereochemistry(mol)
        qed_v = calc_qed(mol)
        logp_v = calc_logp(mol)
        property_str = 'qed' + str(qed_v) + '|' + 'logp' + str(logp_v)

        if ALIGN_XYZ:
            tdsmiles, rot_matrix, coord_offset = out_3dsmiles_alignxyz(mol, ligand_coords_copy)
        else:
            tdsmiles, rot_matrix, coord_offset = out_3dsmiles_randomxyz(mol, ligand_coords_copy, if_train=if_train)

        if not tdsmiles:
            print('not tdsmiles')
            return None, None, None, None, None

        return mol, property_str, tdsmiles, coord_offset, rot_matrix
    else:
        # print('check_atom_symbol failed')
        return None, None, None, None, None


def transform_coord(coord, coord_offset=np.array([0,0,0]), rot_matrix=np.eye(3)):
    coord -= coord_offset
    coord = np.dot(coord, rot_matrix.T)
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


def save_shortest_coords(ligand_coords, vertices, threshold=4.5):
    try:
        # 计算vertices到ligand_coords每个点的距离矩阵
        dist_matrix = distance_matrix(vertices, ligand_coords)

        # 为每个点找到最近的ligand_coords点的索引
        closest_indices = np.argmin(dist_matrix, axis=1)

        # 对于距离最近的点,进一步比较距离,距离大的放在前面
        sorted_vertices = []
        for i in range(len(ligand_coords)):
            closest_verts = [v for j, v in enumerate(vertices) if closest_indices[j] == i and np.linalg.norm(v - ligand_coords[i]) <= threshold]
            if closest_verts:
                dists = [np.linalg.norm(v - ligand_coords[i]) for v in closest_verts]
                sorted_closest = [v for _, v in sorted(zip(dists, closest_verts), reverse=True)]
                sorted_vertices.extend(sorted_closest)

        # 现在sorted_vertices就有了一定的顺序和规律

        return 0, sorted_vertices
    except:
        return -1, []


def save_vertice(ligand_coords, vertice, threshold=4.5):
    id_list = []
    dist_mat = distance_matrix(ligand_coords, vertice)
    # 保留距离ligand_coords附近5单位的vertice的点
    # 按照ligandcoords的顺序返回vertice的点，并且去重

    for indexi, i in enumerate(dist_mat):
        for indexj, j in enumerate(i):
            if j < threshold and indexj not in id_list:
                id_list.append(indexj)
    
    # 转为vertice
    return vertice[id_list]


def gen_pocket_str(
        vertices, 
        ligand_mol=None, 
        coord_offset=np.array([0.,0.,0.]), 
        dist_t=4.5,
        rot_matrix=np.eye(3),
        use_expand=False,
        ori_xscore=-7,
        use_msms=True
    ):
    # 读取pdb文件，生成pocket_str
    # pocket_str = 'XXXXX&{0,0,0{0,0,0{0,0,0{0,0,0{0,0,0'
    ret_str = ''
    pocket_coord_str = ''

    ver_list = copy.deepcopy(vertices)
    for f_vert in ver_list:
        ret_no, coord_str = transform_coord(f_vert, coord_offset=coord_offset, rot_matrix=rot_matrix)
        if ret_no == 0:
            pocket_coord_str += coord_str
        else:
            return -1, None

    ret_str = pocket_coord_str

    return 0, ret_str


def interpolate_coordinates(coords, max_distance=1.25, num_interpolations=1, max_inter_cnt=1, random_t=0):
    coordinates = copy.deepcopy(coords)
    interpolated_coordinates = []
    cnt = -1
    # 遍历每对相邻坐标
    for i in range(len(coordinates)):
        i_cnt = 0
        for j in range(len(coordinates)):
            if i != j:
                start_coord = np.array(coordinates[i])
                end_coord = np.array(coordinates[j])
                # 计算相邻坐标之间的距离
                dist = distance.euclidean(start_coord, end_coord)

                # 如果距离小于等于最大距离，则进行插值
                if dist <= max_distance:
                    i_cnt += 1
                    if i_cnt > max_inter_cnt:
                        break
                    if random.random() < random_t:
                        break
                    # 计算相邻坐标之间的插值步长
                    step = (end_coord - start_coord) / (num_interpolations + 1)
                    
                    # 对相邻坐标之间进行插值
                    for k in range(1, num_interpolations + 1):
                        cnt += 1
                        interpolated_coord = list(start_coord + k * step)
                        interpolated_coordinates.append({'index':i + 2 + cnt, 'coord': interpolated_coord})
    
    for c_dict in interpolated_coordinates:
        coordinates.insert(c_dict['index'], c_dict['coord'])
    return coordinates


def expand_vertices(vertice_list, ligand_coords, use_expand=False):

    if len(vertice_list) < 35:
        vertice_list = interpolate_coordinates(vertice_list, num_interpolations = 2, random_t=0.7)
    elif len(vertice_list) < 52:
        vertice_list = interpolate_coordinates(vertice_list, random_t=0.4)
    else:
        vertice_list = interpolate_coordinates(vertice_list, random_t=0.8)
    
    ret_list = []
    for vertice in vertice_list:
        nearest_coord = find_nearest_coord(vertice, ligand_coords)
        new_vertice = expand_vertice(vertice, nearest_coord, offset=1.5)  # 1.5
        ret_list.append(new_vertice)
    return ret_list


def narrow_vertices(vertice_list, ligand_coords):
    ret_list = []
    for vertice in vertice_list:
        nearest_coord = find_nearest_coord(vertice, ligand_coords)
        new_vertice = expand_vertice(vertice, nearest_coord, offset=-2.5)  # 1.5
        ret_list.append(new_vertice)
    return ret_list


def find_nearest_coord(vertice, ligand_coords):
    min_dist = 1e8
    min_coord = ligand_coords[0]
    for coord in ligand_coords:
        dist = calc_dist(vertice, coord)
        if dist < min_dist:
            min_dist = dist
            min_coord = coord

    return min_coord


def expand_vertice(A, B, offset=1.4):
    # 计算BA向量的坐标差值
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    dz = B[2] - A[2]
    
    # 计算BA向量的长度
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    
    # 计算单位向量
    unit_vector_x = dx / distance
    unit_vector_y = dy / distance
    unit_vector_z = dz / distance
    
    # 计算新的A的坐标
    new_A_x = A[0] - unit_vector_x * offset
    new_A_y = A[1] - unit_vector_y * offset
    new_A_z = A[2] - unit_vector_z * offset
    
    # 返回新的A坐标
    # print(A, B, [new_A_x, new_A_y, new_A_z])
    return [new_A_x, new_A_y, new_A_z]


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

        sur = Surface(config, radii=RADII)
        ret_dict = sur.calc_pocket_vertice(ori_pocket_path)
        if ret_dict['error_no']:
            return None, -1
        
        return ret_dict['vertices'], 0
    
    except Exception as e:
        return None, -1


def get_pocket_vertice(pocket_path, ligand_coords, dist_t=6, config={}):
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
        ret_dict = sur.calc_pocket_vertice(ori_pocket_path, cube_size=CUBE_SIZE)
        if ret_dict['error_no']:
            return None, -1
        
        return ret_dict['vertices'], 0
    except Exception as e:
        traceback.print_exc()
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

        mol = Chem.SDMolSupplier(ligand_path)[0]
        if not mol:
            # print('read sdf failed: {}'.format(ligand_path))
            continue

        # 获取蛋白表面点
        if use_msms:
            vertice, errno = get_pocket_vertice(pocket_path, mol.GetConformer().GetPositions(), config=config)
        else:
            # 获取残基点
            vertice, errno = get_pocket_aa_vertice(pocket_path, mol.GetConformer().GetPositions(), config=config)
        
        if errno:
            # print('get pocket vertice failed: {}'.format(pocket_path))
            continue

        err, ligand_coords_copy = fix_coords(mol)
        if err:
            continue

        err, ver_list = save_shortest_coords(ligand_coords_copy, vertice)
        if err:
            continue
    
        for dist_t in [3.2, 3.7, 4.2, 4.7, 4.7, 4.7]:
            ligand_mol, property_str, ligand_str, coord_offset, rot_matrix = gen_3dsmiles(
                ligand_path, 
                ligand_coords_copy,
                if_train=True
            )
            
            if not ligand_mol or not property_str or not ligand_str:
                # print('gen 3dsmiles failed: {}'.format(ligand_path))
                continue

            ret_no, pocket_str = gen_pocket_str(
                ver_list, 
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
            tdsmiles = pocket_str  + '|' + property_str + '|' + ligand_str
            smiles_3d_list.append(tdsmiles)

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

    # path_dict_list = path_dict_list[:100]

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
    out_3dsmiles_path = 'data/pocket_cube2_thres45_randomxyz.txt'
    use_msms = True
    n_worker = 90

    print('start', out_3dsmiles_path)
    print('use_msms', use_msms)

    main(pdb_split_path, out_3dsmiles_path, pdb_more_path=pdb_more_path, n_worker=n_worker, use_msms=use_msms)


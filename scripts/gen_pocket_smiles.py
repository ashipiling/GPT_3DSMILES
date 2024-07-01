# 该脚本将crossdocked数据集中的蛋白质口袋和配体提取出来，生成3dsmiles
import os, copy
from multiprocessing import Pool
import traceback
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit import RDLogger
import torch
from prody import parsePDB, writePDB
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

    # 随机在xyz轴上平移11a以内坐标
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
        pocket_path, 
        ligand_mol=None, 
        coord_offset=np.array([0.,0.,0.]), 
        dist_t=4.5,
        rot_matrix=np.eye(3)
    ):
    # 读取pdb文件，生成pocket_str
    # pocket_str = 'XXXXX&{0,0,0{0,0,0{0,0,0{0,0,0{0,0,0'
    ret_str = ''

    pocket_res_str = ''
    pocket_coord_str = ''
    tmp_dict_list = []
    ligand_coords = ligand_mol.GetConformer().GetPositions()
    try:
        structure = parsePDB(pocket_path)
        protein = structure.select('protein')  # remove water and other useless
        selected = protein.select('same residue as within %s of ligand' % dist_t, ligand=ligand_coords)
        if len(selected) == 0:
            print('no pocket')
            return None
        # 打印出selected的氨基酸及其位置
        for nn in selected:
            tmp_dict_list.append({'resname': nn.getResname(), 'coord': nn.getCoords()})
            
        index_set = set()
        for ligand_atom_coord in ligand_coords:
            min_index = -1
            min_dist = 99999
            for index, pocket_atom in enumerate(tmp_dict_list):
                dist = calc_dist(ligand_atom_coord, pocket_atom['coord'])
                if dist < min_dist:
                    min_dist = dist
                    min_index = index
            if not min_index in index_set:
                index_set.add(min_index)
                coord = copy.deepcopy(tmp_dict_list[min_index]['coord'])
                resname = tmp_dict_list[min_index]['resname']
                if resname not in AA_DICT:
                    pocket_res_str += 'X'
                else:
                    pocket_res_str += AA_DICT[resname]
                
                pocket_coord_str += transform_coord(coord, coord_offset=coord_offset, rot_matrix=rot_matrix)

        ret_str = pocket_res_str + '&' + pocket_coord_str

    except:
        traceback.print_exc()
        print('parsePDB failed: {}'.format(pocket_path))
        return None

    return ret_str
    

def gen_3dsmiles_list(pocket_ligand_dict_list):
    smiles_3d_list = []
    for index, pocket_ligand_dict in enumerate(pocket_ligand_dict_list):
        pocket_path = pocket_ligand_dict['pocket_path']
        ligand_path = pocket_ligand_dict['ligand_path']
        for dist_t in [4.2, 4.7, 5.2, 5.7]:
            ligand_mol, property_str, ligand_str, coord_offset, rot_matrix = gen_3dsmiles(
                ligand_path, 
                if_train=True
            )
            if not ligand_mol or not property_str or not ligand_str:
                continue

            pocket_str = gen_pocket_str(
                pocket_path, 
                ligand_mol=ligand_mol, 
                coord_offset=coord_offset, 
                dist_t=dist_t,
                rot_matrix=rot_matrix
            )
            if not pocket_str:
                continue

            smiles_3d_list.append(property_str + '|' + pocket_str + '|' + ligand_str)
            
            if index % 1000 == 0:
                print('已处理{}个smiles'.format(index))

    return smiles_3d_list


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main(data_dir, out_3dsmiles_path, split_dict, n_worker=32):
    path_dict_list = []
    train_path_list = split_dict['train']
    val_path_list = split_dict['test']
    for path in train_path_list:
        ligand_path = os.path.join(data_dir, path[1])
        pocket_path = os.path.join(data_dir, path[0])
        path_dict_list.append({
            'ligand_path': ligand_path,
            'pocket_path': pocket_path
        })

    max_smiles_string_num = 0
    smiles_string_num_list = []

    # 生成3dsmiles
    smiles_3d_list = []
    # 启用多进程

    with Pool(processes=n_worker) as pool:
        results = pool.map(gen_3dsmiles_list, split_list(path_dict_list, n_worker))
    
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
    data_dir = 'data/cross_docked/'
    out_3dsmiles_path = 'data/pocketsmiles_crossdocked_with_property_7.txt'
    split_path = 'data/split_by_name.pt'
    split_dict = torch.load(split_path)

    main(data_dir, out_3dsmiles_path, split_dict)


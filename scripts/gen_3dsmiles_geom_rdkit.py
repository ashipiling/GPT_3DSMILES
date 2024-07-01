from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
import copy
from rdkit import RDLogger
from multiprocessing import Pool
import json
import os
import pickle
import numpy as np
import random


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

MAX_ATOM_NUM = 48
not_use_full = True
max_index = 64

if_high = True
if_iso = False

def check_atom_symbol(smiles):
    atom_symbol_list = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B']
    # 去除 H
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print('smiles格式错误: {}'.format(smiles))
        return False
    try:
        mol = Chem.RemoveHs(mol)
    except:
        print('RemoveHs failed: {}'.format(smiles))
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in atom_symbol_list:
            print('smiles中包含了不支持的元素: {}'.format(atom.GetSymbol()), smiles)
            return False
    if mol.GetNumAtoms() > MAX_ATOM_NUM:
        print('smiles中原子数超过了最大限制: {}'.format(mol.GetNumAtoms()))
        return False
    return True


def encode_number(num):
    if num > 39.9 or num < -39.9:
        print('coords 太大')
        return -1, None
    
    if if_high:
        num = int(round(num * 50, 0))
    else:
        num = int(round(num * 10, 0))

    prefix = ''
    if abs(num) != num:
        prefix = '-'
    num_str = prefix + '0' * (4 - len(str(abs(num)))) + str(abs(num))

    return 0, num_str


def rotate_coord(coords):
    # 找到坐标的中心点
    coord_offset = np.array([0., 0., 0.])
    for c in coords:
        coord_offset += c
    coord_offset /= len(coords)

    # 随机在xyz轴上平移2a以内坐标
    if random.random() > 0.8:
        coord_offset += np.random.rand(3) * 22 - 11

    coords -= coord_offset

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
    
    return coords


def calc_mol_rmsd(mol):
    # 计算mol的rmsd
    # mol已经有3d坐标
    old_mol1 = copy.deepcopy(mol)
    try_time = 3
    rmsd_list = []
    while try_time > 0:
        old_mol = copy.deepcopy(mol)
        try:
            AllChem.MMFFOptimizeMolecule(old_mol)
            rmsd = AllChem.GetBestRMS(old_mol1, old_mol)
            rmsd_list.append(rmsd)
        except:
            print('faild')
        try_time -= 1
    if len(rmsd_list) > 0:
        return min(rmsd_list)
    else:
        return 10


def fix_coords(mol):
    cp_mol = copy.deepcopy(mol)
    smi = Chem.MolToSmiles(cp_mol, isomericSmiles=True)
    core = Chem.MolFromSmarts(smi)
    match = cp_mol.GetSubstructMatch(core)

    if len(match) != cp_mol.GetNumAtoms():
        print('match长度不一致', match)
        return -1, None

    # out_mol = Chem.RenumberAtoms(cp_mol, newOrder=match)
    coords = mol.GetConformer().GetPositions()

    ret_coords = []
    # 根据align_dict对coords进行重排
    # print('match', match)
    for index_cp in match:
        ret_coords.append(list(coords[index_cp]))
    
    return 0, np.array(ret_coords)


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def out_3dsmiles(relative_str, mol):
    ret, coords = fix_coords(mol)
    if ret:
        return ret, None

    coords = rotate_coord(coords)
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    
    # 生成3dsmiles
    three_dimension_smiles = relative_str + '|' + canonical_smiles
    three_dimension_smiles += '&'
    out = ''

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        out += str(idx) + ' ' +  atom.GetSymbol()
        ret, x = encode_number(coords[idx][0])
        if ret:
            return -1, None
        ret, y = encode_number(coords[idx][1])
        if ret:
            return -1, None
        ret, z = encode_number(coords[idx][2])
        if ret:
            return -1, None
        coord = '{' + str(x) + ',' + str(y) + ',' + str(z)
        three_dimension_smiles += coord

    return 0, three_dimension_smiles


def gen_3dsmiles(dic):
    ret_list = []
    pickle_path = os.path.join('data/rdkit_folder', dic['pickle_path'])
    mol_dic = pickle.load(open(pickle_path, 'rb'))
    confs = mol_dic['conformers']

    if not_use_full:
        confs = confs[ :max_index]
    else:
        confs = confs

    for conf in confs:
        relativeenergy = float(conf['relativeenergy'])
        mol = conf['rd_mol']
        mol = Chem.RemoveHs(mol)
        if if_iso:
            pass
        else:
            Chem.RemoveStereochemistry(mol)

        if relativeenergy > 2.57:
            relativeenergy_str = '0'
        else:
            relativeenergy_str = '1'

        smiles = Chem.MolToSmiles(mol)

        if check_atom_symbol(smiles):
            err, tdsmiles = out_3dsmiles(relativeenergy_str, mol)
            if err:
                print('out_3dsmiles error')
                continue
            else:
                ret_list.append(tdsmiles)

    return ret_list


def gen_3dsmiles_list(lst):
    smiles_3d_list = []
    failed_cnt = 0
    print('lst', len(lst))

    for index, dic in enumerate(lst):
        try:
            three_dimension_smiles_list = gen_3dsmiles(dic)
            smiles_3d_list += three_dimension_smiles_list
        except:
            # print('failed')
            failed_cnt += 1

        if index % 200 == 0:
            print('comp index: ', index)

    return smiles_3d_list


n_worker = 92


geom_json_path = 'data/rdkit_folder/summary_drugs.json'
out_3dsmiles_path = 'data/3dsmiles_pubchem_10m_38_atom_with_property_geom_not_full_low_rdkit_high.txt'

dic = json.load(open(geom_json_path, 'r'))

dic_list = []
for key in dic:
    dic_list.append(dic[key])

# dic_list = dic_list[:200]
print('dic_list', len(dic_list))

with Pool(processes=n_worker) as pool:
    results = pool.map(gen_3dsmiles_list, split_list(dic_list, n_worker))

ret_list = [item for sublist in results for item in sublist]

print('3dsmiles数量: {}'.format(len(ret_list)))
# 保存3dsmiles
with open(out_3dsmiles_path, 'w') as f:
    for smiles in ret_list:
        f.write(smiles + '\n')

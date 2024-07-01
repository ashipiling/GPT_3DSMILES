# 该脚本将化学smiles转为3dsmiles
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit import RDLogger
from multiprocessing import Pool
import random
import traceback
import os, copy
import numpy as np
from scipy.spatial import distance_matrix
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
try:
    from surface import Surface
except:
    from scripts.surface import Surface

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import pickle

MAX_ATOM_NUM = 42
TRY_TIME = 1  # 4
USE_POCKET = False

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

# RADII = {
#     "N": "1.650000", 
#     "O": "1.600000", 
#     "F": "1.650000", 
#     "C": "1.60000", 
#     "H": "1.200000", 
#     "S": "1.650000", 
#     "P": "1.600000", 
#     "Z": "1.39", 
#     "X": "0.770000", 
#     "B": "1.60", 
#     "I": "1.65",
#     "Cl": "1.60"
# }

def check_atom_symbol(smiles):
    atom_symbol_list = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']
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
            print('smiles中包含了不支持的元素: {}'.format(atom.GetSymbol()))
            return False
    if mol.GetNumAtoms() > MAX_ATOM_NUM:
        # print('smiles中原子数超过了最大限制: {}'.format(mol.GetNumAtoms()))
        return False
    return True


def get_ligand_vertice_from_mol(mol, config={}, cube_size=2.5):
    os.makedirs(config['surface_save_dir'], exist_ok=True)
    pdb_path = os.path.join(config['surface_save_dir'], 'ligand.pdb')
    try:
        Chem.MolToPDBFile(mol, pdb_path)
        sur = Surface(config, radii=RADII)
        ret_dict = sur.calc_pocket_vertice(pdb_path, cube_size=cube_size)
        if ret_dict['error_no']:
            return None, -1
        
        return ret_dict['vertices'], 0
    except Exception as e:
        traceback.print_exc()
        print(Chem.MolToSmiles(mol))
        return None, -1


def generate_vertieces(mol, process_id=0, remove=True):
    config = {
        'surface_msms_bin': '/home/luohao/molGen/msms/msms',
        'surface_save_dir': 'data/surface/' + str(process_id),
        'mesh_threshold': 2
    }
    vertices, error = get_ligand_vertice_from_mol(mol, config, cube_size=0)
    if error:
        return None

    return vertices


def gen_3dsmiles(smiles, process_id=0):
    # smiles 过滤
    # 如果包含C O N S P F Cl Br I以外的元素，过滤掉
    # return [smiles3d, smiles3d, ...]
    ret_smiles_3d_list = []
    if check_atom_symbol(smiles):
        ori_mol = Chem.MolFromSmiles(smiles)
        ori_mol = Chem.RemoveHs(ori_mol)
        # 引入iso
        isomers = tuple(EnumerateStereoisomers(ori_mol))
        isomers = isomers[-2: ]
        for mol in isomers:
        # 取前三个
            try_time = TRY_TIME
            while try_time:
                try:
                    ret = AllChem.EmbedMolecule(mol)
                    if ret != 0:
                        try_time -= 1
                        continue
                except:
                    print('EmbedMolecule failed: {}'.format(smiles))
                    try_time -= 1
                    continue
                # 能量最小化
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except:
                    print('MMFFOptimizeMolecule failed: {}'.format(smiles))
                    try_time -= 1
                    continue
                # 生成3dsmiles
                if USE_POCKET:
                    vertices = generate_vertieces(mol, process_id)
                    if vertices is None:
                        try_time -= 1
                        continue
                else:
                    vertices = []

                ret_smiles_3d_list.append((copy.deepcopy(mol), vertices))
                try_time -= 1

        return ret_smiles_3d_list
    else:
        return []


def gen_3dsmiles_list(dic):
    smiles_3d_list = []
    for index, smiles in enumerate(dic['lst']):
        try:
            three_dimension_smiles_list = gen_3dsmiles(smiles, process_id=dic['process_id'])
        except:
            three_dimension_smiles_list = []
            pass
        smiles_3d_list += three_dimension_smiles_list
        
        if index % 100 == 0:
            print('已处理{}个smiles'.format(index), dic['process_id'])
            # print('tdsmiles', smiles_3d_list[-1])
    return smiles_3d_list


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    res = []
    for i in range(n):
        l = lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        res.append({'lst': l, 'process_id': i})
    return res


def main(smiles_path, out_3dsmiles_path, n_worker=64):
    # 读取smiles文件
    with open(smiles_path, 'r') as f:
        smiles_list = f.readlines()
    # 生成3dsmiles
    smiles_3d_list = []
    # 启用多进程
    print(len(smiles_list))
    smiles_list = smiles_list[:4000000]

    batch_size = 100000
    num_batches = len(smiles_list) // batch_size + 1

    for i in range(num_batches):
        print('index', i)
        start = i * batch_size
        end = start + batch_size
        batch_data = smiles_list[start:end]
        
        with Pool(processes=n_worker) as pool:
            results = pool.map(gen_3dsmiles_list, split_list(batch_data, n_worker))
        
        print('finished')
        smiles_3d_list = [item for sublist in results for item in sublist]
        pickle.dump(smiles_3d_list, open(out_3dsmiles_path + str(i) + '.pkl', 'wb'))


if __name__ == '__main__':
    smiles_path = 'data/pubchem-10m.txt'
    out_3dsmiles_path = 'data/pickle_nopocket/data_batch_'
    
    print(smiles_path, out_3dsmiles_path)
    main(smiles_path, out_3dsmiles_path)

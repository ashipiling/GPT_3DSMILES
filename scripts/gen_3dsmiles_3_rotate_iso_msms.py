# 该脚本将化学smiles转为3dsmiles
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit import RDLogger
from multiprocessing import Pool
import random
import traceback
import os
import numpy as np
from scipy.spatial import distance_matrix
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
try:
    from surface import Surface
except:
    from scripts.surface import Surface

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


MAX_ATOM_NUM = 42
TRY_TIME = 2  # 4

# RADII = {
#     "N": "1.850000", 
#     "O": "1.800000", 
#     "F": "1.750000", 
#     "C": "1.90000", 
#     "H": "1.200000", 
#     "S": "1.950000", 
#     "P": "1.900000", 
#     "Z": "1.39", 
#     "X": "0.770000", 
#     "B": "2.00", 
#     "I": "1.95",
#     "Cl": "1.90"
# }

RADII = {
    "N": "1.650000", 
    "O": "1.600000", 
    "F": "1.650000", 
    "C": "1.60000", 
    "H": "1.200000", 
    "S": "1.650000", 
    "P": "1.600000", 
    "Z": "1.39", 
    "X": "0.770000", 
    "B": "1.60", 
    "I": "1.65",
    "Cl": "1.60"
}

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


def calc_qed(mol):
    qed_v = 0.0
    try:
        qed_v = QED.qed(mol)
    except:
        pass
    if qed_v >= 0.5:
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


def save_shortest_coords(vertices, mol):
    ligand_coords = mol.GetConformer().GetPositions()

    # 计算vertices到ligand_coords每个点的距离矩阵
    dist_matrix = distance_matrix(vertices, ligand_coords)

    # 为每个点找到最近的ligand_coords点的索引
    closest_indices = np.argmin(dist_matrix, axis=1)
    # 根据closest_indices对vertices进行排序
    sorted_vertices = [vertices[i] for i in np.argsort(closest_indices)]
    # 现在sorted_vertices就有了一定的顺序和规律
    return sorted_vertices


def remove_some_vertices(vertices, removed_min_num=3, removed_max_num=7, remove=True):
    # vertices后65%的数据中
    # 连续删除3-7个点
    if not remove:
        return vertices
    try:
        removed_num = np.random.randint(removed_min_num, removed_max_num)
        # 选择删除的起始点
        start_index = np.random.randint(len(vertices) // 2.5, len(vertices) - removed_num)
        # 删除后的点
        vertices = np.delete(vertices, range(start_index, start_index + removed_num), axis=0)
    except:
        traceback.print_exc()
    
    return vertices


def generate_vertieces(mol, process_id=0, remove=True):
    config = {
        'surface_msms_bin': '/home/luohao/molGen/msms/msms',
        'surface_save_dir': 'data/surface/' + str(process_id),
        'mesh_threshold': 2
    }
    vertices, error = get_ligand_vertice_from_mol(mol, config)
    if error:
        return None
    sorted_vertices = save_shortest_coords(vertices, mol)
    final_vertices = remove_some_vertices(sorted_vertices, remove=remove)

    return final_vertices


def rotate_coord(coords, vertices=None):
    # 找到坐标的中心点
    coord_offset = np.array([0., 0., 0.])
    for c in coords:
        coord_offset += c
    coord_offset /= len(coords)

    # 随机在xyz轴上平移2a以内坐标
    if random.random() > 0.8:
        coord_offset += np.random.rand(3) * 22 - 11

    coords -= coord_offset
    if vertices is not None:
        vertices -= coord_offset

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
    if vertices is not None:
        vertices = np.dot(vertices, rot_matrix)
    
    return coords, vertices


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


def encode_number(num):
    if num > 199.9 or num < -199.9:
        print('coords 太大')
        return -1, None

    num = int(round(num * 10, 0))
    prefix = ''
    if abs(num) != num:
        prefix = '-'
    num_str = prefix + '0' * (4 - len(str(abs(num)))) + str(abs(num))

    return 0, num_str


def out_3dsmiles(qed_v, logp_v, mol, vertices=None):
    coords = mol.GetConformer().GetPositions()
    
    coords, vertices = rotate_coord(coords, vertices=vertices)
    
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    vertices_str = coord_2_str(vertices)

    # 生成3dsmiles
    three_dimension_smiles = vertices_str + '|' + 'qed' + str(qed_v) + '|' + 'logp' + str(logp_v) + '|' + canonical_smiles

    three_dimension_smiles += '&'
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
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
        isomers = isomers[-3: ]
        for mol in isomers:
        # 取前三个
            try_time = TRY_TIME
            qed_v = calc_qed(mol)
            logp_v = calc_logp(mol)
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
                vertices = generate_vertieces(mol, process_id)
                if vertices is None:
                    try_time -= 1
                    continue
                err, tdsmiles = out_3dsmiles(qed_v, logp_v, mol, vertices=vertices)
                if err:
                    try_time -= 1
                    continue
                # print('3dsmiles: {}'.format(tdsmiles))
                ret_smiles_3d_list.append(tdsmiles)
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


def main(smiles_path, out_3dsmiles_path, n_worker=92):

    max_smiles_string_num = 0
    smiles_string_num_list = []
    # 读取smiles文件
    with open(smiles_path, 'r') as f:
        smiles_list = f.readlines()
    # 生成3dsmiles
    smiles_3d_list = []
    # 启用多进程
    print(len(smiles_list))
    smiles_list = smiles_list[5000000:6700000]

    with Pool(processes=n_worker) as pool:
        results = pool.map(gen_3dsmiles_list, split_list(smiles_list, n_worker))
    
    print('finished')
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
    smiles_path = 'data/3dsmiles_msms_46.pkl'
    out_3dsmiles_path = 'data/3dsmiles_msms_alignxyz.txt'
    print(smiles_path, out_3dsmiles_path)
    main(smiles_path, out_3dsmiles_path)

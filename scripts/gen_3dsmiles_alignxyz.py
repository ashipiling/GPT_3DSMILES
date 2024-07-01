import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED
import scipy
from scipy.spatial import distance, distance_matrix
import traceback
import pickle
from multiprocessing import Pool
from rdkit import RDLogger
import random
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import glob
import os
import gc


CUBE_SIZE = 1.9


def load_molecule(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    for mol in suppl:
        if mol is not None:
            return mol
    return None


def get_coords(mol):
    conf = mol.GetConformer()
    return conf.GetPositions()


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


def new_mol(mol, coords):
    conf = mol.GetConformer()

    for atom in mol.GetAtoms():
        tmp_coords = coords[atom.GetIdx()]
        conf.SetAtomPosition(atom.GetIdx(), tmp_coords)


def save_shortest_coords(vertices, ligand_coords):
    try:
        # 计算vertices到ligand_coords每个点的距离矩阵
        dist_matrix = distance_matrix(vertices, ligand_coords)

        # 为每个点找到最近的ligand_coords点的索引
        closest_indices = np.argmin(dist_matrix, axis=1)

        # 对于距离最近的点,进一步比较距离,距离大的放在前面
        sorted_vertices = []
        for i in range(len(ligand_coords)):
            closest_verts = [v for j, v in enumerate(vertices) if closest_indices[j] == i]
            if closest_verts:
                dists = [np.linalg.norm(v - ligand_coords[i]) for v in closest_verts]
                sorted_closest = [v for _, v in sorted(zip(dists, closest_verts), reverse=True)]
                sorted_vertices.extend(sorted_closest)

        # 现在sorted_vertices就有了一定的顺序和规律

        return 0, sorted_vertices
    except:
        return -1, []


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


def cubefy(vert, cube_size=1.5):
    # 使用scipy的KDTree进行快速的空间查询
    kdtree = scipy.spatial.cKDTree(vert)
    # 选择一个合适的立方体大小
    cube_size = cube_size  # 1.5

    # 在每个立方体中选择一个点
    sparse_data = []
    for point in vert:
        if len(sparse_data) == 0 or np.min(scipy.spatial.distance.cdist([point], sparse_data)) >= cube_size:
            sparse_data.append(point)
    sparse_data = np.array(sparse_data)

    return sparse_data


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


def get_prop(mol):
    qed_v = calc_qed(mol)
    logp_v = calc_logp(mol)
    property_str = str(qed_v) + '|' + str(logp_v)
    return property_str


def process_vertices(vertices, mol_final_coords, rotation_matrix, translation_vector):
    vertices = cubefy(vertices, cube_size=CUBE_SIZE)
    new_v = np.dot((vertices - translation_vector), rotation_matrix.T)
    err, new_v = save_shortest_coords(new_v, mol_final_coords)
    if err:
        return None
    new_v = remove_some_vertices(new_v)
    return new_v


def coords_2_str(coords):
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
    coords = np.dot(coords, rot_matrix.T)

    return coords, rot_matrix, coord_offset


def find_nearest_not_ring_atom_to_a(coords, a, mol):
    dist_to_a = np.linalg.norm(coords - coords[a], axis=1)
    non_ring_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    # 设置环内原子的距离为无穷大，以便在求最小值时被忽略
    dist_to_a[[atom for atom in range(len(dist_to_a)) if atom not in non_ring_atoms]] = np.inf
    # 找到距离最小的非环原子
    nearest_not_ring_atom = np.argmin(dist_to_a)
    
    return nearest_not_ring_atom


def random_xyz(mol):
    coords = get_coords(mol)
    return rotate_coord(coords)


def align_xyz(mol):
    coords = get_coords(mol)
    a, b = find_farthest_atoms(coords)
    if a > b:  # make sure a is the atom with smaller index
        a, b = b, a
    c = find_nearest_atom_to_a(coords, a)

    coords, rotation_matrix, translation_vector = calculate_rotation_matrix(coords, a, b, c)
    # d = find_nearest_not_ring_atom_to_a(coords, a, mol)
    conf = mol.GetConformer()
    for index, atom in enumerate(mol.GetAtoms()):
        conf.SetAtomPosition(atom.GetIdx(), coords[index])

    cp_mol = Chem.Mol(mol)
    smi = Chem.MolToSmiles(cp_mol, isomericSmiles=True, rootedAtAtom=int(a), canonical=True)
    core = Chem.MolFromSmarts(smi)
    matches = cp_mol.GetSubstructMatches(core)
    match = matches[0]
    
    # 根据 match 列表重新排列原子顺序
    ans = match

    # 使用 RenumberAtoms 函数重新排列原子编号
    cp_mol = Chem.RenumberAtoms(cp_mol, ans)
    coords = cp_mol.GetConformer().GetPositions()
    return smi, coords, rotation_matrix, translation_vector


def gen_text(input_dict):
    # renumber
    ret_list = []

    for index, item in enumerate(input_dict['lst']):
        try:
            mol = item[0]
            vertices = item[1]
            
            # canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            canonical_smiles, mol_final_coords, rotation_matrix, translation_vector = align_xyz(mol)
            # mol_final_coords, rotation_matrix, translation_vector = random_xyz(mol)
            ligand_coords_str = coords_2_str(mol_final_coords)
            # if vertices:
            #     mol_final_vertices = process_vertices(vertices, mol_final_coords, rotation_matrix, translation_vector)
            #     if mol_final_vertices is None:
            #         continue
            #     vertices_coords_str = coords_2_str(mol_final_vertices)
            # else:
            vertices_coords_str = ''
            
            prop_str = get_prop(mol)
            final_str = vertices_coords_str + '|' + prop_str + '|' + canonical_smiles + '&' + ligand_coords_str
            ret_list.append(final_str)
        except:
            traceback.print_exc()
        if index % 1000 == 0:
            print('index', index)
            
    return ret_list


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    res = []
    for i in range(n):
        l = lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        res.append({'lst': l, 'process_id': i})
    return res


def main(data, out_3dsmiles_path, n_worker=64):

    with Pool(processes=n_worker) as pool:
        results = pool.map(gen_text, split_list(data, n_worker))
    
    print('finished')
    smiles_3d_list = [item for sublist in results for item in sublist]

    smiles_string_num_list = []
    max_smiles_string_num  = 0
    for smiles in smiles_3d_list:
        smiles_string_num_list.append(len(smiles))
        if len(smiles) > max_smiles_string_num:
            max_smiles_string_num = len(smiles)

    # 打印统计信息
    print('最大smiles字符串长度: {}'.format(max_smiles_string_num))
    print('平均smiles字符串长度: {}'.format(sum(smiles_string_num_list) / len(smiles_string_num_list)))
    print('3dsmiles数量: {}'.format(len(smiles_3d_list)))

    # 保存3dsmiles
    with open(out_3dsmiles_path, 'a') as f:
        for smiles in smiles_3d_list:
            f.write(smiles + '\n')


if __name__ == '__main__':

    out_3dsmiles_path = 'data/3dsmiles_msms_alignxyz_nopocket.txt'
    dir_name = 'data/pickle/'
    path_list = glob.glob(os.path.join(dir_name, '*.pkl'))
    for pickle_path in path_list:
        if pickle_path.split('.')[-1] != 'pkl':
            continue
        print(pickle_path, out_3dsmiles_path)
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(data[0], len(data[0][1]))
        main(data, out_3dsmiles_path)
        
        # delete the loaded data and collect garbage
        del data
        gc.collect()

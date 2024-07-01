from collections import deque

from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
import re
from rdkit.Chem import AllChem, QED
from rdkit import RDLogger
from multiprocessing import Pool
from scipy.spatial import distance_matrix
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import traceback
import numpy as np
import random
import os
try:
    from surface import Surface
except:
    from scripts.surface import Surface


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


TRY_TIME = 2
ISO_NUM = 2
MAX_ATOM_NUM = 38
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


def extract_tags(smiles):
    pattern = r'\[(\d+\*)\]'
    tags = re.findall(pattern, smiles)
    return tags


def get_rings(mol):
    rings = []
    for ring in list(Chem.GetSymmSSSR(mol)):
        ring = list(ring)
        rings.append(ring)
    return rings


def get_other_atom_idx(mol, atom_idx_list):
    ret_atom_idx = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atom_idx_list:
            ret_atom_idx.append(atom.GetIdx())
    return ret_atom_idx


def find_parts_bonds(mol, parts):
    ret_bonds = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            i_part = parts[i]
            j_part = parts[j]
            for i_atom_idx in i_part:
                for j_atom_idx in j_part:
                    bond = mol.GetBondBetweenAtoms(i_atom_idx, j_atom_idx)
                    if bond is None:
                        continue
                    ret_bonds.append((i_atom_idx, j_atom_idx))
    return ret_bonds


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


def get_anchors(all_coords):
    ret_coords = []
    for c in all_coords:
        ret_coords.append(np.array(c).mean(0).tolist())
    return ret_coords


def save_shortest_coords(vertices, all_coords):
    anchor_coords = get_anchors(all_coords)
    # 计算vertices到ligand_coords每个点的距离矩阵
    dist_matrix = distance_matrix(vertices, anchor_coords)

    # 为每个点找到最近的ligand_coords点的索引
    closest_indices = np.argmin(dist_matrix, axis=1)

    # 对于距离最近的点,进一步比较距离,距离大的放在前面
    sorted_vertices = []
    for i in range(len(anchor_coords)):
        closest_verts = [v for j, v in enumerate(vertices) if closest_indices[j] == i]
        if closest_verts:
            dists = [np.linalg.norm(v - anchor_coords[i]) for v in closest_verts]
            sorted_closest = [v for _, v in sorted(zip(dists, closest_verts), reverse=True)]
            sorted_vertices.extend(sorted_closest)

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


def generate_vertieces(mol, all_coords, process_id=0, remove=True, cube_size=2.5):
    config = {
        'surface_msms_bin': '/home/luohao/molGen/msms/msms',
        'surface_save_dir': 'data/surface/' + str(process_id),
        'mesh_threshold': 2
    }
    vertices, error = get_ligand_vertice_from_mol(mol, config, cube_size=cube_size)
    if error:
        return None
    sorted_vertices = save_shortest_coords(vertices, all_coords)
    final_vertices = remove_some_vertices(sorted_vertices, remove=remove)

    return final_vertices


class RING_R_Fragmenizer():
    def __init__(self):
        self.type = 'RING_R_Fragmenizer'

    def bonds_filter(self, mol, bonds):
        filted_bonds = []
        for bond in bonds:
            bond_type = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType()
            if not bond_type is BondType.SINGLE:
                continue
            f_atom = mol.GetAtomWithIdx(bond[0])
            s_atom = mol.GetAtomWithIdx(bond[1])
            if f_atom.GetSymbol() == '*' or s_atom.GetSymbol() == '*':
                continue
            if mol.GetBondBetweenAtoms(bond[0], bond[1]).IsInRing():
                continue
            filted_bonds.append(bond)
        return filted_bonds

    def get_bonds(self, mol):
        bonds = []
        rings = get_rings(mol)
        if len(rings) > 0:
            for ring in rings:
                rest_atom_idx = get_other_atom_idx(mol, ring)
                bonds += find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
        return bonds

    def fragmenize(self, mol, dummyStart=1):
        rings = get_rings(mol)
        if len(rings) > 0:
            bonds = []
            for ring in rings:
                rest_atom_idx = get_other_atom_idx(mol, ring)
                bonds += find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
            if len(bonds) > 0:
                bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
                bond_ids = list(set(bond_ids))
                dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
                break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
                dummyEnd = dummyStart + len(dummyLabels) - 1
            else:
                break_mol = mol
                dummyEnd = dummyStart - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1
        return break_mol, dummyEnd


class BRICS_Fragmenizer():
    def __inti__(self):
        self.type = 'BRICS_Fragmenizers'

    def get_bonds(self, mol):
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]
        return bonds

    def fragmenize(self, mol, dummyStart=1):
        # get bonds need to be break
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]

        # whether the molecule can really be break
        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]

            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return break_mol, dummyEnd


class BRICS_RING_R_Fragmenizer():
    def __init__(self):
        self.type = 'BRICS_RING_R_Fragmenizer'
        self.brics_fragmenizer = BRICS_Fragmenizer()
        self.ring_r_fragmenizer = RING_R_Fragmenizer()

    def fragmenize(self, mol, dummyStart=1):
        brics_bonds = self.brics_fragmenizer.get_bonds(mol)
        ring_r_bonds = self.ring_r_fragmenizer.get_bonds(mol)
        bonds = brics_bonds + ring_r_bonds

        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
            bond_ids = list(set(bond_ids))
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return break_mol, dummyEnd


# def frag_to_smiles(frag):
#     return Chem.MolToSmiles(frag, rootedAtAtom=0, canonical=True)


def frag_to_smiles(frag):

    cp_mol = Chem.Mol(frag)
    smi = Chem.MolToSmiles(cp_mol, isomericSmiles=True, rootedAtAtom=0, canonical=True)
    core = Chem.MolFromSmarts(smi)
    match = cp_mol.GetSubstructMatch(core)
    
    # 根据 match 列表重新排列原子顺序
    # 使用 RenumberAtoms 函数重新排列原子编号
    cp_mol = Chem.RenumberAtoms(cp_mol, match)

    ret = ''
    for atom in cp_mol.GetAtoms():
        ret += atom.GetSmarts()

    return smi, cp_mol


def encode_number(num):
    if num > 199.9 or num < -199.9:
        print('coords 太大', num)
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


def coord_2_str_v(coords):
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


def coord_2_str(coords):
    three_dimension_smiles = ''
    for c in coords:
        for idx in range(len(c)):
            ret, x = encode_number(c[idx][0])
            if ret:
                return None
            ret, y = encode_number(c[idx][1])
            if ret:
                return None
            ret, z = encode_number(c[idx][2])
            if ret:
                return None

            three_dimension_smiles += '{' + str(x) + ',' + str(y) + ',' + str(z)
        three_dimension_smiles += '^'

    return three_dimension_smiles[: -1]


def connect_frags_bfs(frags):

    start_frag = next((f for f in frags))
    if start_frag is None:
        raise ValueError("No acyclic fragment found as starting point.")

    visited = set()
    queue = deque([start_frag])
    bfs_smiles = []
    all_coords = []
    ori_coords = []

    while queue:
        frag = queue.popleft()
        visited.add(frag)

        frag_smi, new_mol = frag_to_smiles(frag)
        neib_list = extract_tags(frag_smi)
        bfs_smiles.append(frag_smi)
        posis = new_mol.GetConformer().GetPositions()
        all_coords.append(posis.tolist())
        for index, atom in enumerate(new_mol.GetAtoms()):
            if atom.GetSymbol() != '*':
                ori_coords.append(list(posis[index]))

        for nbr in frags:
            if nbr not in visited:
                nbr_list = extract_tags(frag_to_smiles(nbr)[0])
                if len(list(set(neib_list) & set(nbr_list))) > 0:
                    queue.append(nbr)

    return '^'.join(bfs_smiles), all_coords, ori_coords


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


def transfer(coords, ori_coords, vertices=None):
    coord_offset = np.array([0., 0., 0.])
    for c in np.array(ori_coords):
        coord_offset += c
    coord_offset /= len(coords)

    # 随机在xyz轴上平移2a以内坐标
    if random.random() > 0.8:
        coord_offset += np.random.rand(3) * 22 - 11

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

    if vertices is not None:
        vertices -= coord_offset
        vertices = np.dot(vertices, rot_matrix)
    
    ret = []
    for c in coords:
        c = np.array(c)
        c -= coord_offset
        c = np.dot(c, rot_matrix)
        ret.append(c.tolist())

    return ret, vertices


def gen_3dsmiles(smiles, process_id='0'):
    if not check_atom_symbol(smiles):
        return []
    
    fragmenizer = BRICS_RING_R_Fragmenizer()
    ori_mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True))
    ori_mol = Chem.RemoveHs(ori_mol)
    isomers = tuple(EnumerateStereoisomers(ori_mol))
    isomers = isomers[-ISO_NUM: ]
    ret_smiles_3d_list = []
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
            try:
                frags, _ = fragmenizer.fragmenize(mol)
                frags = Chem.GetMolFrags(frags, asMols=True)

                smi, all_coords, ori_coords = connect_frags_bfs(frags)

                vertices = generate_vertieces(mol, all_coords, process_id, cube_size=2)
                if vertices is None:
                    try_time -= 1
                    continue

                coords, vertices = transfer(all_coords, ori_coords, vertices)
                coords_str = coord_2_str(coords)
                vertices_str = coord_2_str_v(vertices)

                ret_smiles = vertices_str + '|' + str(qed_v) + '|' + str(logp_v) + '|' + smi + '&' + coords_str

                # print('3dsmiles: {}'.format(tdsmiles))
                ret_smiles_3d_list.append(ret_smiles)
            except:
                print('fragmenize failed: {}'.format(smiles))
                traceback.print_exc()

            try_time -= 1
        
    return ret_smiles_3d_list


def gen_3dsmiles_list(dic):
    smiles_3d_list = []
    for index, smiles in enumerate(dic['lst']):
        try:
            three_dimension_smiles_list = gen_3dsmiles(smiles, process_id=dic['process_id'])
        except:
            three_dimension_smiles_list = []
            traceback.print_exc()

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
    # smiles_list = random.sample(smiles_list, 1000000)
    smiles_list = smiles_list[6000000: 8000000]

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
    smiles_path = 'data/pubchem-10m.txt'
    out_3dsmiles_path = 'data/frag_rotate_iso_68.txt'
    print(smiles_path, out_3dsmiles_path)
    main(smiles_path, out_3dsmiles_path)

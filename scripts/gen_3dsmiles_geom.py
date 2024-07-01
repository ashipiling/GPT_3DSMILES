import msgpack
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
import copy
from rdkit import RDLogger
from multiprocessing import Pool
from timeout_decorator import timeout
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


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


def float_to_str(float_data):
    if float_data > 0:
        return ' ' * (5 - len(str(int(float_data)))) + f"{float_data:.6f}"
    else:
        return ' ' * (4 - len(str(int(abs(float_data))))) + f"{float_data:.6f}"

def dict_to_xyz_string(xyz_data):
    xyz_string = f"{len(xyz_data)}\n\n"
    for atom in xyz_data:
        # atom[0]为浮点数，转为CNOS等原子符号
        atom[0] = Chem.GetPeriodicTable().GetElementSymbol(int(atom[0]))
        # C      3.300000   -1.300000   -3.300000
        string_0 = atom[0] + ' '* (3 - len(atom[0]))
        string_1 = float_to_str(float(atom[1]))
        string_2 = float_to_str(float(atom[2]))
        string_3 = float_to_str(float(atom[3]))
        xyz_string += f"{string_0}{string_1}{string_2}{string_3}\n"
    return xyz_string


def xyz_string_to_mol(xyz_string):
    raw_mol = Chem.MolFromXYZBlock(xyz_string)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol)
    return mol


def encode_number(num):
    if num > 39.9 or num < -39.9:
        print('coords 太大')
        return -1, None

    num = int(round(num * 10, 0))
    prefix = ''
    if abs(num) != num:
        prefix = '-'
    num_str = prefix + '0' * (4 - len(str(abs(num)))) + str(abs(num))

    return 0, num_str


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def out_3dsmiles(relative_str, mol):
    coords = mol.GetConformer().GetPositions()
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    # 生成3dsmiles
    three_dimension_smiles = relative_str + '|' + canonical_smiles
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


@timeout(0.1)
def gen_3dsmiles(dic):
    xyz = dic['xyz']
    relativeenergy = float(dic['relativeenergy'])
    xyz_string = dict_to_xyz_string(xyz)
    mol = xyz_string_to_mol(xyz_string)
    mol = Chem.RemoveHs(mol)
    if relativeenergy > 1.07:
        relativeenergy_str = '0'
    else:
        relativeenergy_str = '1'
    smiles = Chem.MolToSmiles(mol)
    err, tdsmiles = out_3dsmiles(relativeenergy_str, mol)
    if err:
        return []

    return [tdsmiles]


def gen_3dsmiles_list(lst):
    smiles_3d_list = []
    failed_cnt = 0
    for index, dic in enumerate(lst):
        try:
            three_dimension_smiles_list = gen_3dsmiles(dic)
            smiles_3d_list += three_dimension_smiles_list
        except:
            failed_cnt += 1

        # if index % 1000 == 0:
        #     print('已处理{}个smiles'.format(index), failed_cnt)
    return smiles_3d_list


def main(smiles_list, out_3dsmiles_path, n_worker=32):

    max_smiles_string_num = 0
    smiles_string_num_list = []

    # 生成3dsmiles
    smiles_3d_list = []
    # 启用多进程

    with Pool(processes=n_worker) as pool:
        results = pool.map(gen_3dsmiles_list, split_list(smiles_list, n_worker))
    
    smiles_3d_list = [item for sublist in results for item in sublist]

    # smiles_3d_list = gen_3dsmiles_list(smiles_list)
    # for smiles in smiles_3d_list:
    #     smiles_string_num_list.append(len(smiles))
        # if len(smiles) > max_smiles_string_num:
        #     max_smiles_string_num = len(smiles)

    # 打印统计信息
    # print('最大smiles字符串长度: {}'.format(max_smiles_string_num))
    # if len(smiles_string_num_list):
    #     print('平均smiles字符串长度: {}'.format(sum(smiles_string_num_list) / len(smiles_string_num_list)))

    print('3dsmiles数量: {}'.format(len(smiles_3d_list)))

    # 保存3dsmiles
    with open(out_3dsmiles_path, 'a') as f:
        for smiles in smiles_3d_list:
            f.write(smiles + '\n')


not_use_full = True
max_index = 10
relativeenergy_list = []

geom_path = 'data/drugs_crude.msgpack'
out_3dsmiles_path = 'data/3dsmiles_pubchem_10m_38_atom_with_property_geom_not_full_low.txt'
unpacker = msgpack.Unpacker(open(geom_path, "rb"))

for drug_index, drug_dic in enumerate(unpacker):
    if drug_index < 181:
        print('continue', drug_index)
        continue
    xyz_list = []
    for smiles in drug_dic:
        confs = drug_dic[smiles]['conformers']
        for index, conf in enumerate(confs):
            xyz_l = conf['xyz']
            relativeenergy = conf['relativeenergy']
            xyz_list.append({'xyz': xyz_l, 'relativeenergy': relativeenergy})

            if not_use_full and index > max_index:
                break
    print('#######start: ', drug_index)
    main(xyz_list, out_3dsmiles_path)
    print('#######finished: ', drug_index)

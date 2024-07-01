# 该脚本用于将3dsmiles解析为mol文件，并嵌入3d坐标
import copy
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import traceback

# 定义不同元素之间的键长范围
bond_lengths = {
    ('C', 'C'): (1.1, 1.8),
    ('C', 'O'): (1.1, 1.8),
    ('C', 'N'): (1.1, 1.8),
    ('C', 'S'): (1.4, 2.0),
    ('C', 'P'): (1.4, 2.0),
    ('C', 'F'): (1.1, 1.7),
    ('C', 'Cl'): (1.4, 2.0),
    ('C', 'Br'): (1.5, 2.0),
    ('C', 'I'): (1.5, 2.2),
    ('C', 'B'): (1.4, 1.8),
    # 添加其他元素的键长范围
}


def decode_number(number):
    # 将62进制的number转为10进制
    # 62进制: 0~9, a~z, A~Z
    vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    res = 0
    for i in range(len(number)):
        res += vocab.index(number[i]) * (62 ** (len(number) - i - 1))

    if res >= 1922:
        res -= 1922
        res = -res
    res /= 10
    return res


def check_valid_3dsmiles(threed_smiles, have_score=True):
    # CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1&{0505vc{v704v6{vhv4vc{vuv4v8{vDvdvd{vzvmvn{vmvmvs{vdvdvm{va0g01{vm0nv1{vo0A04{vf0F0d{v40y0h{v10l0a{090e0f{0k0j0f{07020k{0hv60p{0ovf0f{0wvo0l{0dvm07{0jvtv5{0pvjvd{0Bvev6{0xv606
    if '|' not in threed_smiles:
        return False
    if have_score:
        mol_str = threed_smiles.split('|')[-2]  # -1
    else:
        mol_str = threed_smiles.split('|')[-1]  # -1
    if not mol_str:
        print('not mol_str')
        return False
    # 校验是否存在 &
    if '&' not in mol_str:
        print('not &')
        return False
    smiles = mol_str.split('&')[0]
    coords_str = mol_str.split('&')[1]
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
        if mol is None:
            print('mol is None')
            return False
    except:
        print('Chem.MolFromSmiles failed', smiles)
        return False
    # 校验是否存在 { }
    if '{' not in threed_smiles:
        print('{ not in threed_smiles')
        return False
    # 校验坐标是否合规
    # 1. { 的数量必须与原子个数一致
    # 3. 每一个{ 后面都要都有三个坐标
    # 4. 坐标是否为数字

    coords_str_list = coords_str.split('{')
    if len(coords_str_list) < mol.GetNumAtoms() + 1:
        print('len(coords_str_list) != mol.GetNumAtoms() + 1')
        return False

    for coord_str in coords_str_list[1:]:
        c_list = coord_str.split(',')
        if len(c_list) != 3:
            return False
        try:
            number1 = float(c_list[0])
            number2 = float(c_list[1])
            number3 = float(c_list[2])
        except:
            return False
        if number1 < -392.2 or number1 > 392.2 or number2 < -392.2 or number2 > 392.2 or number3 < -392.2 or number3 > 392.2:
            print("number too big or too small")
            return False

    return True


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


def is_bond_length_valid(conf, bond):
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    symbol1 = atom1.GetSymbol()
    symbol2 = atom2.GetSymbol()
    length = AllChem.GetBondLength(conf, atom1.GetIdx(), atom2.GetIdx())
    min_length, max_length = bond_lengths.get((symbol1, symbol2), (0.0, float('inf')))
    if min_length <= length <= max_length:
        return True
    else:
        print(f"Invalid bond length ({symbol1}-{symbol2}): {length:.2f} Å")
        return False


def check_bond_lengths(mol):
    conf = mol.GetConformer()
    for bond in mol.GetBonds():
        if not is_bond_length_valid(conf, bond):
            return False
    return True


def transfer_3dsmiles_2_mol(
        threed_smiles, 
        coord_offset=np.array([0.,0.,0.]), 
        rot_matrix=np.eye(3),
        have_score=True,
        if_high=False
    ):
    # 1|1|XXXXX&{000000{000000{000000{000000{000000|CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1&{0505vc{v704v6{vhv4vc{vuv4v8{vDvdvd{vzvmvn{vmvmvs{vdvdvm{va0g01{vm0nv1{vo0A04{vf0F0d{v40y0h{v10l0a{090e0f{0k0j0f{07020k{0hv60p{0ovf0f{0wvo0l{0dvm07{0jvtv5{0pvjvd{0Bvev6{0xv606
    # 校验smiles是否合规
    if if_high:
        div = 50
    else:
        div = 10
    if check_valid_3dsmiles(threed_smiles, have_score=have_score):
        try:
            # 生成mol
            if have_score:
                mol_str = threed_smiles.split('|')[-2]
            else:
                mol_str = threed_smiles.split('|')[-1]
            mol = Chem.MolFromSmiles(mol_str.split('&')[0])
            mol = Chem.RemoveHs(mol)
            # 生成3d坐标
            coords = []
            for coord in mol_str.split('&')[-1].split('{')[1:]:
                c_list = coord.split(',')
                number1 = float(c_list[0]) / div
                number2 = float(c_list[1]) / div
                number3 = float(c_list[2]) / div
                coords.append([number1, number2, number3])
            # 嵌入3d坐标
            rwmol = Chem.RWMol(mol)
            conf = Chem.Conformer(rwmol.GetNumAtoms())

            for atom in rwmol.GetAtoms():
                tmp_coords = np.array(coords[atom.GetIdx()])
                # 根据rot_matrix反向旋转
                inv_rot_matrix = np.linalg.inv(rot_matrix)
                # tmp_coords = np.dot(inv_rot_matrix, tmp_coords) + coord_offset
                tmp_coords = np.dot(tmp_coords, inv_rot_matrix) + coord_offset

                conf.SetAtomPosition(atom.GetIdx(), tmp_coords)

            rwmol.AddConformer(conf)

            return rwmol.GetMol()
        except:
            traceback.print_exc()
            return None

    print('check_valid_3dsmiles failed')
    return None


if __name__ == '__main__':
    import random
    path = 'data/pocketsmiles_crossdocked_with_property_17_no_offset_iso.txt'
    f = open(path, 'r')
    lines = f.readlines()
    rlines = random.sample(lines, 100)
    for i, line in enumerate(rlines):
        threed_smiles = line

        try:
            mol = transfer_3dsmiles_2_mol(
                threed_smiles,
                # have_score=False
            )
            if mol:
                # save mol to sdf file
                writer = Chem.SDWriter('output/pocket_no_offset/test' + str(i) + '.sdf')
                writer.write(mol)
                writer.close()
                rmsd = calc_mol_rmsd(mol)
                print('rmsd:', rmsd)
        except:
            print('failed')
            pass

    # threed_smiles = '&{0039,-0078,0024{0070,-0059,-0023{0041,-0086,-0024{0031,-0087,0019{0015,-0086,0016{-0004,-0088,-0030{0025,-0093,-0025{-0025,-0073,0027{-0010,-0075,0034{0012,-0020,-0067{-0041,-0077,0010{-0032,-0084,-0022{-0023,-0088,-0027{0027,-0067,0042{0004,-0057,0056{-0055,-0003,-0051{-0022,-0016,-0069{-0037,-0019,-0066{-0001,0005,-0075{-0005,0022,-0072{-0050,0018,-0058{-0020,0032,-0071{-0061,0018,-0052{0041,0059,-0049{0054,0041,-0049{0028,0084,0036{0043,0084,-0002{0048,0079,0016{0039,0076,-0034{0042,0080,0033{-0054,0075,-0015{-0053,0065,0031{-0031,0066,-0051{-0063,0060,0025{-0024,0087,-0044{-0069,0063,-0016{0006,0100,0027{-0013,0093,0044{0024,0094,0027{-0067,0005,0050{-0056,0001,0063{-0028,-0010,0076{-0079,0005,0041{-0088,0020,0017{-0086,-0002,0035{-0016,-0015,0077{0011,-0030,0071{0002,-0026,0076{0002,-0040,0069{0069,-0011,-0045{0076,0025,-0014{0074,0022,-0028{0081,0008,-0027{0075,-0003,-0041{0074,-0044,-0033|vina0|qed0|logp0|O=C1CC(=O)N([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)C(=O)N1&{0032,-0046,-0004{0023,-0037,-0005{0010,-0040,-0011{0000,-0031,-0006{-0012,-0034,-0007{0003,-0018,-0001{-0007,-0009,0000{-0007,0000,-0010{-0010,0012,-0005{0001,0021,-0005{0001,0031,0003{-0007,0044,0000{-0021,0039,-0004{0000,0052,-0011{-0007,0053,0012{-0015,0010,0009{-0028,0006,0008{-0006,-0001,0012{-0011,-0008,0024{0016,-0015,0000{0019,-0003,0004{0025,-0024,0000|-6.52'
    # mol = transfer_3dsmiles_2_mol(
    #     threed_smiles,
    #     have_score=True
    # )
    # if mol:
    #     writer = Chem.SDWriter('output/has_iso/test_4.sdf')
    #     writer.write(mol)
    #     writer.close()
    #     rmsd = calc_mol_rmsd(mol)
    #     print('rmsd:', rmsd)

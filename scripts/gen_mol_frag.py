import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import re


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


def check_valid_3dsmiles(threed_smiles):
    if '|' not in threed_smiles:
        return False
    mol_str = threed_smiles.split('|')[-1]

    if not mol_str:
        print('not mol_str')
        return False
    # 校验是否存在 &
    if '&' not in mol_str:
        print('not &')
        return False
    
    smiles = mol_str.split('&')[0]
    coords_str = mol_str.split('&')[1]
    # 检查片段数量一致
    frag_str_list = smiles.split('^')
    coords_str_list = coords_str.split('^')
    if len(frag_str_list) != len(coords_str_list):
        print('length not equal')
        return False
    
    try:
        frag_list = [Chem.RemoveHs(Chem.MolFromSmiles(i)) for i in frag_str_list]
        if None in frag_list:
            print('mol is None')
            return False
    except:
        print('Chem.MolFromSmiles failed', smiles)
        return False

    # 校验坐标是否合规
    all_coords_str_list = coords_str.replace('^', '').split('{')

    frag_num_list = [i.GetNumAtoms() for i in frag_list]
    frag_coords_list = [len(i.split('{')) - 1 for i in coords_str.split('^')]

    for index in range(len(frag_num_list)):
        if frag_num_list[index] != frag_coords_list[index]:
            print('frag_num_list[index] != frag_coords_list[index]')
            return False

    for coord_str in all_coords_str_list[1:]:
        c_list = coord_str.split(',')
        if len(c_list) != 3:
            return False
        try:
            number1 = float(c_list[0])
            number2 = float(c_list[1])
            number3 = float(c_list[2])
        except:
            return False
        if number1 < -199.9 or number1 > 199.9 or number2 < -199.9 or number2 > 199.9 or number3 < -199.9 or number3 > 199.9:
            print("number too big or too small")
            return False

    return True


def merge_single_frag(frag_str, coords_str):
    mol = Chem.MolFromSmiles(frag_str)
    coords = []
    coords_str_list = coords_str.split('{')[1:]
    div = 10
    for coord in coords_str_list:
        c_list = coord.split(',')
        number1 = float(c_list[0]) / div
        number2 = float(c_list[1]) / div
        number3 = float(c_list[2]) / div
        coords.append([number1, number2, number3])

    # 嵌入坐标
    rwmol = Chem.RWMol(mol)
    conf = Chem.Conformer(rwmol.GetNumAtoms())

    for atom in rwmol.GetAtoms():
        tmp_coords = np.array(coords[atom.GetIdx()])
        conf.SetAtomPosition(atom.GetIdx(), tmp_coords)

    rwmol.AddConformer(conf)
    mol = rwmol.GetMol()
    return mol


def merge_frag_str(threed_smiles):
    mol_str = threed_smiles.split('|')[-1]
    frag_str_list = mol_str.split('&')[0].split('^')
    coords_str_list = mol_str.split('&')[1].split('^')

    ret_list = []
    for i in range(len(frag_str_list)):
        frag = merge_single_frag(frag_str_list[i], coords_str_list[i])
        ret_list.append(frag)
    
    return ret_list


def extract_tags(smiles):
    pattern = r'\[(\d+\*)\]'
    tags = re.findall(pattern, smiles)
    return tags


def check_end(frag):
    frag_smi = Chem.MolToSmiles(frag)
    tags_list = extract_tags(frag_smi)
    if len(tags_list):
        return False
    else:
        return True


def find_neibid_and_dummy_id(frag):
    dumy_id = None
    connect_id = None
    for atom in frag.GetAtoms():
        if atom.GetSymbol() == '*':
            dumy_id = atom.GetIdx()
            dummy_atom = atom
            break
    connect_id = dummy_atom.GetNeighbors()[0].GetIdx()

    return connect_id, dumy_id


def get_neiid_by_tag(merged_mol, tag):
    neib_id_0, neib_id_1 = -1, -1
    first = True
    for atom in merged_mol.GetAtoms():
        if atom.GetSmarts() == '[' + tag + ']':
            if first:
                neib_id_0 = atom.GetNeighbors()[0].GetIdx()
                first = False
            else:
                neib_id_1 = atom.GetNeighbors()[0].GetIdx()
                break
        
    return neib_id_0, neib_id_1


def get_id_by_tag(merged_mol, tag):
    for atom in merged_mol.GetAtoms():
        if atom.GetSmarts() == '[' + tag + ']':
            return atom.GetIdx()
    return -1


def combine_two_mols(frag1, frag2, tag):
    merged_mol = Chem.CombineMols(frag1, frag2)
    bind_pos_a, bind_pos_b = get_neiid_by_tag(merged_mol, tag)
    if bind_pos_a == -1 or bind_pos_b == -1:
        return None
    
    # 转换成可编辑分子，在两个待连接位点之间加入单键连接，特殊情形需要其他键类型的情况较少，需要时再修改
    ed_merged_mol = Chem.EditableMol(merged_mol)
    ed_merged_mol.AddBond(bind_pos_a, bind_pos_b, order=Chem.rdchem.BondType.SINGLE)
    # 将图中多余的marker原子逐个移除，先移除marker a
    marker_a_idx = get_id_by_tag(merged_mol, tag)
    if marker_a_idx == -1:
        return None
    ed_merged_mol.RemoveAtom(marker_a_idx)
    # marker a移除后原子序号变化了，所以又转换为普通分子后再次编辑，移除marker b
    temp_mol = ed_merged_mol.GetMol()
    marker_b_idx = get_id_by_tag(temp_mol, tag)
    if marker_b_idx == -1:
        return None
    ed_merged_mol = Chem.EditableMol(temp_mol)
    ed_merged_mol.RemoveAtom(marker_b_idx)
    final_mol = ed_merged_mol.GetMol()

    return final_mol


def combine_mol_frag(frag_with_coords_list):
    init_frag = frag_with_coords_list.pop(0)
    while True:
        frag_smi = Chem.MolToSmiles(init_frag, rootedAtAtom=0, canonical=True)
        tags_list = extract_tags(frag_smi)
        if len(tags_list) == 0 or len(frag_with_coords_list) == 0:
            return init_frag

        for tag in tags_list:
            find_flag = False
            for index, neib_frag in enumerate(frag_with_coords_list):
                if tag in extract_tags(Chem.MolToSmiles(neib_frag, rootedAtAtom=0, canonical=True)):
                    find_flag = True
                    break
            if not find_flag:
                print('cannot find frag')
                return None
            # 匹配成功
            # 进行连接
            connect_frag = frag_with_coords_list.pop(index)
            init_frag = combine_two_mols(init_frag, connect_frag, tag)


def transfer_3dsmiles_2_mol(
        threed_smiles, 
        coord_offset=np.array([0.,0.,0.]), 
        rot_matrix=np.eye(3),
    ):
    if not check_valid_3dsmiles(threed_smiles):
        return None
    # 构建frag_list
    frag_with_coords_list = merge_frag_str(threed_smiles)
    # 持续取片段进行组合
    mol = combine_mol_frag(frag_with_coords_list)
    if mol is None:
        return None
    
    coords = mol.GetConformer().GetPositions()
    inv_rot_matrix = np.linalg.inv(rot_matrix)
    final_coords = np.dot(coords, inv_rot_matrix) + coord_offset

    conf = mol.GetConformer()
    for index, atom in enumerate(mol.GetAtoms()):
        conf.SetAtomPosition(atom.GetIdx(), final_coords[index])
    

    cp_mol = Chem.Mol(mol)
    smi = Chem.MolToSmiles(cp_mol, isomericSmiles=True, rootedAtAtom=0, canonical=True)
    core = Chem.MolFromSmarts(smi)
    match = cp_mol.GetSubstructMatch(core)
    # 根据 match 列表重新排列原子顺序
    # 使用 RenumberAtoms 函数重新排列原子编号
    cp_mol = Chem.RenumberAtoms(cp_mol, match)

    ret_mol = Chem.RWMol(Chem.MolFromSmiles(smi))

    conf = Chem.Conformer(ret_mol.GetNumAtoms())
    coords = cp_mol.GetConformer().GetPositions()

    for atom in ret_mol.GetAtoms():
        tmp_coords = np.array(coords[atom.GetIdx()])
        conf.SetAtomPosition(atom.GetIdx(), tmp_coords)

    ret_mol.AddConformer(conf)

    return ret_mol.GetMol()


if __name__ == '__main__':
    transfer_3dsmiles_2_mol('{0039,-0006,-0011{0030,0012,-0019{-0037,0006,0006{0005,0009,-0034{0022,-0012,-0025{-0024,-0008,-0001{-0013,0002,-0021{0014,0018,0009{-0003,-0016,-0011{0034,-0026,-0009{0037,-0023,0013{0011,-0031,-0008{-0015,-0038,0006{-0005,-0048,0036{0014,-0056,0040{0025,-0065,0016{0013,-0033,0035{0024,-0047,0001{0029,-0047,0030{-0003,-0059,0018{0006,0029,-0045{-0042,0047,-0029{0023,0027,-0033{-0029,0036,-0040{-0013,0049,-0037{-0015,0017,-0035{-0033,0020,-0027{0005,0047,-0022{-0020,0053,-0017{0018,0030,-0010{0009,0042,0010{-0002,0053,-0003{-0019,0046,0010{-0032,0031,0004{-0007,0030,0020{-0027,-0024,0018{0029,0002,0007{-0020,0000,0025{0013,0000,0042{-0013,-0019,0033{-0001,0013,0030{0025,-0007,0027|1|1|C/[NH+]=C(/[3*])NC(=S)N[2*]^c1([2*])ncc([1*])cc1[4*]^C1([3*])CC1^S([1*])(N)(=O)=O^C[4*]&{-0010,-0004,0009{0000,-0008,0000{0007,-0001,-0004{0012,0012,-0003{0015,-0009,-0010{0021,-0020,-0006{0010,0000,-0021{0001,-0005,-0028{0000,0008,0017^{0009,0012,-0006{0007,-0001,-0004{0010,0019,-0018{0006,0029,-0023{-0002,0034,-0013{-0013,0041,-0021{-0004,0029,-0001{0001,0019,0002{0001,0013,0015^{0012,0012,-0003{0007,-0001,-0004{0001,0021,-0013{-0012,0018,-0006^{-0013,0041,-0021{-0002,0034,-0013{-0028,0035,-0019{-0013,0058,-0027{-0013,0033,-0033^{0001,0013,0015{0001,0019,0002')


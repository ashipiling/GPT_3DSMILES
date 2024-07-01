# 该脚本将化学smiles转为3dsmiles
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit import RDLogger
from multiprocessing import Pool

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


MAX_ATOM_NUM = 38
TRY_TIME = 2  # 4

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
    # 62进制: 0~9, a~z, A~Z
    # 将num转为62进制，并消灭正负号
    vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if num < 0:
        num = -num
        num += 1922
    res = ''
    while num > 0:
        res = vocab[num % 62] + res
        num //= 62
    if len(res) == 1:
        res = '0' + res
    elif len(res) == 0:
        res = '00'
    assert len(res) == 2
    return 0, res


def decode_number(num):
    if len(num) != 2:
        return -1, None
    # 62进制: 0~9, a~z, A~Z
    # 将num转为62进制，并消灭正负号
    vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    res = 0
    for i in range(len(num)):
        res += vocab.index(num[i]) * (62 ** (len(num) - i - 1))

    if res >= 1922:
        res -= 1922
        res = -res
    res /= 10
    return 0, res
    


def out_3dsmiles(qed_v, logp_v, mol, pocket='XXXXX&{000000{000000{000000{000000{000000'):
    coords = mol.GetConformer().GetPositions()
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    # 生成3dsmiles
    three_dimension_smiles = str(qed_v) + '|' + str(logp_v) + '|' + pocket + '|' + canonical_smiles
    # atom_symbol_list = ['C', 'c', 'O', 'o', 'N', 'n', 'S', 's', 'P', 'F', 'B', 'I', 'l', 'r']
    # atom_cnt = 0
    # old_index = 0
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
        coord = '{' + x + y + z
        three_dimension_smiles += coord
    # for index, s in enumerate(canonical_smiles):
    #     if s in atom_symbol_list:
    #         if (index != len(canonical_smiles) - 1) and (canonical_smiles[index + 1] == 'l' or canonical_smiles[index + 1] == 'r'):
    #             continue
    #         coord = '{' + str(round(coords[atom_cnt][0], 1)) + ',' + str(round(coords[atom_cnt][1], 1)) + ',' + str(round(coords[atom_cnt][2], 1)) + '}'
    #         three_dimension_smiles += canonical_smiles[old_index: index + 1] + coord
    #         old_index = index + 1
    #         atom_cnt += 1
    return 0, three_dimension_smiles


def gen_3dsmiles(smiles):
    # smiles 过滤
    # 如果包含C O N S P F Cl Br I以外的元素，过滤掉
    # return [smiles3d, smiles3d, ...]
    ret_smiles_3d_list = []
    if check_atom_symbol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
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
            err, tdsmiles = out_3dsmiles(qed_v, logp_v, mol)
            if err:
                try_time -= 1
                continue
            print('3dsmiles: {}'.format(tdsmiles))
            ret_smiles_3d_list.append(tdsmiles)
            try_time -= 1

        return ret_smiles_3d_list
    else:
        return []


def gen_3dsmiles_list(smiles_list):
    smiles_3d_list = []
    for index, smiles in enumerate(smiles_list):
        three_dimension_smiles_list = gen_3dsmiles(smiles)
        smiles_3d_list += three_dimension_smiles_list

        if index % 1000 == 0:
            print('已处理{}个smiles'.format(index))
    return smiles_3d_list


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main(smiles_path, out_3dsmiles_path, n_worker=32):

    max_smiles_string_num = 0
    smiles_string_num_list = []
    # 读取smiles文件
    with open(smiles_path, 'r') as f:
        smiles_list = f.readlines()
    # 生成3dsmiles
    smiles_3d_list = []
    # 启用多进程

    with Pool(processes=n_worker) as pool:
        results = pool.map(gen_3dsmiles_list, split_list(smiles_list, n_worker))
    
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
    smiles_path = 'data/pubchem-10k.txt'
    out_3dsmiles_path = 'data/3dsmiles_pubchem_10m_38_atom_with_property.txt'
    print(smiles_path, out_3dsmiles_path)
    main(smiles_path, out_3dsmiles_path)

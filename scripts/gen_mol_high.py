# 该脚本用于将3dsmiles解析为mol文件，并嵌入3d坐标
import copy
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def check_valid_3dsmiles(threed_smiles, have_score=True):
    # CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1&{0505vc{v704v6{vhv4vc{vuv4v8{vDvdvd{vzvmvn{vmvmvs{vdvdvm{va0g01{vm0nv1{vo0A04{vf0F0d{v40y0h{v10l0a{090e0f{0k0j0f{07020k{0hv60p{0ovf0f{0wvo0l{0dvm07{0jvtv5{0pvjvd{0Bvev6{0xv606
    if '|' not in threed_smiles:
        return False
    if have_score:
        mol_str = threed_smiles.split('|')[-2]  # -1
    else:
        mol_str = threed_smiles.split('|')[-1]  # -1
    if not mol_str:
        return False
    # 校验是否存在 &
    if '&' not in mol_str:
        return False
    smiles = mol_str.split('&')[0]
    coords_str = mol_str.split('&')[1]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
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



def transfer_3dsmiles_2_mol(
        threed_smiles, 
        coord_offset=np.array([0.,0.,0.]), 
        rot_matrix=np.eye(3),
        have_score=True
    ):
    # 1|1|XXXXX&{000000{000000{000000{000000{000000|CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1&{0505vc{v704v6{vhv4vc{vuv4v8{vDvdvd{vzvmvn{vmvmvs{vdvdvm{va0g01{vm0nv1{vo0A04{vf0F0d{v40y0h{v10l0a{090e0f{0k0j0f{07020k{0hv60p{0ovf0f{0wvo0l{0dvm07{0jvtv5{0pvjvd{0Bvev6{0xv606
    # 校验smiles是否合规
    if check_valid_3dsmiles(threed_smiles, have_score=have_score):
        # 生成mol
        if have_score:
            mol_str = threed_smiles.split('|')[-2]
        else:
            mol_str = threed_smiles.split('|')[-1]
        mol = Chem.MolFromSmiles(mol_str.split('&')[0])
        # 生成3d坐标
        coords = []
        for coord in mol_str.split('&')[-1].split('{')[1:]:
            c_list = coord.split(',')
            number1 = float(c_list[0]) / 50
            number2 = float(c_list[1]) / 50
            number3 = float(c_list[2]) / 50
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

    print('check_valid_3dsmiles failed')
    return None


if __name__ == '__main__':
    # 0|0|EEDIDDIDHLILLLAVEFADVNDGDLF&{41,62,23{35,65,-4{-40,28,34{-4,51,-31{-48,21,29{-45,6,28{-10,8,46{0,20,-30{-61,0,10{-69,-8,-11{-66,-5,-38{-37,-31,-35{7,0,-51{-25,-22,21{-44,-41,-35{-12,-41,11{26,-35,-43{38,2,-29{9,-70,-30{53,-64,-9{14,-45,40{56,-28,39{51,14,-11{-3,-12,62{53,-11,52{58,40,27{54,-23,50{-25,-56,-36{-8,-63,-50|CC(=O)NC1C(O)OC(CO)C(OC2OC(CO)C(O)C(O)C2NC(C)=O)C1O&{-16,-55,-1{-20,-41,-3{-31,-37,-1{-11,-34,-9{-12,-20,-11{-10,-17,-26{1,-23,-32{-11,-3,-28{-1,3,-19{-1,18,-21{-2,21,-35{-2,-1,-5{7,4,2{5,17,6{-7,19,12{-10,12,24{-24,11,29{-26,-1,33{1,20,33{3,34,33{15,16,27{26,24,32{18,17,12{28,26,7{30,24,-7{43,32,-12{23,15,-14{14,12,-6{-14,-16,-3{-14,-17,11
    # 0|0|EEDIDDIDHLILLLAVEFADVNDGDLF&{41,62,23{35,65,-4{-40,28,34{-4,51,-31{-48,21,29{-45,6,28{-10,8,46{0,20,-30{-61,0,10{-69,-8,-11{-66,-5,-38{-37,-31,-35{7,0,-51{-25,-22,21{-44,-41,-35{-12,-41,11{26,-35,-43{38,2,-29{9,-70,-30{53,-64,-9{14,-45,40{56,-28,39{51,14,-11{-3,-12,62{53,-11,52{58,40,27{54,-23,50{-25,-56,-36{-8,-63,-50|CC(=O)NC1C(O)OC(CO)C(OC2OC(CO)C(O)C(O)C2NC(C)=O)C1O&{-6,-59,-1{-3,-45,-2{-9,-39,-12{2,-38,8{4,-24,8{17,-22,15{28,-29,8{21,-7,14{21,-2,28{11,-2,34{13,2,3{11,15,3{2,19,-6{-11,20,-3{-18,12,-13{-32,10,-9{-37,23,-5{-15,17,-27{-14,5,-35{-1,21,-30{7,31,-36{4,29,-15{16,35,-14{-1,18,16{-11,27,20{-15,33,32{-24,44,34{-7,30,39{2,-4,-8{-9,-4,-13
    # 0|0|EEDIDDIDHLILLLAVEFADVNDGDLF&{41,62,23{35,65,-4{-40,28,34{-4,51,-31{-48,21,29{-45,6,28{-10,8,46{0,20,-30{-61,0,10{-69,-8,-11{-66,-5,-38{-37,-31,-35{7,0,-51{-25,-22,21{-44,-41,-35{-12,-41,11{26,-35,-43{38,2,-29{9,-70,-30{53,-64,-9{14,-45,40{56,-28,39{51,14,-11{-3,-12,62{53,-11,52{58,40,27{54,-23,50{-25,-56,-36{-8,-63,-50|O=c1c2ccccc2oc2c(Cc3ccc(O)cc3)[nH]c(-c3cccs3)c12&{-15,-24,-18{-5,-22,-12{7,-29,-13{9,-38,-23{21,-45,-24{32,-41,-15{32,-31,-6{20,-25,-5{19,-14,3{8,-7,4{3,3,13{-5,14,8{0,18,-6{-6,28,-14{-19,33,-10{-25,43,-17{-18,48,-28{-23,58,-34{-4,44,-31{1,33,-24{-14,15,17{-9,5,26{-15,3,38{-12,-5,48{0,-13,46{6,-12,33{0,0,17
    threed_smiles = '{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000|qed1|logp1|CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1&{-0140,-0095,0081{-0110,-0051,0031{-0133,0015,0035{-0134,0057,-0021{-0156,0124,-0017{-0177,0149,0044{-0177,0108,0101{-0155,0042,0096{-0086,-0085,-0026{-0117,-0146,-0046{-0099,-0180,-0104{-0049,-0154,-0144{-0016,-0096,-0126{-0034,-0062,-0067{0010,-0005,-0051{0029,0034,-0095{0029,0000,0015{0072,0053,0036{0147,0037,0035{0159,-0014,0083{0170,0014,-0033{0244,0018,-0044{0279,0036,0015{0257,0100,0037{0184,0099,0057'
    mol = transfer_3dsmiles_2_mol(
        threed_smiles, 
        have_score=False
    )
    # save mol to sdf file
    writer = Chem.SDWriter('output/test.sdf')
    writer.write(mol)
    writer.close()
    rmsd = calc_mol_rmsd(mol)
    print('rmsd:', rmsd)

chembl_path = 'data/3dsmiles_pubchem_10m_38_atom_with_property_rotate_iso_2.txt'
chembl_path_2 = 'data/3dsmiles_pubchem_10m_38_atom_with_property_rotate_iso.txt'
cross_path = 'data/pocketsmiles_crossdocked_with_property_17_no_offset_iso.txt'
out_path = 'data/pretrain_finetune_rotate_iso.txt'

ret_list = []
lines = open(chembl_path, 'r').readlines()
lines_2 = open(chembl_path_2, 'r').readlines()
lines_3 = open(cross_path, 'r').readlines()

start = '&'
for line in lines + lines_2:
    strs = line.strip().split('|')
    ret = start + strs[0] + '|' + 'NaN' + '|' + strs[1] + '|' + strs[2] + '|' + strs[3] + '|NaN'
    ret_list.append(ret)


ret_list += lines_3


with open(out_path, 'w') as f:
    for smiles in ret_list:
        f.write(smiles.strip() + '\n')

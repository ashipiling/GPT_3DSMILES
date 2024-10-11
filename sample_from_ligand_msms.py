import os, time, copy
import numpy as np
import torch
import random
import torch.nn as nn
from rdkit import Chem
from multiprocessing import Pool

import smiles_gpt as gpt
from transformers import GPT2Config

from gpt_model import GPT2LMHeadModel
from scripts.gen_mol import transfer_3dsmiles_2_mol, check_bond_lengths, calc_mol_rmsd
from scripts.gen_3dsmiles_3_rotate_iso_msms import generate_vertieces

from scripts.vinascore import Vscore_Reward
# from scripts.remove_clash import remove_clashes
from scripts.remove_clash_new import remove_clashes
from evaluate import common_eval
import argparse  # 导入argparse模块

#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

all_cnt = 0


class SmilesGPTAdapter(nn.Module):
    def __init__(self, checkpoint):
        super(SmilesGPTAdapter, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
        blocks = [b[1] for b in self.model.base_model.iter_layers()]
        self.model.base_model = nn.Sequential(nn.ModuleList(blocks))

    def forward(self):
        pass


def rotate_coord(coords, vertices=None):
    # 找到坐标的中心点
    coord_offset = np.array([0., 0., 0.])
    for c in coords:
        coord_offset += c
    coord_offset /= len(coords)

    # 随机在xyz轴上平移2a以内坐标
    # if random.random() > 0.8:
    #     coord_offset += np.random.rand(3) * 22 - 11

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
    
    return coords, vertices, coord_offset, rot_matrix


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


def add_sugget_coords(mol, coords):
    # 找到距离最远的两个原子
    max_dist = 0
    for i in range(mol.GetNumAtoms()):
        for j in range(i+1, mol.GetNumAtoms()):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist > max_dist:
                max_dist = dist
                atom1 = i
                atom2 = j
    start_coord = coords[0].tolist()
    min_rings_coords = []
    min_rings_coords.append((coords[atom1], np.linalg.norm(coords[atom1] - start_coord)))
    min_rings_coords.append((coords[atom2], np.linalg.norm(coords[atom2] - start_coord)))
    # 计算所有环的坐标中心
    
    rings = Chem.GetSymmSSSR(mol)
    for ring in rings:
        coord = np.array([0.,0.,0.])
        for id in list(ring):
            coord += coords[id]
        coord = coord / len(ring)
        min_rings_coords.append((coord, np.linalg.norm(coord - start_coord)))
    tmp = sorted(min_rings_coords, key=lambda x: x[1])
    ret = [(i[0] + np.random.uniform(-0.1, 0.1, size=3)).tolist() for i in tmp]
    return ret


def gen_prompt_from_ligand(ligand_path, use_msms=True, use_suggest=False):
    mol = Chem.SDMolSupplier(ligand_path)[0]
    # 重新排序
    # 
    cp_mol = Chem.Mol(mol)
    smi = Chem.MolToSmiles(cp_mol, isomericSmiles=True)
    core = Chem.MolFromSmarts(smi)
    match = cp_mol.GetSubstructMatch(core)
    
    # 根据 match 列表重新排列原子顺序
    ans = match
    print('match', match)
    # 使用 RenumberAtoms 函数重新排列原子编号
    cp_mol = Chem.RenumberAtoms(cp_mol, ans)

    vertices = generate_vertieces(cp_mol, remove=False)
    coords = mol.GetConformer().GetPositions()
    coords, vertices, coord_offset, rot_matrix = rotate_coord(coords, vertices)
    prompt = coord_2_str(vertices)
    prompt += '|qed1|logp1|'
    if use_suggest:
        sugget_coords = add_sugget_coords(cp_mol, coords)
        sugget_coords_str = coord_2_str(sugget_coords)
        prompt = sugget_coords_str + '|' + prompt 

    return prompt, coord_offset, rot_matrix, None 


def gen_main(
        n_generated = 5, 
        checkpoint = '',
        max_length = 900,
        ligand_path='',
        pocket_path='',
        device=torch.device('cuda:0'),
        n_worker=32,
        batch_size=32,
        work_dir='',
        use_msms=True,
        pocket_str=None,
        use_suggest=False
    ):
    global all_cnt
    sdf_save_dir = os.path.join(work_dir, 'sdf')
    os.makedirs(sdf_save_dir, exist_ok=True)

    max_loop_cnt = (n_generated // batch_size) * 3

    model_config = GPT2Config.from_pretrained(checkpoint)
    tokenizer_file = os.path.join(checkpoint, "tokenizer.json")
    tokenizer = gpt.SMILESBPETokenizer.get_hf_tokenizer(
            tokenizer_file, 
            model_max_length=model_config.n_positions
        )
    model_config.pad_token_id = tokenizer.pad_token_id

    model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)
    model.eval()

    ligand_mol = Chem.SDMolSupplier(ligand_path)[0]
    pocket_name = pocket_path.split('/')[-2]
    sdf_dir = os.path.join(sdf_save_dir, pocket_name)
    os.makedirs(sdf_dir, exist_ok=True)

    pocket_mol = Chem.MolFromPDBFile(pocket_path)
    # pocket_mol = extract_pocket_mol(pocket_path, ligand_mol=ligand_mol)
    # _x_score = Xscore_Reward(xscore_save_path, xscore_path, pocket_mol, ligand_mol, need_init_score=True)
    _x_score = Vscore_Reward(xscore_save_path, pocket_path, ligand_mol, need_init_score=True, need_clean=True)
    print('init_score', _x_score.score)
    out_loop_cnt = 0
    final_mols_list = []
    use_expand = False

    while len(final_mols_list) < n_generated and out_loop_cnt < max_loop_cnt:
        prompt, coord_offset, rot_matrix, ligand_str = gen_prompt_from_ligand(
            ligand_path, 
            use_msms=use_msms,
            use_suggest=use_suggest
        )
        # prompt, coord_offset, rot_matrix, ligand_str = gen_prompt_from_pocket(
        #     pocket_path, 
        #     ligand_path, 
        #     dist_threshold=5, 
        #     if_gen=True,
        #     ori_xscore=_x_score.score,
        #     use_expand=use_expand,
        #     # ori_xscore=-6,
        #     use_msms=use_msms
        # )
        use_expand = False
        if prompt is None:
            print('next pocket')
            break
        
        print('ligand_mol', Chem.MolToSmiles(ligand_mol), coord_offset, rot_matrix)
        if pocket_str:
            prompt = pocket_str
        print('prompt', prompt)
        
        # 将前缀编码，并将编码后的序列转换为PyTorch张量
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        smiles_start = torch.LongTensor([[1]])
        # 在encoded_prompt的起始位置加个1
        encoded_prompts_list = []
        for i in range(batch_size):
            encoded_prompts_list.append(torch.cat([smiles_start, encoded_prompt], dim=-1))

        encoded_prompts = torch.cat(encoded_prompts_list, dim = 0).to(device)

        out_loop_cnt += 1
        loop_cnt = 0
        generated_mos_list = []
        while len(generated_mos_list) < (n_generated // 3 * 2) and loop_cnt < (max_loop_cnt // 3):
            loop_cnt += 1
            # Generate from "<s>" so that the next token is arbitrary.
            # Get generated token IDs.
            start_time = time.time()
            generated_ids = model.generate(
                    encoded_prompts,
                    max_length=max_length,
                    do_sample=True, 
                    top_p=0.95,
                    # temperature=1.5,
                    pad_token_id=tokenizer.eos_token_id,
                )
            timeuesd = time.time() - start_time
            print('time used', timeuesd)
            # Decode the IDs into tokens and remove "<s>" and "</s>".
            smiles_set = set()
            one_batch_smiles_cnt = 0
            for generated_id in generated_ids:
                generated_smiles = tokenizer.decode(
                        generated_id,
                        skip_special_tokens=True
                    )

                mol = transfer_3dsmiles_2_mol(
                    generated_smiles, 
                    coord_offset=coord_offset, 
                    rot_matrix=rot_matrix,
                    have_score=False
                )
                print('generated_smiles', generated_smiles)
                if mol is not None:
                    generated_mos_list.append({
                            'ligand_mol': copy.deepcopy(mol),
                            'pocket_path': pocket_path,
                            'pocket_mol': pocket_mol,
                            'output': generated_smiles,
                            'cnt': all_cnt
                        })
                    one_batch_smiles_cnt += 1
                    smiles_set.add(Chem.MolToSmiles(mol))
                    all_cnt += 1
            print(one_batch_smiles_cnt, len(smiles_set))
            if timeuesd > 60:
                break

        with Pool(processes=n_worker) as pool:
            results = pool.map(score_func, split_list(generated_mos_list, n_worker))

        dock_results_list = [item for sublist in results for item in sublist]

        for i in dock_results_list:
            i['ligand_path'] = os.path.join(sdf_save_dir, pocket_name, str(i['cnt']) + '.sdf')
            print('smiles: ', i['smiles'], 'score: ', i['xscore'])

        if _x_score.score > -8:
            final_mols_list.extend([item for item in dock_results_list if item['xscore'] < 200])
            # final_mols_list.extend([item for item in dock_results_list if item['xscore'] < -6.6])
            # final_mols_list.extend([item for item in dock_results_list if item['xscore'] < -6.2]) # no pocket
        else:
            final_mols_list.extend([item for item in dock_results_list if item['xscore'] < 200])
            # final_mols_list.extend([item for item in dock_results_list if item['xscore'] < -6.9])
            # final_mols_list.extend([item for item in dock_results_list if item['xscore'] < -6.5]) # no pocket

    return final_mols_list


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    res = []
    for i in range(n):
        l = lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        res.append({'lst': l, 'process_id': i})
    return res


def score_func(dic):
    ret_list = []
    dict_list = dic['lst']
    process_id = dic['process_id']

    xscore_save_path  = 'data/xscore_2/' + str(process_id)
    xscore_path = '/home/luohao/molGen/xscore_v1.3/xscore'
    
    for dic in dict_list:
        tmp_dict = {}
        ligand_mol = dic['ligand_mol']
        pocket_mol = dic['pocket_mol']
        try:
            start_time = time.time()
            ligand_mol = remove_clashes(ligand_mol, dic['pocket_path'], max_translation=3, max_rotation=90, add_fold=1.2, num_trys=16)

        except:
            ligand_mol = dic['ligand_mol']
            pass
        # xscore = Xscore_Reward(xscore_save_path, xscore_path, pocket_mol, ligand_mol, need_init_score=True)
        # xscore = Vscore_Reward(xscore_save_path, dic['pocket_path'], ligand_mol, need_init_score=True)
        tmp_dict['xscore'] = 0
        tmp_dict['mol'] = ligand_mol
        tmp_dict['smiles'] = Chem.MolToSmiles(ligand_mol)
        tmp_dict['pocket_path'] = dic['pocket_path']
        tmp_dict['output'] = dic['output']
        tmp_dict['cnt'] = dic['cnt']
    
        # if not check_bond_lengths(ligand_mol):
        #     print('check_bond_lengths failed!', tmp_dict['smiles'])
        #     continue

        # if calc_mol_rmsd(ligand_mol) == 10:
        #     print('rmsd too big')
        #     continue

        ret_list.append(tmp_dict)

    return ret_list


def eval_main(
        path_dict_list, 
        n_generated=50, 
        checkpoint=None, 
        max_length=1024, 
        save_sdf=False,
        work_dir='',
        device=torch.device('cuda:0'),
        ligand_p=None,
        pocket_p=None,
        use_good_smiles=False,
        use_msms=True,
        pocket_str=None,
        use_suggest=False,
        n_worker=36
    ):

    results_list = []
    now_time_stamp = time.time()

    for index, path_dict in enumerate(path_dict_list):
        if index == 100:
            break
        if pocket_p:
            pocket_path = pocket_p
            ligand_path = ligand_p
        else:
            ligand_path = path_dict['ligand_path']
            pocket_path = path_dict['pocket_path']

        ret_list = gen_main(
            n_generated=n_generated,
            checkpoint=checkpoint, 
            max_length=max_length,
            ligand_path=ligand_path,
            pocket_path=pocket_path,
            device=device,
            work_dir=work_dir,
            n_worker=n_worker,
            use_msms=use_msms,
            pocket_str=pocket_str,
            use_suggest=use_suggest
        )
        try:
            print('index', index, sum([i['xscore'] for i in ret_list]) / len(ret_list))
        except:
            print('index', index, 0, 'xscore error')
        
        if save_sdf:
            for dic in ret_list:
                writer = Chem.SDWriter(dic['ligand_path'])
                writer.write(dic['mol'])
                writer.close()
        
        results_list.extend(ret_list)
    
    print('time cost: ', time.time() - now_time_stamp)

    common_eval(results_list, os.path.join(work_dir, 'output.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate molecules using a pretrained model.')
    parser.add_argument('--task_name', type=str, default='just_test', help='Name of the task')
    parser.add_argument('--ligand_p', type=str, default=None, help='Path to the ligand file')
    parser.add_argument('--pocket_p', type=str, default=None, help='Path to the pocket file')
    parser.add_argument('--n_generated', type=int, default=100, help='Number of molecules to generate')
    parser.add_argument('--n_worker', type=int, default=64, help='Number of worker')
    
    args = parser.parse_args()
    # 生成单个口袋 2qrv
    # pocket_p = 'data/5p9j_blank.pdb'
    # ligand_p = 'data/5p9j.sdf'

    save_sdf = True
    task_name = args.task_name  # pretrain_with_outer_anchor_point
    pocket_p = args.pocket_p
    ligand_p = args.ligand_p
    n_generated = args.n_generated
    n_worker = args.n_worker

    work_dir = os.path.join('output', task_name)
    os.makedirs(work_dir, exist_ok=True)
    # checkpoint='checkpoints/benchmark-14m_19'
    # checkpoint='checkpoints/8gur_finetune'
    checkpoint='checkpoints/pretrain_with_outer_anchor_point_copy/'  #  pocket_no_offset_copy
    xscore_path = '/home/luohao/molGen/xscore_v1.3/xscore'
    xscore_save_path = 'data/xscore_2/'
    max_length = 1024

    n_pocket = 100
    if pocket_p:
        n_pocket = 1
    
    use_msms = True
    use_suggest = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)

    split_path = 'data/split_by_name.pt'
    split_dict = torch.load(split_path)
    data_dir = 'data/cross_docked/'
    path_dict_list = []
    val_path_list = split_dict['test']
    val_path_list = val_path_list[: n_pocket]

    for path in val_path_list:
        ligand_path = os.path.join(data_dir, path[1])
        pocket_path = os.path.join(data_dir, path[0])
        path_dict_list.append({
            'ligand_path': ligand_path,
            'pocket_path': pocket_path
        })

    eval_main(
        path_dict_list, 
        n_generated=n_generated, 
        checkpoint=checkpoint, 
        max_length=max_length, 
        save_sdf=save_sdf,
        work_dir=work_dir,
        device=device,
        pocket_p=pocket_p,
        ligand_p=ligand_p,
        use_msms=use_msms,
        use_suggest=use_suggest,
        n_worker=n_worker
    )

import os, time, copy
import numpy as np
import torch
import random
import torch.nn as nn
from rdkit import Chem
from multiprocessing import Pool

import smiles_gpt as gpt
from transformers import GPT2Config

from smiles_gpt.gpt_model import GPT2LMHeadModel
from scripts.gen_mol import transfer_3dsmiles_2_mol, check_bond_lengths, calc_mol_rmsd
from scripts.gen_pocket_smiles_3_2 import gen_3dsmiles, gen_pocket_str, extract_pocket_mol, get_pocket_vertice
from scripts.xscore import Xscore_Reward
from evaluate import common_eval


os.environ["TOKENIZERS_PARALLELISM"] = "false"

all_cnt = 0
MIN_VERTICE_CNT = 1200
# MIN_VERTICE_CNT = 20

class SmilesGPTAdapter(nn.Module):
    def __init__(self, checkpoint):
        super(SmilesGPTAdapter, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
        blocks = [b[1] for b in self.model.base_model.iter_layers()]
        self.model.base_model = nn.Sequential(nn.ModuleList(blocks))

    def forward(self):
        pass


def gen_prompt_from_pocket(
        pdb_file_path, 
        sdf_file_path, 
        dist_threshold=5.0, 
    ):
    need_pretrain = False

    ligand_mol, property_str, ligand_str, coord_offset, rot_matrix = gen_3dsmiles(sdf_file_path)
    if not ligand_mol:
        print('get ligand_mol gen_3dsmiles failed')
        return None, None, None, None, None
    config = {
        'surface_msms_bin': '/home/luohao/molGen/msms/msms',
        'surface_save_dir': 'data/surface/sample',
        'mesh_threshold': 2
    }
    if not os.path.exists(config['surface_save_dir']):
        os.makedirs(config['surface_save_dir'])

    cube_size = 1

    vertice, errno = get_pocket_vertice(pdb_file_path, ligand_mol.GetConformer().GetPositions(), config=config, cube_size=cube_size)
    
    print('vertice', len(vertice))
    
    if len(vertice) < MIN_VERTICE_CNT:
        need_pretrain = True
    
    if errno:
        print('get vertice error')
        return None, None, None, None, None

    errno, pocket_str = gen_pocket_str(
        vertice, 
        ligand_mol=ligand_mol, 
        coord_offset=coord_offset, 
        dist_t=dist_threshold,
        rot_matrix=rot_matrix,
        use_expand=False,
        ori_xscore=-9
    )

    if errno:
        print('gen_pocket_str error')
        return None, None, None, None, None

    # combined_str = pocket_str + '|vina1|qed1|logp1|'
    combined_str = pocket_str + ''

    return need_pretrain, combined_str, coord_offset, rot_matrix, ligand_str


def gen_main(
        n_generated = 5, 
        pretrain_checkpoint = '',
        finetune_checkpoint = '',
        max_length = 900,
        ligand_path='',
        pocket_path='',
        device=torch.device('cuda:0'),
        n_worker=32,
        batch_size=32,
        work_dir=''
    ):
    global all_cnt
    sdf_save_dir = os.path.join(work_dir, 'sdf')
    os.makedirs(sdf_save_dir, exist_ok=True)

    max_loop_cnt = (n_generated // batch_size) * 3

    model_config = GPT2Config.from_pretrained(finetune_checkpoint)
    tokenizer_file = os.path.join(finetune_checkpoint, "tokenizer.json")
    tokenizer = gpt.SMILESBPETokenizer.get_hf_tokenizer(
            tokenizer_file, 
            model_max_length=model_config.n_positions
        )
    model_config.pad_token_id = tokenizer.pad_token_id

    pretrain_model = GPT2LMHeadModel.from_pretrained(pretrain_checkpoint).to(device)
    pretrain_model.eval()

    finetune_model = GPT2LMHeadModel.from_pretrained(finetune_checkpoint).to(device)
    finetune_model.eval()

    ligand_mol = Chem.SDMolSupplier(ligand_path)[0]
    pocket_mol = extract_pocket_mol(pocket_path, ligand_mol=ligand_mol)
    _x_score = Xscore_Reward(xscore_save_path, xscore_path, pocket_mol, ligand_mol, need_init_score=True)
    print('init_score', _x_score.score)
    out_loop_cnt = 0
    final_mols_list = []
    pretrain_prompt = '{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000|qed1|logp1|'


    while len(final_mols_list) < n_generated and out_loop_cnt < max_loop_cnt:
        need_pretrain, prompt, coord_offset, rot_matrix, ligand_str = gen_prompt_from_pocket(
            pocket_path, 
            ligand_path, 
            dist_threshold=5, 
        )
        if prompt is None:
            print('next pocket')
            break
        
        print('ligand_mol', Chem.MolToSmiles(ligand_mol), coord_offset, rot_matrix)
        print('prompt', prompt, need_pretrain)
        
        # 将前缀编码，并将编码后的序列转换为PyTorch张量
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        smiles_start = torch.LongTensor([[1]])
        # 在encoded_prompt的起始位置加个1
        encoded_prompts_list = []
        for i in range(batch_size):
            encoded_prompts_list.append(torch.cat([smiles_start, encoded_prompt], dim=-1))
        encoded_prompts = torch.cat(encoded_prompts_list, dim = 0).to(device)

        # 将前缀编码，并将编码后的序列转换为PyTorch张量
        pretrain_encoded_prompt = tokenizer.encode(pretrain_prompt, add_special_tokens=False, return_tensors="pt")

        pretrain_encoded_prompts_list = []
        for i in range(batch_size):
            pretrain_encoded_prompts_list.append(torch.cat([smiles_start, pretrain_encoded_prompt], dim=-1))
        pretrain_encoded_prompts = torch.cat(pretrain_encoded_prompts_list, dim = 0).to(device)

        out_loop_cnt += 1
        loop_cnt = 0
        generated_mos_list = []
        while len(generated_mos_list) < (n_generated // 3 * 2) and loop_cnt < (max_loop_cnt // 3):
            if need_pretrain:
                pretrain_generated_ids = pretrain_model.generate(
                    pretrain_encoded_prompts,
                    max_length=max_length,
                    do_sample=True, 
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
                tmp_smiles = []
                for generated_id in pretrain_generated_ids:
                    pretrain_generated_smiles = tokenizer.decode(
                            generated_id,
                            skip_special_tokens=True
                        )
                    mol = transfer_3dsmiles_2_mol(
                        pretrain_generated_smiles, 
                        coord_offset=coord_offset, 
                        rot_matrix=rot_matrix,
                        have_score=False,
                    )
                    if mol is not None:
                        tmp_smiles.append(pretrain_generated_smiles.split('|')[3].split('&')[0])
            
                # 将tmp_smiles扩充到batch_size的长度
                if len(tmp_smiles) < batch_size:
                    # 随机抽取进行补充
                    tmp_smiles.extend(random.choices(tmp_smiles, k=batch_size - len(tmp_smiles)))
                # 将smiles进行encode，tmp_smiles编码后长度不一致，按照最短的长度进行截取
                smiles_prompts = []

                for i in tmp_smiles:
                    encoded_prompt = tokenizer.encode(i, add_special_tokens=False, return_tensors="pt")
                    encoded_prompt = tokenizer.encode(i, add_special_tokens=False, return_tensors="pt")
                    # encoded_prompt shape: [1, len]
                    # 去掉外面的[]，只保留里面的内容
                    encoded_prompt = encoded_prompt[0]
                    smiles_prompts.append(encoded_prompt)

                min_len = max(0, min(1, min([len(i) for i in smiles_prompts])))
                
                smiles_prompts = [i[:min_len] for i in smiles_prompts]
                smiles_prompts = torch.stack(smiles_prompts, dim=0).to(device)
                # smiles_prompts接到finetune的prompt后面
                encoded_prompts = torch.cat([encoded_prompts, smiles_prompts], dim=-1)

            loop_cnt += 1
            # Generate from "<s>" so that the next token is arbitrary.
            # Get generated token IDs.
            start_time = time.time()
            generated_ids = finetune_model.generate(
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
                    rot_matrix=rot_matrix
                )
                # print('generated_smiles', generated_smiles)
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
            i['ligand_path'] = os.path.join(sdf_save_dir, str(i['cnt']) + '.sdf')
            print('smiles: ', i['smiles'], 'score: ', i['xscore'])

        # if _x_score.score > -8:
        #     final_mols_list.extend([item for item in dock_results_list if item['xscore'] < -6.6])
        # else:
        #     final_mols_list.extend([item for item in dock_results_list if item['xscore'] < -6.9])
        
        if _x_score.score > -8:
            final_mols_list.extend([item for item in dock_results_list if item['xscore'] < -1])
        else:
            final_mols_list.extend([item for item in dock_results_list if item['xscore'] < -1])

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
        xscore = Xscore_Reward(xscore_save_path, xscore_path, pocket_mol, ligand_mol, need_init_score=True)
        tmp_dict['xscore'] = xscore.score
        tmp_dict['mol'] = ligand_mol
        tmp_dict['smiles'] = Chem.MolToSmiles(ligand_mol)
        tmp_dict['pocket_path'] = dic['pocket_path']
        tmp_dict['output'] = dic['output']
        tmp_dict['cnt'] = dic['cnt']
    
        if not check_bond_lengths(ligand_mol):
            print('check_bond_lengths failed!', tmp_dict['smiles'])
            continue

        ret_list.append(tmp_dict)

    return ret_list


def eval_main(
        path_dict_list, 
        n_generated=50, 
        pretrain_checkpoint=None, 
        finetune_checkpoint=None,
        max_length=1024, 
        save_sdf=False,
        work_dir='',
        device=torch.device('cuda:0'),
        ligand_p=None,
        pocket_p=None,
    ):

    results_list = []
    now_time_stamp = time.time()

    for index, path_dict in enumerate(path_dict_list):
        if index == 100:
            break

        if pocket_p:
            ligand_path = ligand_p
            pocket_path = pocket_p
        else:
            ligand_path = path_dict['ligand_path']
            pocket_path = path_dict['pocket_path']

        ret_list = gen_main(
            n_generated=n_generated,
            pretrain_checkpoint=pretrain_checkpoint, 
            finetune_checkpoint=finetune_checkpoint,
            max_length=max_length,
            ligand_path=ligand_path,
            pocket_path=pocket_path,
            device=device,
            work_dir=work_dir
        )
        try:
            print('index', index, sum([i['xscore'] for i in ret_list]) / len(ret_list))
        except:
            print('index', index, 0, 'xscore error')
        
        results_list.extend(ret_list)
    
    print('time cost: ', time.time() - now_time_stamp)

    if save_sdf:
        for dic in results_list:
            writer = Chem.SDWriter(dic['ligand_path'])
            writer.write(dic['mol'])
            writer.close()
    
    common_eval(results_list, os.path.join(work_dir, 'output.csv'))


if __name__ == '__main__':
    save_sdf = True

    pocket_p = 'data/pocket_5bvk_7d3i/5bvk_blank.pdb'
    ligand_p = 'data/pocket_5bvk_7d3i/5bvk.sdf'

    task_name = '5bvk_without_two_model'
    work_dir = os.path.join('output', task_name)
    os.makedirs(work_dir, exist_ok=True)

    pretrain_checkpoint='checkpoints/pretrain_copy_0129'
    finetune_checkpoint='checkpoints/pocket_no_offset_copy'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)

    xscore_path = '/home/luohao/molGen/xscore_v1.3/xscore'
    xscore_save_path = 'data/xscore_2/'

    max_length = 1024
    n_generated = 200
    n_pocket = 1

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
        pretrain_checkpoint=pretrain_checkpoint,
        finetune_checkpoint=finetune_checkpoint, 
        max_length=max_length, 
        save_sdf=save_sdf,
        work_dir=work_dir,
        device=device,
        pocket_p=pocket_p,
        ligand_p=ligand_p,
    )

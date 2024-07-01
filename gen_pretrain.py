import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
from rdkit import Chem

import smiles_gpt as gpt
from transformers import GPT2Config

from smiles_gpt.gpt_model import GPT2LMHeadModel
from scripts.gen_mol import transfer_3dsmiles_2_mol, calc_mol_rmsd

IF_HIGH = False

class SmilesGPTAdapter(nn.Module):
    def __init__(self, checkpoint):
        super(SmilesGPTAdapter, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
        blocks = [b[1] for b in self.model.base_model.iter_layers()]
        self.model.base_model = nn.Sequential(nn.ModuleList(blocks))

    def forward(self):
        pass


def main(
        prompt = "{0,0,0{0,0,0{0,0,0{0,0,0{0,0,0|", 
        n_generated = 5, 
        checkpoint = 'checkpoints/benchmark-14m_4',
        max_length = 1024,
        coord_offset = np.array([0.,0.,0.]),
        rot_matrix=np.eye(3),
        xscore=None,
        save_sdf=False,
        task_name='benchmark-14m_5',
        batch_size=32,
        device=torch.device('cuda:0'),
    ):

    model_config = GPT2Config.from_pretrained(checkpoint)
    tokenizer_file = os.path.join(checkpoint, "tokenizer.json")
    tokenizer = gpt.SMILESBPETokenizer.get_hf_tokenizer(
            tokenizer_file, 
            model_max_length=model_config.n_positions
        )
    model_config.pad_token_id = tokenizer.pad_token_id
    model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)

    model_state = model.state_dict()
    keys = model_state.keys()

    model.eval()

    generated_smiles_list = []
    
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    smiles_start = torch.LongTensor([[1]])
    encoded_prompts_list = []

    for i in range(batch_size):
        encoded_prompts_list.append(torch.cat([smiles_start, encoded_prompt], dim=-1))
    
    encoded_prompts = torch.cat(encoded_prompts_list, dim = 0).to(device)
    if not prompt:
        encoded_prompts = encoded_prompts.long()
    # encoded_prompts = torch.cat(encoded_prompts_list, dim = 0)

    count = 0
    rmsd_list = []
    succ_mol_list = []
    succ_cnt = 0
    while succ_cnt < n_generated:
        print('succ_cnt', succ_cnt)

        generated_ids = model.generate(
                encoded_prompts,
                max_length=max_length,
                do_sample=True, 
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        for generated_id in generated_ids:
            generated_smiles = tokenizer.decode(
                    generated_id,
                    skip_special_tokens=True
                )
            print('generated_smiles', generated_smiles)
            mol = transfer_3dsmiles_2_mol(
                generated_smiles, 
                coord_offset=coord_offset, 
                rot_matrix=rot_matrix,
                have_score=False,
                if_high=False
            )
            if mol is not None:
                count += 1
                rmsd = calc_mol_rmsd(mol)
                if rmsd < 10:
                    rmsd_list.append(rmsd)
                    succ_cnt += 1
                    succ_mol_list.append(mol)
                print('smiles', Chem.MolToSmiles(mol))

            generated_smiles_list.append(generated_smiles)

    print('success rate: ', count / len(generated_smiles_list))
    print('rmsd: ', sum(rmsd_list) / len(rmsd_list))

    if save_sdf:
        writer = Chem.SDWriter('output/' + task_name + '.sdf')
        for mol in succ_mol_list:
            writer.write(mol)
        writer.close()

        print('saved sdf file to output/' + task_name + '.sdf')


if __name__ == '__main__':
    save_sdf = True
    # prompt = '{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000|qed1|logp1|'
    # prompt = '&{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000{0000,0000,0000|NaN|qed1|logp1|'
    prompt = ''
    task_name = 'pretrain_with_outer_anchor_point'
    checkpoint='checkpoints/frag_token_copy'
    n_generated = 100
    max_length = 1024
    coord_offset = np.array([0., 0., 0.])
    rot_matrix = np.eye(3)
    dist_threshold = 4.5
    ligand_str = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('prompt: ', prompt + ligand_str)

    main(
        n_generated=n_generated,
        prompt=prompt,
        checkpoint=checkpoint, 
        max_length=max_length,
        coord_offset=coord_offset,
        rot_matrix=rot_matrix,
        xscore=None,
        save_sdf=save_sdf,
        task_name=task_name,
        device=device
    )

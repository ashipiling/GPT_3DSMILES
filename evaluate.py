
import numpy as np 
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
import numpy as np
from copy import deepcopy
import pickle, os
from scripts.sascore import compute_sa_score
import pandas as pd


zinc_path = os.path.join(os.path.dirname(__file__), 'data/filter_smiles.pkl')
zinc_smiles_list = pickle.load(open(zinc_path, 'rb'))


def common_eval(smiles_dict_list, csv_save_path=None):
    smiles_list = [i['smiles'] for i in smiles_dict_list]
    xscore_reward_list = np.array([i['xscore'] for i in smiles_dict_list])
    
    ret = calc_info(smiles_list)
    qed_mean, qed_sd = calc_mean_sd([i['qed'] for i in ret])
    sa_mean, sa_sd = calc_mean_sd([i['sa'] for i in ret])
    lipinski_mean, lipinski_sd = calc_mean_sd([i['lipinski'] for i in ret])
    logp_mean, logp_sd = calc_mean_sd([i['logp'] for i in ret])
    
    if xscore_reward_list is None:
        xscore_mean = 0
        xscore_sd = 0
    else:
        xscore_mean, xscore_sd = calc_mean_sd(xscore_reward_list)

    validity_ratio = check_validity(smiles_list)
    uniqueness_ratio = check_uniqueness(smiles_list)
    novelty_ratio = novelty_metric(zinc_smiles_list, smiles_list)

    ret_dict = {
            'qed_mean': qed_mean, 'qed_sd': qed_sd, 
            'sa_mean': sa_mean, 'sa_sd': sa_sd, 
            'lipinski_mean': lipinski_mean, 'lipinski_sd': lipinski_sd, 
            'logp_mean': logp_mean, 'logp_sd': logp_sd, 
            'xscore_mean': xscore_mean, 'xscore_sd': xscore_sd,
            'validity_ratio': validity_ratio, 
            'uniqueness_ratio': uniqueness_ratio, 'novelty_ratio': novelty_ratio
        }
    
    print('ret_dict', ret_dict)

    if csv_save_path:
        df = pd.DataFrame({
            'smiles': [i['smiles'] for i in smiles_dict_list],
            'score': [i['xscore'] for i in smiles_dict_list],
            'ligand_path': [i['ligand_path'] for i in smiles_dict_list],
            'pocket_path': [i['pocket_path'] for i in smiles_dict_list],
            'qed': [i['qed'] for i in ret],
            'sa': [i['sa'] for i in ret],
            'lipinski': [i['lipinski'] for i in ret],
            'logp': [i['logp'] for i in ret],
        })
        df.to_csv(csv_save_path, index=False)

    return ret_dict


def check_validity(generated_all_smiles):       
    count = 0
    for smiles in generated_all_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            count += 1
    return count / len(generated_all_smiles)


def check_uniqueness(generated_all_smiles):
    original_num = len(generated_all_smiles)
    all_smiles = set(generated_all_smiles)
    new_num = len(all_smiles)
    return new_num / original_num


def novelty_metric(all_smiles, generated_all_smiles):
    total_new_molecules = 0
    for generated_smiles in generated_all_smiles:
        if generated_smiles not in all_smiles:
            total_new_molecules += 1
    
    return float(total_new_molecules) / len(generated_all_smiles)


def calc_info(smiles_list):
    ret_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            ret_list.append({
                'smiles': smiles,
                'qed': qed(mol),
                'sa': compute_sa_score(mol),
                'lipinski': obey_lipinski(mol),
                'logp': get_logp(mol),
            })
        else:
            ret_list.append({
                'smiles': smiles,
                'qed': 0,
                'sa': 0,
                'lipinski': 0,
                'logp': 0,
            })
    print(ret_list[: 10])
    return ret_list


def calc_mean_sd(l):
    mean = np.mean(l)
    sd = np.std(l)
    return mean, sd


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = Crippen.MolLogP(mol)
    rule_4 = (logp>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def get_logp(mol):
    return Crippen.MolLogP(mol)


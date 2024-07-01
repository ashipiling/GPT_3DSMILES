import os
import re
import json
import lmdb
import torch
import numpy as np
from rdkit import Chem
from datasets.tokenizer import *
from datasets.pocket import pdb_to_pocket_data3, pdb_to_pocket_data4
from typing import Optional
import threading, logging
logging.getLogger('prody').setLevel(logging.WARNING)

mapping = {'PAD': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4,
           'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
           'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18,
           'Y': 19, 'V': 20}


class FragSmilesPocketPretrainDataset():
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.data = []
        print('加载数据')

        path_list = torch.load(cfg.DATA.split_path)[mode]

        for index, path in enumerate(path_list):
            pocket_path = os.path.join(cfg.DATA.data_dir, path[0])
            ligand_path = os.path.join(cfg.DATA.data_dir, path[1])
            
            self.data.append({
                'pocket_path': pocket_path,
                'ligand_path': ligand_path,
                'index': index
            })

        self.max_res_seq_len = cfg.DATA.MAX_RES_LEN
        self.radius = 5
        self.num_types = len(mapping)
        self.max_seq_len = cfg.DATA.MAX_SMILES_LEN

        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
            cfg.MODEL.TOKENIZER_PATH,
            model_max_length=cfg.DATA.MAX_SMILES_LEN
        )
        self.end_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.cls_id = self.tokenizer.convert_tokens_to_ids("<cls>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        self.thread_id = threading.current_thread().ident

        print("当前进程号：", os.getpid(), self.thread_id)

    def read_lmdb(self, lmdb_path):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        return env, txn

    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        smiles3d, protein_dict, _, _, _ = pdb_to_pocket_data4(
            data['pocket_path'], 
            data['ligand_path'], 
            threading_id=self.thread_id
        )
        if smiles3d is None:
            return self.__getitem__((item + 73) % len(self))
        
        res_seq = protein_dict['res_seq']
        res_seq = torch.LongTensor([mapping[i] for i in res_seq])
        res_edge_type = res_seq.view(-1, 1) * self.num_types + res_seq.view(1, -1)
        res_dis = self.get_distance_matrix(protein_dict['pkt_node_xyz'])
        res_coords = protein_dict['pkt_node_xyz']

        return {
                'smiles': smiles3d, 
                'res_seq': res_seq, 
                'res_dis': res_dis,
                'res_coords': res_coords, 
                'res_edge_type': res_edge_type
            }

    def get_distance_matrix(self, pos):
        assert pos.shape[1] == 3, 'The shape of pos is error! '
        return torch.pow((pos.unsqueeze(1) - pos.unsqueeze(0)), 2).sum(-1) ** 0.5

    def collator(self, batch):
        new_batch = {}
        smiles_batch = [data['smiles'] for data in batch]
        token_batch = self.tokenizer(smiles_batch, truncation=True)
        new_batch["input_ids"] = self._torch_collate_batch(token_batch['input_ids'], self.tokenizer)
        labels = new_batch["input_ids"].clone()
        labels[labels == self.pad_id] = -100
        new_batch["labels"] = labels

        pad_pocket_seq = []
        pad_pocket_edge_type = []
        pad_pocket_dis = []
        pocket_coords = []
        max_pocket_length = min(self.max_res_seq_len, max([d['res_seq'].shape[0] for d in batch]))
        for d in batch:
            pad_pocket_seq.append(pad_to_max_length_1d(d['res_seq'], max_pocket_length).type(torch.LongTensor).unsqueeze(0))
            pad_pocket_edge_type.append(pad_to_max_length_2d(d['res_edge_type'], max_pocket_length).type(torch.LongTensor).unsqueeze(0))
            pad_pocket_dis.append(pad_to_max_length_2d(d['res_dis'], max_pocket_length).unsqueeze(0))
            pocket_coords.append(pad_to_max_length_1d(d['res_coords'], max_pocket_length).unsqueeze(0))

        new_batch['pocket_seq'] = torch.cat(pad_pocket_seq)
        new_batch['pocket_edge_type'] = torch.cat(pad_pocket_edge_type)
        new_batch['pocket_dis'] = torch.cat(pad_pocket_dis)
        new_batch['pocket_coords'] = torch.cat(pocket_coords)
        return new_batch

    def _torch_collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result


def pad_to_max_length_1d(x, max_length):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    l = x.shape[0]
    new_x = torch.zeros_like(x)
    if l > max_length:
        new_x = new_x[:max_length]
    else:
        x_shape = [i for i in x.shape]
        x_shape[0] = max_length - l
        new_x = torch.cat([new_x, torch.zeros(tuple(x_shape))])
    new_x[:l] = x[:max_length]
    return new_x


def pad_to_max_length_2d(x, max_length):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    l = x.shape[0]
    new_x = torch.zeros((max_length, max_length))
    new_x[:l, :l] = x[:max_length, :max_length]
    return new_x


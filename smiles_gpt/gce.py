# -*- coding:utf-8 -*-
# @Author: jikewang
# @Time: 2024/3/25 20:24
# @File: gce.py

import copy

import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from torch import distributions
import torch.nn.functional as F


def get_all_normal_dis_pdf(voc_len, confs_num):
    means = torch.arange(1, confs_num + 1)  # create x of normal distribution for conf num
    std_dev = 2.0
    normal_dist_list = [distributions.Normal(mean.float(), std_dev) for mean in means]

    # 对数概率密度函数(log PDF)
    pdf_list = []
    zero_pdf = torch.zeros(voc_len)
    zero_pdf[0] = 1
    pdf_list.append(zero_pdf)
    for idx, normal_dist in enumerate(normal_dist_list):
        # if not confs num, make it as 0
        pdf = torch.zeros(voc_len)

        pdf[1:confs_num + 1] = normal_dist.log_prob(means.float()).exp().float()  # 计算 PDF
        # rate of ground truth 50% default is set to 4
        pdf[idx + 1] = pdf[idx + 1] * 2
        # normalized pdf
        normalized_pdf = pdf / pdf.sum()
        # print(normalized_pdf[idx+1])
        pdf_list.append(normalized_pdf)

    return np.array(pdf_list)


def calculate_loss_and_accuracy_confs(logits, batch, pdf_array):
    # Shift so that tokens < n predict n
    labels = batch['input_ids']
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    # need to ignore mask:i, mask 0-5 id: 830 smiles
    # shift_labels
    # covert_maskid_to_padid
    shift_labels_copy = copy.deepcopy(shift_labels)
    shift_labels_copy = shift_labels_copy.masked_fill((shift_labels_copy != 4001), 0)

    shift_labels = shift_labels.masked_fill((shift_labels >= 4000), 0)
    shift_labels_copy_copy = copy.deepcopy(shift_labels)

    shift_labels = shift_labels + shift_labels_copy
    one_hot = F.one_hot(shift_labels, num_classes=5088).float()  # 对标签进行one_hot编码

    non_zero_indices = torch.nonzero(shift_labels_copy_copy)

    # todo speed up this part
    for i in non_zero_indices:
        row = i[0]
        li_index = i[1]
        poisson_one_hot = pdf_array[shift_labels[row][li_index].cpu()]

        one_hot[row][li_index] = poisson_one_hot

    # softmax = torch.exp(shift_logits) / torch.sum(torch.exp(shift_logits), dim=1).reshape(-1, 1)
    # logsoftmax = torch.log(softmax)

    # custom cross entropy loss
    logsoftmax = F.log_softmax(shift_logits, dim=-1)

    # mask 0
    not_ignore = shift_labels.ne(0)
    one_hot = not_ignore.unsqueeze(-1) * one_hot

    # / shift_labels.shape[0]
    loss = -torch.sum(one_hot * logsoftmax)

    # loss_fct = CrossEntropyLoss(ignore_index=0, reduction='sum')
    # loss2 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #
    # print(loss,loss2)

    _, preds = shift_logits.max(dim=-1)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    # rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, accuracy


def calculate_loss_and_accuracy(logits, batch):
    # Shift so that tokens < n predict n
    labels = batch['input_ids']
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()


    shift_labels = shift_labels.masked_fill((shift_labels < 4000), 0)

    loss_fct = CrossEntropyLoss(ignore_index=0, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(0)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy






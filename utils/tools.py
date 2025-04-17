import os
import csv
import json
import copy
import random
import numpy as np
import pandas as pd
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}


#https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L70
def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob
    
    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=0.25)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            inp[sent_tokens_inds] = new_ids[shuffled_inds]
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos


def load_prompts(prompt_path):
    prompts = []
    with open(prompt_path, 'r') as f:
        prompt = ''
        parse_regs = []
        for line in f:
            if line.startswith('# prompt'):
                if prompt.strip():
                    prompts.append((prompt.strip(), parse_regs))
                prompt = ''
                parse_regs = []
            else:
                if not line.startswith('* parse_reg'):
                    prompt += line
                else:
                    parse_regs.append(line.strip().split('@')[1].strip())
        else:
            prompts.append((prompt.strip(), parse_regs))
    return prompts


def cdist_memory_efficient(feats, bsz=64, device=None, topk=50):
    n = feats.shape[0]
    dl_row = DataLoader(TensorDataset(torch.arange(n)), batch_size=bsz, shuffle=False)
    dl_col = DataLoader(TensorDataset(torch.arange(n)), batch_size=bsz, shuffle=False)

    dist = torch.zeros((n, n), dtype=feats.dtype)
    topk_dist, topk_ind = [], []
    for row_batch in tqdm(dl_row, desc="update_dist_topk_row"):
        dist_rows = torch.zeros((row_batch[0].shape[0], n), dtype=feats.dtype).to(device)
        for col_batch in dl_col:
            feats_row = feats[row_batch].to(device)
            feats_col = feats[col_batch].to(device)
            dist_batch = torch.cdist(feats_row, feats_col, 1)
            dist_rows[:, col_batch[0]] += dist_batch
        dist[row_batch[0]] += dist_rows.clone().detach().cpu()
        topk_batch = torch.topk(dist_rows, topk + 1, largest=False, sorted=True)
        topk_dist.append(topk_batch[0].clone().detach().cpu())
        topk_ind.append(topk_batch[1].clone().detach().cpu())
    topk_dist_all = torch.cat(topk_dist)
    topk_ind_all = torch.cat(topk_ind)
    return dist, topk_dist_all, topk_ind_all


def load_prompts_dataset(prompt_path, dataset):
    prompts = []
    with open(prompt_path, 'r') as f:
        prompt = ''
        flag = False
        for line in f:
            if line.startswith(f'# prompt_{dataset}'):
                flag = True
                if prompt.strip():
                    prompts.append((prompt.strip()))
                prompt = ''
            else:
                if line.startswith('# prompt_'):
                    flag = False

                if flag:
                    # prompt += '\n' + line
                    prompt += line
        else:
            prompts.append((prompt.strip()))
    return prompts


def get_adjacency(inds, neighbors, targets, relation=None):
    """get adjacency matrix"""
    adj = torch.zeros(inds.shape[0], inds.shape[0])
    for b1, n in enumerate(neighbors):
        adj[b1][b1] = 1
        for b2, j in enumerate(inds):
            if j in n:
                adj[b1][b2] = 1 # if in neighbors
            if (targets[b1] == targets[b2]) and (targets[b1]>0) and (targets[b2]>0):
                adj[b1][b2] = 1 # if same labels
                # this is useful only when both have labels
    return adj

import numpy as np
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers
import logging
import CONFIG
import os
import sklearn
from tqdm import tqdm
from sklearn import model_selection
from datetime import datetime
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import json
import argparse
from pymongo import MongoClient
import re
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

pad_id = None
text_len_flexible = True
text_len_stable = 50


def device_info(device):
    result = "cpu"
    if torch.cuda.is_available():
        counter = torch.cuda.device_count()
        print("There are {} GPU(s) is available.".format(counter))
        for i in range(counter):
            print("GPU {} Name:{}".format(i, torch.cuda.get_device_name(i)))
        if device == "gpu":
            result = "cuda:0"
            print("We will use {}".format(result))
    return result


def model_paramters_num(model):
    result = 0
    parameters = model.parameters()
    for paramter in parameters:
        result += paramter.numel()
    return result


def load_pickle(path):
    with open(path, 'rb') as fil:
        data = pickle.load(fil)
    return data


def save_pickle(en, path):
    with open(path, 'wb') as fil:
        pickle.dump(en, fil)

def compute_kl_loss(p, q, tokenizer):
    pad_mask = None
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean(dim=1).mean(dim=1)
    q_loss = q_loss.mean(dim=1).mean(dim=1)

    loss = (p_loss + q_loss) / 2
    return loss

class SPLLoss(nn.NLLLoss):
    def __init__(self, *args, n_samples=0, **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)
        self.threshold = 0.5
        #self.threshold_init = 5 #Solve cold start problem.
        self.growing_factor = 4
        self.v = (torch.rand(n_samples) > 0.5).int().cuda() #Solve cold start problem.

    def forward(self, epoch, index, weight, CL, outputs_1, outputs_2, labels, device, tokenizer):
        super_loss = calculate_loss_and_accuracy_stable(weight,CL,outputs_1, outputs_2, labels, device, tokenizer)

        v = self.spl_loss(epoch,super_loss)
        loss = (super_loss * self.v[index]).mean()
        self.v[index] = v
        
        return loss

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, epoch,super_loss):
        #if epoch == 0:
        #    v = super_loss < self.threshold_init
        #else:
        #    v = super_loss < self.threshold
        v = super_loss < self.threshold
        return v.int()

    def save_choice(self, epoch):
        torch.save(self.v, "v"+str(epoch)+".pt")

def calculate_loss_and_accuracy_stable(weight,CL,outputs_1, outputs_2, labels, device, tokenizer):
    lm_logits_1 = outputs_1.logits
    lm_logits_2 = outputs_2.logits
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id,reduction='mean')
    kl_loss = compute_kl_loss(lm_logits_1, lm_logits_2, tokenizer)
    batch_size = lm_logits_1.shape[0]
    #loss_1 = loss_fct(lm_logits_1.view(-1, lm_logits_1.size(-1)), labels.view(-1))
    #loss_2 = loss_fct(lm_logits_2.view(-1, lm_logits_2.size(-1)), labels.view(-1))
    for i in range(batch_size):
        loss_1 = loss_fct(lm_logits_1[i], labels[i])
        loss_2 = loss_fct(lm_logits_2[i], labels[i])
        kl_loss[i] =  (4 * kl_loss[i] + 0.5 * (loss_1+loss_2)) * weight[i] * CL[i]
    #ce_loss = 0.5 * (loss_1+loss_2)

    loss = kl_loss
    
    return loss


def read_data_train(data_dir, tokenizer):
    global pad_id
    assert pad_id is not None
    cfg = CONFIG.CONFIG()
    with open(data_dir, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    final_result = []
    len_all = len(json_data)
    # len_all=500
    for i in tqdm(range(len_all)):

        dic = json_data[i]
        input_text = dic["input_text"]
        label_text = dic["label"]
        input_ids = tokenizer.encode(input_text)
        labels = tokenizer.encode(label_text)
        weight = torch.from_numpy(np.array(len(labels[1:-2])))
        CL = torch.from_numpy(np.array(dic["CL"]))
        if len(input_ids) >= cfg.n_desc:
            input_ids = input_ids[:cfg.n_desc]
        else:
            input_ids.extend([tokenizer.pad_token_id for i in range(cfg.n_desc - len(input_ids))])
        
        if len(labels) >= cfg.n_labels:
            labels = input_ids[:cfg.n_labels]
        else:
            labels.extend([tokenizer.pad_token_id for i in range(cfg.n_labels - len(labels))])
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        final_result.append([input_ids, labels, input_text, weight, CL])

    return final_result


def read_data_test(data_dir, tokenizer):
    global pad_id
    assert pad_id is not None
    cfg = CONFIG.CONFIG()
    with open(data_dir, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    final_result = []
    len_all = len(json_data)
    # len_all=500
    for i in tqdm(range(len_all)):

        dic = json_data[i]
        input_text = dic["input_text"]
        #label_text = dic["label"]
        input_ids = tokenizer.encode(input_text)
        #labels = tokenizer.encode(label_text)
        #if len(input_ids) >= cfg.n_desc:
        #    input_ids = input_ids[:cfg.n_desc]
        #else:
        #    input_ids.extend([tokenizer.pad_token_id for i in range(cfg.n_desc - len(input_ids))])
        
        #if len(labels) >= cfg.n_labels:
        #    labels = input_ids[:cfg.n_labels]
        #else:
        #    labels.extend([tokenizer.pad_token_id for i in range(cfg.n_labels - len(labels))])
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        #labels = torch.tensor(labels, dtype=torch.long)
        final_result.append([input_ids, input_text])

    return final_result


class ClassDataset:
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return item,self.data_list[item]

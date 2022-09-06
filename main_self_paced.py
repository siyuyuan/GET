
import numpy as np
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers
import logging
import CONFIG
import os
import sklearn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import model_selection
from datetime import datetime
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import json
import argparse
import util
from util import ClassDataset, calculate_loss_and_accuracy_stable, SPLLoss
from util import read_data_train, read_data_test
from transformers import T5Tokenizer, T5ForConditionalGeneration
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

logger = None
pad_id = 0
PAD = '[PAD]'


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def create_logger(log_path):
    """
       将日志输出到日志文件和控制台
       """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def train(model, train_dataset, train_loader, device, cfg, tokenizer):
    model.train()
    total_steps = int(train_dataset.__len__() * cfg.epochs / cfg.batch_size)
    logger.info("We will process {} steps.".format(total_steps))
    optimizer = transformers.AdamW(model.parameters(), lr=cfg.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    criterion = SPLLoss(n_samples=len(train_loader.dataset))
    logger.info("starting training.")
    overall_step = 0
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, (index, sample) in enumerate(train_loader):
            # batch_size = len(sample)
            #print(sample[1])
            input_ids = sample[0].to(device)
            labels = sample[1].to(device)
            weight = sample[3].to(device)
            CL = sample[4].to(device)
            # inputs['input_ids'] = inputs['input_ids'].to(device)
            # inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
            # inputs['attention_mask'] = inputs['attention_mask'].to(device)

            outputs_1 = model(input_ids=input_ids, labels=labels)
            outputs_2 = model(input_ids=input_ids, labels=labels)
            loss = criterion(epoch, index, weight, CL, outputs_1,outputs_2,labels=labels,device=device,tokenizer=tokenizer)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                overall_step += 1
            if (overall_step + 1) % cfg.log_step == 0:
                logger.info(
                    "batch {}/{} of epoch {}/{}, loss {}, weight {}".format(batch_idx + 1, train_loader.__len__(),
                    	epoch + 1, cfg.epochs, loss.item(), weight))
        model_path = os.path.join(cfg.saved_model_path, 'model_epoch{}'.format(epoch + 1))
        criterion.increase_threshold()
        criterion.save_choice(epoch)
        logger.info("epoch {} finished.".format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if cfg.save_mode:
            torch.save(model.state_dict(), model_path + '/Prompt_IsA_Model.pth')
        epoch_finish_time = datetime.now()
        logger.info("time for one epoch : {}".format(epoch_finish_time - epoch_start_time))
    logger.info("finished train")



def test(model, test_set, device, cfg, tokenizer):
    with torch.no_grad():
        result = {}
        len_all = len(test_set)
        with open(cfg.result_text ,"w" ,encoding="UTF-8") as f_out:
            for i in tqdm(range(len_all)):
                input_ids = test_set[i][0]
                input_text = test_set[i][1]
                # inputs['input_ids'] = inputs['input_ids'].to(device)
                # inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
                # inputs['attention_mask'] = inputs['attention_mask'].to(device)
                labels = test_set[i][1]
                outputs = model.generate(
                    input_ids=input_ids.unsqueeze(0).to(device),
                    num_beams=cfg.topk,
                    max_length=cfg.n_desc+cfg.n_labels+50,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=cfg.topk,
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_attentions = True)
                dic ={}
                dic["desc" ] = input_text
                dic["preds" ] =[]
                for j in range(cfg.topk):
                    pred = tokenizer.decode(outputs[0][j])
                    score = torch.exp(outputs[1][j])
                    print("desc:{} Concept:{} Score:{:.5f}".format(dic["desc"],pred, score))
                    #print("Concept:{} Score:{:.5f}".format(pred, score))
                    #print("Concept:{}".format(pred))
                    dic["preds"].append([pred,score])
                f_out.write("New data!\n")
                f_out.write("{}\n".format(dic["desc"]))
                for ele in dic["preds"]:
                    f_out.write("{}\t{}\n".format(ele[0],ele[1]))
                f_out.write("\n")




def main():
    global logger
    cfg = CONFIG.CONFIG()
    device = util.device_info(cfg.device)
    print(device)
    logger = create_logger(cfg.log_path)
    tokenizer = T5Tokenizer.from_pretrained(cfg.tokenizer_path)
    vocab_size = len(tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(cfg.pretrained_model_path)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    # device = 'cpu'
    model.to(device)
    # for p in model.topic_embeddings.parameters():
    # p.requires_grad = False
    n_desc = cfg.n_desc
    global pad_id
    #pad_id = tokenizer.convert_tokens_to_ids(PAD)
    util.pad_id = tokenizer.pad_token_id
    if not os.path.exists(cfg.saved_model_path):
        logger.info("build mkdir {}".format(cfg.saved_model_path))
        os.mkdir(cfg.saved_model_path)
    # 获得模型参数个数
    num_parameters = util.model_paramters_num(model)
    logger.info("number of model parameters:{}".format(num_parameters))


    if cfg.train:
        print("***train start****")
        train_set = read_data_train(cfg.train_data_path, tokenizer)
        train_dataset = ClassDataset(train_set)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=cfg.num_workers)
        #model.load_state_dict(torch.load(cfg.c1))
        
        train(model, train_dataset, train_loader, device, cfg, tokenizer)
    if cfg.test:
        print("***test start****")
        test_set = read_data_test(cfg.test_data_path, tokenizer)
        model.load_state_dict(torch.load(cfg.save_model_dic))
        test(model, test_set, device, cfg, tokenizer)


#os.environ["CUDA_VISIBLE_DEVICES" ] ="1"
if __name__ == "__main__":
    print("hll")
    main()

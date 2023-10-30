import os
import sys
import time
import argparse
from datetime import datetime
sys.path.append('D:\code\OD-GPT')

import random
import pickle
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from od_gpt.model import GPT
from od_gpt.trainer import Trainer
from od_gpt.utils import set_seed, setup_logging, CfgNode as CN

from od_gpt import get_config
from od_gpt import CharDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_executions', type=int, default=10, help='number of executions')
    parser.add_argument('--fid', type=int, default=1791)
    parser.add_argument('--his_length', type=int, default=93)
    args = parser.parse_args()
    
    config = get_config()
    # config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    
    seed = int(time.time())  # 使用当前时间的时间戳作为随机种子
    set_seed(seed)
    
    text = []
        
    train_dataset = CharDataset(config.data, text)
    
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    trainer = Trainer(config.trainer, model, train_dataset)
    
    # model.load_state_dict(torch.load('./model/quanzhou/weekday/model100.pt'))
    # model.load_state_dict(torch.load('./model/quanzhou/weekday/model_finetuning_38716.pt'))
    model.load_state_dict(torch.load('./model/quanzhou/weekend/model_finetuning_best.pt'))
    
    model.eval()
    
    num_executions = args.num_executions  # 设置要执行的次数
    # fid = args.fid
    
    # 测试 1
    # test_name = str(config.trainer.file_test)
    # file_test = test_name + ".txt"
    
    # 测试 2
    file_test = "20191002.txt"
    
    # his_length = int(config.trainer.his_length)
    his_length = args.his_length
    
    text = open("./dataset/quanzhou/od_flow/" + file_test, "r").read()
    split_data = text.split("\n")[:-1]
    text = []
    for x in split_data:
        text.append(["start"] + x.split('\t')[:-1] + ["end"])
    
    # filename = f'flow_pre23_nj.txt'
    t = 1.016
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cnt = 0
    print(len(text))
    for historical_data in text:
        cnt += 1
        flag = 1
        while flag:
            
            # with open('./predict/quanzhou/' + str(his_length) + "/" + file_test, 'a') as file:
                
            with open('./predict/quanzhou1/' + str(his_length) + "/" + file_test, 'a') as file:
                
                context_1 = historical_data[:his_length]
                for x in context_1:
                    if x not in train_dataset.stoi:
                        continue
                x = torch.tensor([train_dataset.stoi[x] for x in context_1], dtype=torch.long)[None, ...].to(trainer.device)
                tmp = model.generate(x, 5, temperature=t, do_sample=True, top_k=100)
                y = tmp[0]
                out_data = [train_dataset.itos[int(i)] for i in y]
                if 'start' in out_data[1:] or 'end' in out_data[1:-1]:
                    continue
                completion = ' '.join(out_data[1:-1])
                # print(completion)
                file.write(completion + '\n')
                flag = 0
                if cnt % 1000 == 0:
                    print('已完成：', int(cnt * 1000 / len(text)) / 10 , '%')
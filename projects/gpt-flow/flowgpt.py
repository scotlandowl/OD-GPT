"""
Trains a character-level language model.
"""

import os
import sys
import time
sys.path.append('D:\code\OD-GPT')

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from od_gpt.model import GPT
from od_gpt.trainer import Trainer
from od_gpt.utils import set_seed, setup_logging, CfgNode as CN

import random

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 1028
    # C.system.work_dir = './out/chargpt'
    C.system.work_dir = './model/quanzhou/weekday'
    # C.system.work_dir = './g5/chargpt1'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-flow'
    # C.model.model_type = 'gpt-8001'

    # trainer
    C.trainer = Trainer.get_default_config()
    # C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.trainer.learning_rate = 3e-4
    
    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 98 * 2
        return C

    def __init__(self, config, data):
        self.config = config
        self.data = data
        
        text = ""
        folder_path = "./dataset/quanzhou/od_flow"
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if "weekend" in file_name:
                continue 
            else:
                text += open(file_path, "r").read()
        split_data = text.split("\n")[:-1]
        random.shuffle(split_data)
        text = []
        for x in split_data:
            # 设置2训练长为 96 * 2 的序列，取第二个96，使得初始几个小时随机性降低
            for _ in range(1):
                text.append('start')
                strs = x.split('\t')[:-1]
                text.extend(strs)
                text.append('end')
        data = text

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
        with open(f'stoi.txt', 'w') as file:
            for k, v in self.stoi.items():
                file.write(str(k) + ':' + str(v) + '\n')
        with open(f'itos.txt', 'w') as file:
            for k, v in self.itos.items():
                file.write(str(k) + ':' + str(v) + '\n')
                
        self.vocab_size = vocab_size
        # self.data = data
        
        self.start_indices = [i for i, x in enumerate(self.data) if x == "start"]

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # print(chunk)
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
    
    # def __getitem__(self, idx):
    #     # select the first start index as the beginning of the block
    #     start_index = self.start_indices[idx % len(self.start_indices)]
    #     # grab a chunk of (block_size + 1) characters starting from the selected index
    #     chunk = self.data[start_index:start_index + self.config.block_size]
    #     # print(chunk)
    #     # encode every character to an integer
    #     dix = [self.stoi[s] for s in chunk]
    #     # return as tensors
    #     x = torch.tensor(dix[:-1], dtype=torch.long)
    #     y = torch.tensor(dix[1:], dtype=torch.long)
    #     return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    print(config.trainer.file_name)
    setup_logging(config)
    seed = int(time.time())  # 使用当前时间的时间戳作为随机种子
    set_seed(seed)
    # set_seed(config.system.seed)
    
    # text = ""
    # folder_path = "./dataset/quanzhou/od_flow"
    # for file_name in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, file_name)
    #     if "weekend" in file_name or "test" in file_name:
    #         continue 
    #     else:
    #         text += open(file_path, "r").read()
            
    file_name = str(config.trainer.file_name) + ".txt"
    print("file_name:", file_name)
    folder_path = "./dataset/quanzhou/od_flow/"
    # file_path = os.path.join(folder_path, file_name)
    file_path = folder_path + file_name
    text = open(file_path, "r").read()
            
    split_data = text.split("\n")[:-1]
    random.shuffle(split_data)
    text = []
    for x in split_data:
        # 设置2训练长为 96 * 2 的序列，取第二个96，使得初始几个小时随机性降低
        for _ in range(1):
            text.append('start')
            strs = x.split('\t')[:-1]
            # strs = x.split('\t')
            # strs = strs[16:] + strs[:16]
            text.extend(strs)
            # text.pop()
            text.append('end')

            
    # text = open('input.txt', 'r', encoding = 'utf-8').read()
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # 加载预训练
    pretrained_model_path = './model/quanzhou/weekday/model100.pt'
    pretrained_state_dict = torch.load(pretrained_model_path)
    model.load_state_dict(pretrained_state_dict)
    # # 冻结wte
    # for name, param in model.named_parameters():
    #     if name == 'transformer.wte.weight':
    #         param.requires_grad = False
    model.train()
    
    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 1000 == 0 or trainer.iter_num == 100:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                # context = ["start", "19171".zfill(6)]
                context = ["start", "1791".zfill(6)]
                x = torch.tensor([train_dataset.stoi[x] for x in context], dtype=torch.long)[None,...].to(trainer.device)
                # Top-k采样：设置一个参数k，从概率分布的前k个最高概率的候选词中进行采样。这样可以限制选择范围，并增加模型输出的多样性。
                tmp = model.generate(x, 98 - len(context), temperature=1.01, do_sample=True, top_k=100)
                # print(tmp)
                y = tmp[0]
                completion = ' '.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model" + str(trainer.iter_num) + ".pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()

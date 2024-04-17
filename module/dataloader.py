import os
import pickle
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def read_file(file,analyse=False):
    df = pd.read_excel(file)
    df = shuffle(df)
    if analyse:
        print(f"value counts is {df['class_label'].value_counts()}")
        print(df["class_label"].describe())
    label_id2cate  = dict(enumerate(df.class_label.unique()))
    label_cate2id = {value:key for key,value in label_id2cate.items()}
    df['label'] = df['class_label'].map(label_cate2id)
    return df

class ContentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        text = self.data.loc[item]["content"]
        label = self.data.loc[item]["label"]
        example = self.tokenizer(text, padding='longest')
        
        input_ids = example["input_ids"]
        if len(input_ids) < self.max_len:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
        else:
            input_ids = input_ids[:self.max_len]

        input_ids = torch.tensor(input_ids, dtype=torch.int)

        return {
            'texts': text,
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(data,tokenizer,max_len,batch_size):
    ds = ContentDataset(
        data,
        tokenizer = tokenizer,
        max_len = max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size
    )

if __name__=="__main__":
    file_path = "../train.xlsx"
    df = read_file(file_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '../QwenBase',
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True
    )
    # data = ContentDataset(data=df,tokenizer=tokenizer,max_len=512)
    dataloader = create_data_loader(data=df,tokenizer=tokenizer,max_len=512,batch_size=2)
    



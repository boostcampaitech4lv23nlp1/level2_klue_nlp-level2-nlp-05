import pandas as pd
import torch
from utils.util import label_to_num
from tqdm.auto import tqdm
from ast import literal_eval
import re
from collections import defaultdict


class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = defaultdict()
        item["input_ids"] = self.pair_dataset
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.pair_dataset)


def load_dataset(tokenizer, data_path, conf):
    dataset = pd.read_csv(data_path, index_col=0)
    label = label_to_num(dataset["label"].values)
    tokenized_test = ptuning_tokenized_dataset(dataset, tokenizer)
    RE_dataset = RE_Dataset(tokenized_test, label)
    return RE_dataset


def load_predict_dataset(tokenizer, predict_path, conf):
    predict_dataset = pd.read_csv(predict_path, index_col=0)
    predict_label = None
    tokenized_predict = ptuning_tokenized_dataset(predict_dataset, tokenizer)
    RE_predict_dataset = RE_Dataset(tokenized_predict, predict_label)
    return RE_predict_dataset


def ptuning_tokenized_dataset(dataset, tokenizer):
    template = [3, 2, 3, 2, 3, 4]
    data = []
    for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):

        subj = eval(item["subject_entity"])["word"]
        subj_type = eval(item["subject_entity"])["type"]
        obj = eval(item["object_entity"])["word"]
        obj_type = eval(item["object_entity"])["type"]

        prompt = (
            "<unused1>" * template[0]
            + subj
            + "<unused1>" * template[1]
            + subj_type
            + "<unused1>" * template[2]
            + obj
            + "<unused1>" * template[3]
            + obj_type
            + "<unused1>" * template[4]
            + item["sentence"]
            + "<unused1>" * template[5]
        )

        # input_ids만 전달
        output = tokenizer(prompt, padding=True, truncation=True, max_length=256, add_special_tokens=True, return_tensors="pt").input_ids
        data.append(output)
    return data

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from pathlib import Path
from typing import Dict,  List, Union

from utils import load_data


class TweetDataset(Dataset):
    def __init__(self, data: List[Dict[str, Union[str, int]]], tokenizer: BertTokenizer, max_length=240) -> None:
        super().__init__()
        self.texts = list(map(lambda x: x["text"], data))
        self.tokenizer = tokenizer
        self.targets = list(map(lambda x: x["target"], data))
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        inputs = self.texts[index]
        target = self.targets[index]
        encoding = self.tokenizer.encode_plus(
            inputs,
            add_special_tokens=True,
            max_length=self.max_length,
            padding = 'max_length',
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "inputs": inputs,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "target": torch.tensor(target, dtype=torch.float)
        }


class TweetDataModule(pl.LightningDataModule):
    def __init__(self, data_path: Path, bert_model: str, max_length: int=240, batch_size: int = 1, num_workers=12):
        super(TweetDataModule, self).__init__()
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        train_d, test_d, val_d = load_data(self.data_path)
        self.train_d = train_d
        self.test_d = test_d
        self.val_d = val_d

    def setup(self, stage: str):
        self.train_dataset = TweetDataset(self.train_d, self.tokenizer, self.max_length)
        self.test_dataset = TweetDataset(self.test_d, self.tokenizer, self.max_length)
        self.validation_dataset = TweetDataset(self.val_d, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    datas = TweetDataModule(Path("./clean_data.csv"), "bert-base-cased")
    for data in datas.test_dataset:
        print(data)
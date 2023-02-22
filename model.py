import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import BertModel, AdamW


class SentimentAnalysis(pl.LightningModule):
    def __init__ (self, bert_model: str, num_classes:int = 3, lr: float = 1e-3):
        super(SentimentAnalysis, self).__init__()
        self.lr = lr
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sofmax = nn.Softmax(dim=1)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        temp = self.bert(input_ids, attention_mask)
        out = self.dropout(temp[1])
        out = self.linear(out)
        return out
    
    def configure_optimizers(self):
        optimizer = AdamW(params=self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        input_ids = train_batch["input_ids"]
        attention_mask = train_batch["attention_mask"]
        target = train_batch["target"]
        out = self.forward(input_ids, attention_mask)
        # _, pred_class = torch.max(out, dim=1)
        loss = self.loss_func(out, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input_ids = val_batch["input_ids"]
        attention_mask = val_batch["attention_mask"]
        target = val_batch["target"]
        out = self.forward(input_ids, attention_mask)
        # _, pred_class = torch.max(out, dim=1)
        loss = self.loss_func(out, target)
        self.log('val_loss', loss)


import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertModel, AdamW


class SentimentAnalysis(pl.LightningModule):
    def __init__ (self, bert_model: str, num_classes:int = 3, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sofmax = nn.Softmax(dim=1)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        #self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        temp = self.bert(input_ids, attention_mask)
        out = self.dropout(temp[1])
        out = self.linear(out)
        return self.sofmax(out)

    def _common_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["target"]
        out = self.forward(input_ids, attention_mask)
        #_, pred_class = torch.max(out, dim=0)
        #loss = self.loss_func(out, target)
        #print(out)
        #print(target)
        #print(pred_class)
        #input()
        loss = F.cross_entropy(out, target)

        return loss, out, target
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        loss, pred_class, target = self._common_step(train_batch, batch_idx)
        acc = self.accuracy(pred_class, target)
        f1 = self.f1_score(pred_class, target)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": acc,
                "train_f1_score": f1
            },
            on_epoch=True,
            on_step=False,
            prog_bar=True
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, _, _ = self._common_step(val_batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, _, _ = self._common_step(test_batch, batch_idx)
        self.log('test_loss', loss)
        return loss


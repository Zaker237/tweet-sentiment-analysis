import torch
import pytorch_lightning as pl
from model import SentimentAnalysis
from datamodule import TweetDataModule
from pathlib import Path

DATA_PATH = Path("./clean_data.csv")
BERT_MODEL = "bert-base-cased"
LEARNING_RATE = 1e-3
MAX_LENGTH = 240  # max length of a tweet
NUM_EPOCHS = 3


cls = SentimentAnalysis(
    bert_model=BERT_MODEL,
    num_classes=3,
    lr=LEARNING_RATE
)

data = TweetDataModule(
    data_path=DATA_PATH,
    bert_model=BERT_MODEL,
    max_length=MAX_LENGTH,
    batch_size=4
)

trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
trainer.fit(cls, data)
cls.eval()
trainer.test(data)
trainer.save_checkpoint("trained_model", weights_only=True)
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import SentimentAnalysis
from datamodule import TweetDataModule
from pathlib import Path

torch.set_float32_matmul_precision("medium")


DATA_PATH = Path("./clean_data.csv")
BERT_MODEL = "bert-base-cased"
LEARNING_RATE = 1e-3
MAX_LENGTH = 240  # max length of a tweet
NUM_EPOCHS = 10


cls = SentimentAnalysis(
    bert_model=BERT_MODEL,
    num_classes=3,
    lr=LEARNING_RATE
)

for param in cls.bert.parameters():
    param.requires_grad = False

data = TweetDataModule(
    data_path=DATA_PATH,
    bert_model=BERT_MODEL,
    max_length=MAX_LENGTH,
    batch_size=4,
    num_workers=12
)

logger = TensorBoardLogger("tb_logs", name="tsa_model")

trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    min_epochs=1,
    max_epochs=NUM_EPOCHS,
    precision=16
)
trainer.fit(cls, datamodule=data)
trainer.validate(cls, datamodule=data)
trainer.test(cls, datamodule=data)
trainer.save_checkpoint("trained_model", weights_only=True)
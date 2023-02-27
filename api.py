import os
import torch
from flask import Flask, jsonify, request
from model import SentimentAnalysis
from transformers import BertTokenizer
from utils import clean_text

CHECKPOINT = os.environ.get("BERT_MODEL", "trained_model")
BERT_MODEL = os.environ.get("BERT_MODEL", "bert-base-cased")
MAX_LENGTH = os.environ.get("MAX_LENGTH", 240)

CLASSNAME = {
    0: "Positive",
    1: "Negative",
    2: "Neutral"
}

app = Flask(__name__)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
model = SentimentAnalysis.load_from_checkpoint(CHECKPOINT)
model.eval()

def get_prediction(tweet):
    encoding = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding = 'max_length',
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    inputs = encoding["input_ids"].flatten()
    attention_mask = encoding["attention_mask"].flatten()
    outputs = model.forward(inputs, attention_mask)
    _, pred_class = outputs.max(1)
    details = {
        "Positive": CLASSNAME[outputs[0]],
        "Negative": CLASSNAME[outputs[1]],
        "Neutral": CLASSNAME[outputs[2]]
    }
    class_name = CLASSNAME[pred_class]
    return class_name, details


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        # we will get the text from the request
        tweet = request.data['text']
        # cleaning the tweet
        clean_tweet = clean_text(tweet)
        class_name, details = get_prediction(clean_tweet)
        return jsonify({'prediction': class_name, 'details': details})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

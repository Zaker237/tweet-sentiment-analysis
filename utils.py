import pandas as pd
from bs4 import BeautifulSoup 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopwords = stopwords.words("english")
stopwords.remove('not')
stopwords.remove('no')

# intializing method for lemmatizing words
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # remove any html tags
    new_text = BeautifulSoup(text).get_text()
    
    # remove urls from reviews
    no_urls = new_text.replace('http\S+', '').replace('www\S+', '')
    
    # remove any non-letters
    clean_text = re.sub("[^a-zA-Z]", " ", no_urls)
    
    # convert whole sentence to lowercase and split
    new_words = clean_text.lower().split()
    
    # converting stopwords list to set for faster search
    stops = set(stopwords)
    
    # using stopwords to remove irrelavent words and lemmatizing the final output
    final_words = [lemmatizer.lemmatize(word) for word in new_words if not word in stops]
    
    # return the final result
    return (" ".join(final_words))


def load_data(path):
    pos_data = []
    neg_data = []
    neu_data = []

    with open(path, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            sentiment, text = line[:-1].split(",")
            if sentiment == "Positive":
                pos_data.append({
                    "text": text,
                    "target": 2
                })
            elif sentiment == "Negative":
                neg_data.append({
                    "text": text,
                    "target": 0
                })
            elif sentiment == "Neutral":
                neu_data.append({
                    "text": text,
                    "target": 1
                })

    pos_train_len = int(len(pos_data)*0.8)  # length of positive train data
    neg_train_len = int(len(neg_data)*0.8)  # length of negative train data
    neu_train_len = int(len(neu_data)*0.8)  # length of neutral train data

    pos_test_len  = int((len(pos_data) - pos_train_len) / 2)
    neg_test_len  = int((len(neg_data) - neg_train_len) / 2)
    neu_test_len  = int((len(neu_data) - neu_train_len) / 2)

    pos_train = pos_data[:pos_train_len]
    neg_train = pos_data[:neg_train_len]
    neu_train = pos_data[:neu_train_len]

    pos_test = pos_data[pos_train_len:pos_test_len]
    neg_test = pos_data[neg_train_len:neg_test_len]
    neu_test = pos_data[neu_train_len:neu_test_len]

    pos_val = pos_data[pos_train_len + pos_test_len:]
    neg_val = pos_data[neg_train_len + neg_test_len:]
    neu_val = pos_data[neu_train_len + neu_test_len:]

    train_data = pos_train + neg_train + neu_train
    test_data = pos_test + neg_test + neu_test
    val_data = pos_val + neg_val + neu_val

    return train_data, test_data, val_data

import spacy
import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm
from torch.optim import Adam

from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

train_data = pd.read_parquet('../train.parquet', columns=['context', 'question', 'answerable','lang'], filters=[('lang', 'in', ['ja', 'fi', 'ru'])])
valid_data = pd.read_parquet('../validation.parquet', columns=['context', 'question', 'answerable','lang'], filters=[('lang', 'in', ['ja', 'fi', 'ru'])])


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

train_texts = train_data[(train_data.lang.isin(["ja"]))]['question']
train_labels = train_data[(train_data.lang.isin(["ja"]))]['answerable']
valid_texts = valid_data[(valid_data.lang.isin(["ja"]))]['question']
valid_labels = valid_data[(valid_data.lang.isin(["ja"]))]['answerable']

vectorizer = CountVectorizer(ngram_range=(1,1))
train_features = vectorizer.fit_transform(train_texts)
valid_features = vectorizer.transform(valid_texts)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_features, train_labels)

predictions = classifier.predict(valid_features)

print(classification_report(valid_labels, predictions))

train_texts = train_data[(train_data.lang.isin(["fi"]))]['question']
train_labels = train_data[(train_data.lang.isin(["fi"]))]['answerable']
valid_texts = valid_data[(valid_data.lang.isin(["fi"]))]['question']
valid_labels = valid_data[(valid_data.lang.isin(["fi"]))]['answerable']

vectorizer = CountVectorizer(ngram_range=(1,1))
train_features = vectorizer.fit_transform(train_texts)
valid_features = vectorizer.transform(valid_texts)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_features, train_labels)

predictions = classifier.predict(valid_features)

print(classification_report(valid_labels, predictions))


train_texts = train_data[(train_data.lang.isin(["ru"]))]['question']
train_labels = train_data[(train_data.lang.isin(["ru"]))]['answerable']
valid_texts = valid_data[(valid_data.lang.isin(["ru"]))]['question']
valid_labels = valid_data[(valid_data.lang.isin(["ru"]))]['answerable']

vectorizer = CountVectorizer(ngram_range=(1,1))
train_features = vectorizer.fit_transform(train_texts)
valid_features = vectorizer.transform(valid_texts)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_features, train_labels)

predictions = classifier.predict(valid_features)

print(classification_report(valid_labels, predictions))


train_texts = train_data['context']
train_labels = train_data['answerable']
valid_texts = valid_data['context']
valid_labels = valid_data['answerable']

vectorizer = CountVectorizer(ngram_range=(1,1))
train_features = vectorizer.fit_transform(train_texts)
valid_features = vectorizer.transform(valid_texts)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_features, train_labels)

predictions = classifier.predict(valid_features)

print(classification_report(valid_labels, predictions))
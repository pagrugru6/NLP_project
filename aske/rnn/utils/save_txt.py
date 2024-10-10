import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from spacy.lang.en import English
from transformers import AutoTokenizer
import gensim
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from nlp_lib import *
from bpe.bpe import BPE
from RNNmodel import *
from dataset import SentenceDataset

ds = load_dataset("coastalcph/tydi_xor_rc")
ds_val = ds['validation']
ds_train = ds['train']
ds = ds_val.filter(filter_langs)
ds_ru = ds_train.filter(filter_ru)
ds_ja = ds_train.filter(filter_ja)
ds_fi = ds_train.filter(filter_fi)

# Step 1: Load the dataset
# Replace 'your_dataset.csv' with your actual file path

# Step 2: Extract the column with English sentences
# Replace 'sentence_column' with the actual column name in your dataset
eng_sentences = ds['context']
fi_sentences = ds_fi['question']
ru_sentences = ds_ru['question']
ja_sentences = ds_ja['question']

# Step 3: Write the sentences to a text file

with open('val_en_sentences.txt', 'w', encoding='utf-8') as file:
    for sentence in eng_sentences:
        file.write(f"{sentence}\n")  # Write each sentence on a new line

with open('val_ja_sentences.txt', 'w', encoding='utf-8') as file:
    for sentence in ja_sentences:
        file.write(f"{sentence}\n")  # Write each sentence on a new line
with open('val_ru_sentences.txt', 'w', encoding='utf-8') as file:
    for sentence in ru_sentences:
        file.write(f"{sentence}\n")  # Write each sentence on a new line
with open('val_fi_sentences.txt', 'w', encoding='utf-8') as file:
    for sentence in fi_sentences:
        file.write(f"{sentence}\n")  # Write each sentence on a new line
print("Sentences have been written to output_sentences.txt")

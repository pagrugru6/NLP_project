import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from spacy.lang.en import English
from transformers import AutoTokenizer
from gensim.models import Word2Vec
import gensim
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from nlp_lib import *
from RNNmodel import *
from dataset import SentenceDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = load_dataset("coastalcph/tydi_xor_rc")
ds_val = ds['validation']
ds_train = ds['train']
ds = ds_train.filter(filter_langs)
ds_val = ds_val.filter(filter_langs)

# nlp_en = gen_bpe("train/train_en_sentences.txt", 2000)
test_nlp = gen_bpe("to do", 10)

sentences = ds['context']
nlp_en = BPE(sentences, vocab_size=2000)
nlp_en.load('bpe_model.pkl')
new_sentences = ["hello how are you", "I am fine", "thank you", "this is a sentence on approximately 10 words"]
new_val_sentences = ["and what about you. Have you had a nice weekend?", "I am good", "thanks"]
# val_sentences = ds_val['context'][:3]
vocab, max_len = gen_vocab(sentences, test_nlp)
_, max_len_val = gen_vocab(new_val_sentences, test_nlp)


vocab_size = len(vocab) # +1, otherwise we get an index out of range during embedding
batch_size = 1
embedding_dim = 128
hidden_dim = 256
num_layers = 1
lr = 0.001
epochs = 10

print(f"{vocab=}")
dataset = SentenceDataset(new_sentences, vocab, nlp_en, max_len)
val_dataset = SentenceDataset(new_val_sentences, vocab, nlp_en, max_len_val)
dat_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True).to(device)
val_loader = DataLoader(val_dataset, batch_size = 1).to(device)
model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}:")
    
    train(model, dat_loader, criterion, optimizer, vocab_size)
    validate(model, val_loader, criterion, vocab_size)

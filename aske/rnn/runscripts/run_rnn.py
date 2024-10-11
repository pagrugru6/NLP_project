import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../utils')
sys.path.append('../model')
from tqdm import tqdm
from torch.utils.data import DataLoader
from nlp_lib import *
from RNNmodel import *
from dataset import SentenceDataset

language = "en" # choose betwen en, ru, ja, fi
print(f"training for language {language}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# ds = load_dataset("coastalcph/tydi_xor_rc")
# ds_val = ds['validation']
# ds_train = ds['train']
# ds = ds_train.filter(filter_langs)
# ds_val = ds_val.filter(filter_langs)
file = get_sentence_file(language)
val_file = get_sentence_file(language,val=True)
# nlp = gen_bpe(file, 2000)
# test_nlp = gen_bpe("../../to do", 10)
sentences = []
val_sentences = []
with open(file, "r", encoding="utf-8") as f:
    for line in f:
        sentences.append(line.strip())
with open(val_file, "r", encoding="utf-8") as f:
    for line in f:
        val_sentences.append(line.strip())
print(len(val_sentences))
nlp = get_bpe(sentences, language, vocab_size=2000)
#sentences = ["hello how are you", "I am fine", "thank you", "this is a sentence on approximately 10 words"]
#val_sentences = ["and what about you. Have you had a nice weekend?", "I am good", "thanks"]
# val_sentences = ds_val['context'][:3]
vocab, max_len = gen_vocab(sentences, language, nlp)
print("generated vocab")
# _, max_len_val = gen_vocab(val_sentences, language, nlp)

sentences = sentences# [:9]
val_sentences = val_sentences
vocab_size = len(vocab) # +1, otherwise we get an index out of range during embedding
batch_size = 1
embedding_dim = 128
hidden_dim = 256
num_layers = 1
lr = 0.001
epochs = 2 # more epochs isn't needed it went from loss = 4e-2 to 1e-2 in 6 epochs

dataset = SentenceDataset(sentences, vocab, nlp, max_len)
val_dataset = SentenceDataset(val_sentences, vocab, nlp, max_len)
dat_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 1)
model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

training_loss = []
validation_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}:")
    
    train_loss = train(model, dat_loader, criterion, optimizer, vocab_size, device)
    val_loss = validate(model, val_loader, criterion, vocab_size, device)
    training_loss.append(train_loss)
    validation_loss.append(val_loss)
save_losses(training_loss, validation_loss, language)

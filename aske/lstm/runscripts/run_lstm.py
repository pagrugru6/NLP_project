from datasets import load_dataset, Dataset
import sys
sys.path.append('../utils')
sys.path.append('../model')
from lstm import *
from torch.utils.data import DataLoader
from dataset import QuestionAnswerDataset
from nlp_utils import *

create_dirs()
language = "fi"
print(f"training for language {language}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = get_data_dic(language)
train_list = [dict(zip(train_data.keys(), values)) for values in
                 zip(*train_data.values())]
reduced_train_data = train_list[:1]
val_data = get_data_dic(language, val=True)
val_list = [dict(zip(val_data.keys(), values)) for values in
                 zip(*val_data.values())] 

translated = [train_data['translated'] for train_data in train_list]
context = [train_data['context'] for train_data in train_list]
nlp = get_bpe(translated + context, language, 1000)
vocab, max_len = gen_vocab(translated + context, language, nlp)

batch_size = 1
vocab_size = len(vocab)
embed_size = 100 
hidden_size = 128
epochs = 2
lr = 0.001

val_dataset = QuestionAnswerDataset(val_list[:2], vocab, nlp, max_len)
train_dataset = QuestionAnswerDataset(reduced_train_data, vocab, nlp, max_len)
model = LSTMClassifier(vocab_size, embed_size, hidden_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

training_losses = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()

    train_loss = train(model, train_loader, criterion, optimizer, device)

model.eval()  
validate(model, val_loader, criterion, device)

# missing:
""" Set up validate function in lstm.py
- Save_losses
- moving to cluster
- The evaluation from discord, NOT, use f1-score instead.
ech

"""

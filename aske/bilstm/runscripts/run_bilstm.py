from datasets import load_dataset, Dataset
import sys
sys.path.append('../utils')
sys.path.append('../model')
import os
from torch.optim.lr_scheduler import CyclicLR
from bilstm import *
from nlp_utils import *
from loguru import logger
from torch.utils.data import DataLoader

create_dirs()
# Try to load a DATASET and put in a dataloader with the batch_collect_fn

logger.remove(0)
logger.add(sys.stdout, level="DEBUG")
logger.info('This is a debug message')
language = "fi"
train_data = get_data_dic(language)
val_data = get_data_dic(language, val=True)
train_list = [dict(zip(train_data.keys(), values)) for values in
                 zip(*train_data.values())]
val_list = [dict(zip(val_data.keys(), values)) for values in
                 zip(*val_data.values())]
reduced_train_list = train_list[:1]
reduced_val_list = val_list[:1]
translated = [train_data['translated'] for train_data in train_list]
context = [train_data['context'] for train_data in train_list]
nlp = get_bpe(translated + context, "fi", 1000)
vocab, max_len = gen_vocab(translated + context, "fi", nlp)

def collate_batch_bilstm(input_data):
    context = "hello" # can be deleted once the function is done

    _questions = [encode(nlp.tokenize(dic['translated']), vocab) for dic in input_data]
    _contexts = [encode(nlp.tokenize(dic['context']), vocab) for dic in input_data]

    _labels = [encode(nlp.tokenize(dic['answer']), vocab) for dic in input_data]

    # We have to input the question with the context
    _input = [_q + _c for _q, _c in zip(_questions, _contexts)]

    # seq_lens used by the pack_padded_sequence function in the bilstm model to
    # remove paddding while processing. We must define how much we have padded
    # which is done by seq_lens
    seq_lens = [len(ids) for ids in _input] 
    
    labels = [compute_labels(c, l, vocab) for c,l in zip(_contexts, _labels)]

    # Currently finds the longs context
    max_len = max([len(c) for c in _input])

    # 0 is the id of the padding token
    padded_q_c = [(q_c + [2] * (max_len - len(q_c))) for q_c in _input]# 2 is the id of the <PAD> token


    # 2 is the id of the <PAD> token but we don't want to pad with 2 in the labels as
    # they aren't passed to the model
    final_labels = [(i + [0] * (max_len - len(i))) for i in labels]     
    assert all(len(x) == max_len for x in padded_q_c)
    return torch.tensor(padded_q_c), torch.tensor(seq_lens), torch.tensor(final_labels), torch.tensor(_contexts)

train_dl = DataLoader(reduced_train_list, batch_size=1, shuffle=False,
                        collate_fn=collate_batch_bilstm)
valid_dl = DataLoader(reduced_val_list, batch_size=2, shuffle=False,
                        collate_fn=collate_batch_bilstm)
lstm_dim = 128
dropout_prob = .1
batch_size = 8
lr = 1e-2
n_epochs = 1
n_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM(len(vocab), lstm_dim, dropout_prob, n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CyclicLR(optimizer, base_lr=0., max_lr=lr, step_size_up=1,
                     step_size_down=len(train_dl)*n_epochs,
                     cycle_momentum=False)
losses, learning_rates = train(model, train_dl, valid_dl, optimizer, n_epochs,
                               device, scheduler)
F1 = evaluate(model, valid_dl, device)
model.load_state_dict(torch.load('best_model'))

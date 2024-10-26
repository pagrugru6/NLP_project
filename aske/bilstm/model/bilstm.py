from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from loguru import logger
from nlp_utils import *
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F


class BiLSTM(nn.Module):
    # BiLSTM-CRF model
    def __init__(self, vocab_size: int, lstm_dim: int, dropout_prob: float = 0.1, n_classes: int = 2):
        super(BiLSTM, self).__init__()
        self.model = nn.ModuleDict({
            # vocab_size is the number of unique words in the vocabulary and
            # must be used in a embedding layer as it tells the layer how many
            # different embeddings must exist
            'embedding': nn.Embedding(vocab_size, lstm_dim),
            'bilstm': nn.LSTM(
                lstm_dim, 
                lstm_dim, 
                2,
                batch_first=True, 
                bidirectional=True, 
                dropout=dropout_prob),
            'ff': nn.Linear(lstm_dim * 2, n_classes)
            })
        self.n_classes = n_classes
        self.loss = nn.CrossEntropyLoss()

        self._init_weights()

    def _init_weights(self):
        all_params = list(self.model['bilstm'].named_parameters()) + \
                     list(self.model['ff'].named_parameters())
        for n, p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, inputs, input_lens,  labels = None):
        embeds = self.model['embedding'](inputs)
        
        # pack_padde_sequence is used to handle variable length sequences that
        # has been padded to the same length
        lstm_in = nn.utils.rnn.pack_padded_sequence(
            embeds,
            input_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )


        lstm_out, hidden = self.model['bilstm'](lstm_in) 

        lstm_out, lengths = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                             batch_first=True,
                                                             padding_value=2.0)

        logits = self.model['ff'](lstm_out)
        outputs = (logits, lengths)

        if labels is not None:
            loss = self.loss(logits.reshape(-1, self.n_classes),
                             labels.reshape(-1))
            outputs = outputs + (loss,)

        return outputs

def train(model, train_dl, valid_dl, optimizer, n_epochs, device, scheduler):
    losses = []
    learning_rates = []
    best_f1 = 0

    for epoch in range(n_epochs):
        loss_epoch = []
        for batch in tqdm(train_dl):
            model.train()
            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            inputs, seq_lens, labels, _ = batch

            logits, lengths, loss = model(inputs, seq_lens, labels)
            losses.append(loss.item())
            loss_epoch.append(loss.item())

            loss.backward()

            optimizer.step()
            if scheduler != None:
                scheduler.step()
                learning_rates.append(scheduler.get_last_lr()[0])

        f1 = evaluate(model, valid_dl, device)
        logger.info(f'Validation F1: {f1}, train loss: {sum(loss_epoch) / len(loss_epoch)}')

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model')

    return losses, learning_rates


def evaluate(model, valid_dl, device):
    model.eval()
    labels_all = []
    preds_all = []

    with torch.no_grad():
        for batch in tqdm(valid_dl, desc='Evaluation'):
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            seq_lens = batch[1]
            labels = batch[2]
            hidden_states = None

            logits, _, _ = model(input_ids, seq_lens, labels)
            preds_all.extend(torch.argmax(logits,
                                          dim=-1).reshape(-1).detach().cpu().numpy())
            logger.debug(f"Preds: {preds_all}")
            labels_all.extend(labels.reshape(-1).detach().cpu().numpy())
            print_actual_answer(batch[3], labels[0])
    P, R, F1, _ = precision_recall_fscore_support(labels_all, preds_all,
                                                average='macro')
    logger.info(f"\n Confusion matrix: \n {confusion_matrix(labels_all,preds_all)}")
    return F1

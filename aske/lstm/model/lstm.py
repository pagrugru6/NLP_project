import torch.nn as nn
import torch
from tqdm import tqdm 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(2*hidden_size+1, 2)

    def forward(self, context, question, word_overlap):
        context_embedded = self.embedding(context)
        question_embedded = self.embedding(question)
        _, (context_lstm_out, _) = self.lstm(context_embedded)
        _, (question_lstm_out, _) = self.lstm(question_embedded)
        lstm_out = torch.cat((context_lstm_out[-1], question_lstm_out[-1],
                              word_overlap.float().view(-1, 1)), dim=1)
        logits = self.fc(lstm_out)
        
        return logits

def train(model, data_loader, criterion, optimizer, device):
    total_loss = 0
    for batch in tqdm(data_loader):
        question, context, word_overlap, label = batch
        question = question.to(device)
        context = context.to(device)
        word_overlap = word_overlap.to(device)
        label = label.to(device)


        optimizer.zero_grad()
        logits = model(context, question, word_overlap)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    predictions, true_labels = [], []

    with torch.no_grad(): 
        for batch in data_loader:
            context, question, word_overlap, labels = batch
            context = context.to(device)
            question = question.to(device)
            labels = labels.to(device)
            word_overlap = word_overlap.to(device)

            # Get model outputs
            outputs = model(context, question, word_overlap)
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(predicted_labels)
            true_labels.extend(labels.cpu().numpy())

    # Compute F1 score
    P, R, F1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    print(f"F1 Score: {F1}")
    print(f"P score: {P}")
    print(f"R score: {R}")
    print(f"Confusion matrix: {confusion_matrix(true_labels, predictions)}")
    return F1


import torch.nn as nn
import torch
from tqdm import tqdm 
from sklearn.metrics import f1_score
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
    print(len(data_loader))
    for batch in tqdm(data_loader):
        question, context, word_overlap, label = batch
        question = question.to(device)
        context = context.to(device)
        word_overlap = word_overlap.to(device)
        label = label.to(device)


        optimizer.zero_grad()
        logits = model(context, question, word_overlap)
        print(logits)
        print(label)
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
            print(f'{labels=}')
            word_overlap = word_overlap.to(device)

            # Get model outputs
            outputs = model(context, question, word_overlap)
            print(f'{outputs=}')
            print(f'{labels=}')
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()  # Convert to predicted labels
            print(f'{predicted_labels=}')
            predictions.extend(predicted_labels)
            true_labels.extend(labels.numpy())  # True labels (from DataLoader)

    print(f'{true_labels=}')
    print(f'{predictions=}')
    # Compute F1 score
    f1 = f1_score(true_labels, predictions, average='binary')  # 'binary' for 0 or 1 classification
    print(f"F1 Score: {f1}")
    return f1


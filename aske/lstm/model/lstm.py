import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + 1, 2)  # +1 for the word overlap feature

    def forward(self, context, question, word_overlap):
        # Get embeddings for context and question
        context_embedded = self.embedding(context)
        question_embedded = self.embedding(question)

        # LSTM expects input in the form (batch, seq_len, input_size)
        _, (context_lstm_out, _) = self.lstm(context_embedded)
        _, (question_lstm_out, _) = self.lstm(question_embedded)

        # Use the last hidden state from LSTM
        lstm_out = torch.cat((context_lstm_out[-1], question_lstm_out[-1], word_overlap.float().view(-1, 1)), dim=1)

        # Pass through the final layer
        logits = self.fc(lstm_out)
        return logits


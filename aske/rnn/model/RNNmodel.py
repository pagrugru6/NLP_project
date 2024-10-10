import torch
import torch.nn as nn
# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # Embed input words
        embedded = self.embedding(x)
        # Pass through RNN
        out, hidden = self.rnn(embedded, hidden)
        # Linear layer to get vocab scores for each time step
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)


def train(model, dat_loader, criterion, optimizer, vocab_size):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dat_loader):
        inputs, targets = inputs, targets
        batch_size = inputs.size(0)
        
        # Initialize hidden state
        hidden = model.init_hidden(batch_size)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, hidden = model(inputs, hidden)
        # Reshape outputs to (batch_size * sequence_length, vocab_size)
        outputs = outputs.reshape(-1, vocab_size)
        
        # Reshape targets to (batch_size * sequence_length)
        targets = targets.reshape(-1)

        # Compute loss
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(dat_loader)
    print(f"Training Loss: {avg_loss}")

# Validation loop
def validate(model, dat_loader, criterion, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dat_loader):
            inputs, targets = inputs, targets
            batch_size = inputs.size(0)
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size)
            
            # Forward pass
            outputs, hidden = model(inputs, hidden)
            
            # Reshape outputs to (batch_size * sequence_length, vocab_size)
            outputs = outputs.reshape(-1, vocab_size)
            
            # Reshape targets to (batch_size * sequence_length)
            targets = targets.reshape(-1)
            
            # Compute loss
            try:
                loss = criterion(outputs, targets)
            except:
                breakpoint()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dat_loader)
    print(f"Validation Loss: {avg_loss}")

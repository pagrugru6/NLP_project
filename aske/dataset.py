from torch.utils.data import DataLoader, Dataset
import torch


class SentenceDataset(Dataset):
    def __init__(self, sentences, vocab, nlp, max_len=None):
        self.sentences = sentences
        self.vocab = vocab
        self.max_len = max_len
        self.nlp = nlp
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokenized = []
        for word in self.nlp.tokenize(sentence):
            if str(word) in self.vocab:

                tokenized.append(self.vocab[str(word)])
            else:
                tokenized.append(self.vocab['<UNK>'])
        tokenized.append(self.vocab['<EOS>'])
        # tokenized = [self.vocab[str(word)] for word in self.nlp(sentence)]

        padded_input = tokenized + [self.vocab['<PAD>']] * (self.max_len - len(tokenized)) 
        
        # Create target (shifted sequence)
        target = tokenized[1:] + [self.vocab['<PAD>']] * (self.max_len - len(tokenized) + 1) 
        
        return torch.tensor(padded_input), torch.tensor(target)

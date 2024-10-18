from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

class QuestionAnswerDataset(Dataset):
    def __init__(self, data, vocab, nlp, max_len):
        self.vocab = vocab
        self.nlp = nlp
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['translated'] 
        context = self.data[idx]['context']
        label = self.data[idx]['answerable']
        word_overlap = self.data[idx]['word_overlap']
        tok_question = []
        for word in self.nlp.tokenize(question):
            if str(word) in self.vocab:
                tok_question.append(self.vocab[str(word)])
            else:
                tok_question.append(self.vocab['<UNK>'])
        tok_question.append(self.vocab['<EOS>'])
        tok_question = tok_question + [self.vocab['<PAD>']] * (self.max_len -
                                                               len(tok_question))
        tok_context = []
        for word in self.nlp.tokenize(context):
            if str(word) in self.vocab:
                tok_context.append(self.vocab[str(word)])
            else:
                tok_context.append(self.vocab['<UNK>'])
        tok_context.append(self.vocab['<EOS>'])
        tok_context = tok_context + [self.vocab['<PAD>']] * (self.max_len -
                                                             len(tok_context))
        return torch.tensor(tok_question), torch.tensor(tok_context), torch.tensor(word_overlap), torch.tensor(int(label))

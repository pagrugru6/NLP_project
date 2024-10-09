import torch.nn as nn
import time
import torch
from bpe import BPE
def map_word_to_index(sent, word_to_ix):
    #Assigns a unique index to every word in the corpus
    for word in sent:
        if str(word) not in word_to_ix:
            word_to_ix[str(word)] = len(word_to_ix)
    return word_to_ix

def filter_langs(example):
    return example['lang'] in ['ru', "ja", "fi"]

def filter_ru(ds):
    return ds['lang'] in ['ru']

def filter_ja(ds):
    return ds['lang'] in ['ja']

def filter_fi(ds):
    return ds['lang'] in ['fi']

def keep_columns(ds):
    keep = ['question', 'context', 'answerable']
    all_columns = ds.column_names
    remove_columns = [col for col in all_columns if col not in keep]
    ds.remove_columns(remove_columns)
    return ds

def gen_vocab(sentences, nlp):
    vocab = {}
    max_len = 0
    for sent in sentences:
        doc = nlp.tokenize(sent)
        max_len = max(max_len, len(doc)) + 1 # +1 for <EOS>
        vocab = map_word_to_index(doc, vocab)
    vocab['<UNK>'] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    vocab['<SOS>'] = len(vocab)
    vocab['<EOS>'] = len(vocab)
    return vocab, max_len

def gen_bpe(text_file, vocab_size):
    # computing time spent on this function
    start = time.time()
    with open(text_file, encoding="utf8") as f:
        corpus = f.readlines()
    MyBPE = BPE(corpus=corpus, vocab_size=vocab_size)
    MyBPE.train()
    end = time.time()
    print(f"Time spent on BPE training: {end - start}")
    return MyBPE

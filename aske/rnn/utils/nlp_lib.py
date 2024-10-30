import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
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

def gen_vocab(sentences, language, vocab_size, nlp):
    filepath = f"../checkpoint/{language}_{vocab_size}_vocab.pkl"
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            vocab, max_len = pickle.load(f)
        return vocab, max_len 
    else:
        vocab = {}
        max_len = 0
        print("Generating vocab")

        for sent in tqdm(sentences):
            doc = nlp.tokenize(sent)
            max_len = max(max_len, len(doc)) + 1
            vocab = map_word_to_index(doc, vocab)
        vocab['<UNK>'] = len(vocab)
        vocab['<PAD>'] = len(vocab)
        vocab['<SOS>'] = len(vocab)
        vocab['<EOS>'] = len(vocab)
        with open(filepath, "wb") as f:
            pickle.dump((vocab, max_len), f)
        return vocab, max_len

def get_sentence_file(language, val=False):
    if val:
        return f"../validation/val_{language}_sentences.txt"
    else:
        return f"../train/train_{language}_sentences.txt"

def get_bpe(sentences, language, vocab_size):
    my_file = '../checkpoint/bpe_model_{language}_{vocab_size}.pkl'
    nlp = BPE(corpus=sentences, vocab_size=vocab_size)
    if os.path.exists(my_file):
        nlp.load(f"{my_file}")
    else:
        print(
            f"Training BPE model for {language}, as {my_file} doesn't exist, for \
              vocab size {vocab_size}"
        )
        nlp.train()
        nlp.save(f"{my_file}")
    return nlp

def save_losses(training_losses, validation_losses, language):
    # Example training loss over epochs
    epochs = range(1, len(training_losses) + 1)

    # Plot the loss values
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Save the plot to a file
    print(f"Saving training loss plot for {language}")
    plt.savefig(f'../loss_plots/training_loss_{language}.png')  # Save the plot as a .png file


import torch.nn as nn
import json
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import time
import torch
from bpe import BPE

def add_word_overlap(ds):
    ds['word_overlap'] = calculate_word_overlap(ds['translated'], ds['context'])
    return ds

# Calculate word overlap
def calculate_word_overlap(translated, context):
    translated_words = set(translated.lower().split())
    context_words = set(context.lower().split())
    overlap = len(translated_words.intersection(context_words))
    return overlap

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

def create_dirs():
    if not os.path.exists("../checkpoints"):
        os.makedirs("../checkpoints")
        print("Created ../checkpoints directory")
    if not os.path.exists("../checkpoints/vocab"):
        os.makedirs("../checkpoints/vocab")
        print("Created ../checkpoints/vocab directory")
    if not os.path.exists("../checkpoints/bpe"):
        os.makedirs("../checkpoints/bpe")
        print("Created ../checkpoints/bpe directory")
    if not os.path.exists("../loss_plots"):
        os.makedirs("../loss_plots")
        print("Created ../loss_plots directory")
    if not os.path.exists("../data"):
        os.makedirs("../data")
        print("Created ../data directory")
    if not os.path.exists("../data/train"):
        os.makedirs("../data/train")
        print("Created ../data/train directory")
    if not os.path.exists("../data/val"):
        os.makedirs("../data/val")
        print("Created ../data/val directory")

def gen_vocab(sentences, language, nlp):
    filepath = f"../checkpoints/vocab/{language}_vocab.pkl"
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

def get_data_dic(language, val=False):
    filepath = ""
    if val:
        filepath = f"../data/val/edited_ds_{language}.json"
    else:
        filepath = f"../data/train/edited_ds_{language}.json"
    with open(filepath, "r", encoding='utf-8') as f:
        dic = json.load(f)
    return dic
    
def get_bpe(sentences, language, vocab_size):
    nlp = BPE(sentences, vocab_size)
    path = f"../checkpoints/bpe/bpe_model_{language}_{vocab_size}.pkl"
    if os.path.exists(path):
        nlp.load(f'{path}')
    else:
        print(f"Training BPE model for {language}, as it doesn't exist, for \
              vocab size {vocab_size}")
        nlp.train()
        nlp.save(f'{path}')
    return nlp

def save_losses(training_losses, validation_losses, language):

    # Example training loss over epochs
    epochs = range(1, len(training_losses) + 1)

    # Plot the loss values
    plt.plot(epochs, training_losses, label='Training Loss')
    # plt.plot(epochs, validation_losses, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Save the plot to a file
    print(f"Saving training loss plot for {language}")
    plt.savefig(f'../loss_plots/training_loss_{language}.png')  # Save the plot as a .png file

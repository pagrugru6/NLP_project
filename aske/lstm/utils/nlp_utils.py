import torch.nn as nn
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

def gen_vocab(sentences, language, nlp):
    filepath = f"../checkpoint/{language}_vocab.pkl"
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


def get_sentence_file(language, val=False):
    if val:
        return f"../validation/val_{language}_sentences.txt"
    else:
        return f"../train/train_{language}_sentences.txt"

def get_bpe(sentences, language, vocab_size):
    nlp = BPE(sentences, vocab_size)
    nlp.load(f'../checkpoint/bpe_model_{language}.pkl')
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


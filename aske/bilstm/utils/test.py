#Testfile to load a premade dataset
# Try to load a DATASET and put in a dataloader with the batch_collect_fn
from nlp_utils import *
from torch.utils.data import DataLoader
import logger
import json

# Log at different severity levels
logger.remove(0)
logger.add(sys.stdout, level="DEBUG")
logger.info('This is a debug message')
language = "fi"
train_data = get_data_dic(language)
train_list = [dict(zip(train_data.keys(), values)) for values in
                 zip(*train_data.values())]
reduced_train_list = train_list[:1]

translated = [train_data['translated'] for train_data in train_list]
context = [train_data['context'] for train_data in train_list]
nlp = get_bpe(translated + context, "fi", 1000)
vocab, max_len = gen_vocab(translated + context, "fi", nlp)

def collate_batch_bilstm(input_data):
    # Encode the text data into integers
    # We do input_data[0] as we give it a list of a dictionary. The list has one
    # element
    # Question/context is a list of integers now
    question = [encode(token, vocab) for token in nlp.tokenize(input_data[0]['question'])] 
    context = [encode(token, vocab) for token in nlp.tokenize(input_data[0]['context'])]
    

    # Get the length of each input id
    logger.debug(f"{context=}, this should be a list of integers")

    # Get the length of each element in context?
    seq_leq = [len(ids) for ids in context]

    # Labels = 
    labels = compute_labels(context,
                            encode(nlp.tokenize(input_data[0]['answer']),
                                   vocab), vocab)
    max_len = max([len(i) for i in input_data])
    logger.debug(f"{max_len=}")


# data
# print(nlp.tokenize(dataset['translated'][0]))
dataloader = DataLoader(train_list, batch_size=1, shuffle=True,
                        collate_fn=collate_batch_bilstm)
for batch in dataloader:
    print(batch)
    break

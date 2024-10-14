from datasets import load_dataset, Dataset
import sys
sys.path.append('../utils')
from nlp_utils import *

language = "fi"
print(f"training for language {language}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# ds = load_dataset("coastalcph/tydi_xor_rc")
# ds_val = ds['validation']
# ds_train = ds['train']
# ds = ds_train.filter(filter_langs)
# ds_val = ds_val.filter(filter_langs)
ds = {'question': ['What is the capital of Finland?'],
        'context': ['The capital of Finland is Helsinki.'],
        'answerable': [True]}
ds = Dataset.from_dict(ds)
ds = ds.map(add_word_overlap)
# # ds['word_overlap'] = ds.apply(lambda row: calculate_word_overlap(row['question'], row['context']), axis=1)

print(ds[0])
# Initialize the model
vocab_size = len(vocab)
embed_size = 100  # You can adjust this
hidden_size = 128
model = LSTMClassifier(vocab_size, embed_size, hidden_size)


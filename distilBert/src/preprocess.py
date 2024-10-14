from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

# Load DistilBERT multilingual tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def preprocess_data(file_path):
    """
    Loads the dataset from a JSON file and tokenizes the context and question pairs.
    Args:
    file_path (str): The path to the dataset file (train or validation).
    
    Returns:
    tokenized_dataset: A Hugging Face dataset with tokenized inputs.
    """
    # Load dataset from JSON
    dataset = load_dataset('json', data_files=file_path)

    # Tokenize inputs (context + question)
    def tokenize_fn(examples):
        return tokenizer(
            examples['context'], 
            examples['question'], 
            truncation=True, 
            padding=True, 
            max_length=512
        )

    # Apply tokenization function
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    
    return tokenized_dataset

# Japanese data:
train_ja = preprocess_data('data/train_ja.json')
validation_ja = preprocess_data('data/validation_ja.json')

# Finnish:
train_fi = preprocess_data('data/train_fi.json')
validation_fi = preprocess_data('data/validation_fi.json')

# Russian:
train_ru = preprocess_data('data/train_ru.json')
validation_ru = preprocess_data('data/validation_ru.json')

# Print tokenized dataset examples:

print(train_ja['train'][0])
print(validation_ja['train'][0])

# Load DistilBERT model for Question Answering
# model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-multilingual-cased")


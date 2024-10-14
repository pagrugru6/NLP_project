from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch
import numpy as np

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-multilingual-cased")

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset (use file path here)
dataset = load_dataset('json', data_files={'train': '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train_ja.json', 'validation': '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation_ja.json'})

# Preprocess the dataset using tokenization and moving data to the device
def preprocess_data(examples):
    tokenized_examples = tokenizer(
        examples['context'], examples['question'], truncation=True, padding=True, max_length=512
    )

    # Move input_ids and attention_mask to device
    tokenized_examples["input_ids"] = torch.tensor(tokenized_examples["input_ids"]).to(device)
    tokenized_examples["attention_mask"] = torch.tensor(tokenized_examples["attention_mask"]).to(device)

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, (example, answer_text) in enumerate(zip(examples['context'], examples['answer'])):
        start_char = example.find(answer_text)
        end_char = start_char + len(answer_text) - 1

        word_ids = tokenized_examples.word_ids(batch_index=i)
        start_token_index, end_token_index = None, None

        for j, word_id in enumerate(word_ids):
            if word_id is not None:
                token_start_char = tokenized_examples.token_to_chars(i, j).start
                token_end_char = tokenized_examples.token_to_chars(i, j).end
                if start_char <= token_end_char and token_start_char <= end_char:
                    if start_token_index is None:
                        start_token_index = j
                    end_token_index = j

        if start_token_index is None or end_token_index is None:
            start_token_index = tokenizer.cls_token_id
            end_token_index = tokenizer.cls_token_id

        tokenized_examples["start_positions"].append(start_token_index)
        tokenized_examples["end_positions"].append(end_token_index)

    return tokenized_examples

# Map preprocessing to the dataset
train_dataset = dataset['train'].map(preprocess_data, batched=True)
validation_dataset = dataset['validation'].map(preprocess_data, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define compute_metrics function
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(labels, pred)
    return {"accuracy": accuracy}

# Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics 
)

# Train the model
trainer.train()
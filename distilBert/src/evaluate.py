from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer
from datasets import load_dataset
import torch
import numpy as np
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Directory to store logs and outputs
    per_device_eval_batch_size=16,  # Batch size for evaluation
    remove_unused_columns=False  # Avoid column mismatch errors
)


# Load the trained model and tokenizer
model = DistilBertForQuestionAnswering.from_pretrained('./models/distilbert_ja')
print(model)  # This will print model details
tokenizer = DistilBertTokenizerFast.from_pretrained('./models/distilbert_ja')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test: 
inputs = tokenizer("Who is the president of France?", "Emmanuel Macron is the president of France.", return_tensors="pt")
outputs = model(**inputs)
print(outputs)


# Load validation dataset
dataset = load_dataset('json', data_files={'validation': '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation_ja.json'})

# Preprocess validation data
def preprocess_data(examples):
    return tokenizer(examples['context'], examples['question'], truncation=True, padding=True, max_length=512)

# Tokenize the validation dataset
validation_dataset = dataset['validation'].map(preprocess_data, batched=True)

print(validation_dataset.column_names)


# Define metrics function to compute loss and accuracy
def compute_metrics(eval_pred):
    print("Metrics function triggered")
    logits, labels = eval_pred
    start_logits, end_logits = logits

    # Check if labels are present and have start_positions and end_positions
    if 'start_positions' not in labels or 'end_positions' not in labels:
        print("Labels do not contain start_positions or end_positions.")
        return {}

    start_labels = labels['start_positions']
    end_labels = labels['end_positions']

    # Compute accuracy
    start_preds = np.argmax(start_logits, axis=-1)
    end_preds = np.argmax(end_logits, axis=-1)

    start_accuracy = np.mean(start_preds == start_labels)
    end_accuracy = np.mean(end_preds == end_labels)

    return {
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "eval_loss": np.mean((start_preds != start_labels) | (end_preds != end_labels))
    }

# Initialize Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator 
)

# Evaluate the model
eval_result = trainer.evaluate()

# Print the results
print(f"Evaluation Results: {eval_result}")

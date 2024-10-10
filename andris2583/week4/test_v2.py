import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import Levenshtein as lev  # Import the Levenshtein library

# Load the tokenizer and model from the checkpoint directory
model_save_path = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForTokenClassification.from_pretrained(model_save_path)
model.eval()  # Set model to evaluation mode

# Set the maximum sequence length
MAX_LENGTH = 512

valid_data = pd.read_parquet('../validation.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang', 'answer_start'], filters=[('lang', 'in', ['ja'])])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Store overall true and predicted labels
all_true_labels = []
all_predicted_labels = []

# Variables to track proximity accuracy and Levenshtein count
correct_count = 0
lev_distance_count = 0
total_count = 0

for context, question, answerable, real_answer, lang, answer_start in list(valid_data.itertuples(index=False, name=None)):
    # Tokenize the context and question
    inputs = tokenizer(context, question, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="pt")

    # Move the input tensors to the GPU if available
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Pass the inputs through the model
    with torch.no_grad():  # Disable gradient calculation during inference
        outputs = model(**inputs)

    # The logits represent predictions (before applying activation function like softmax)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]  # Get predicted labels

    # Convert token IDs to tokens (words)
    input_ids = inputs['input_ids'].cpu().numpy()[0]  # Get input token ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Create true label array
    true_labels = np.zeros(len(tokens), dtype=int)
    true_labels[answer_start:answer_start + len(real_answer.split())] = 1  # Set true labels based on actual answer length

    # Collect all true and predicted labels for metric calculations
    all_true_labels.extend(true_labels)
    all_predicted_labels.extend(predictions)

    # Find the first contiguous set of predicted answer tokens in the context
    predicted_answer_tokens = []
    in_answer = False

    for i in range(len(predictions)):
        if predictions[i] == 1:  # If the token is part of the answer
            predicted_answer_tokens.append(tokens[i])
            in_answer = True
        elif in_answer:
            # Stop if we were in an answer sequence and hit a non-answer token
            break

    # Convert the list of predicted answer tokens to a string
    first_adjacent_answer = tokenizer.convert_tokens_to_string(predicted_answer_tokens)

    # Calculate Levenshtein Distance
    distance = lev.distance(first_adjacent_answer.strip(), real_answer.strip())

    # Define a dynamic threshold based on the length of the real answer
    answer_length = len(real_answer.strip())
    if answer_length <= 3:
        threshold = 1  # Short answers
    elif answer_length <= 10:
        threshold = 2  # Medium answers
    else:
        threshold = max(1, round(answer_length * 0.1))  # Long answers

    # Check if the distance is within the threshold
    if distance <= threshold:
        lev_distance_count += 1

    # Print for debugging
    if first_adjacent_answer != "":
        print(f"Predicted: '{first_adjacent_answer}' - Real: '{real_answer.strip()}' - Levenshtein Distance: {distance} - Threshold: {threshold}")

    # Count total predictions
    total_count += 1

# Convert lists to numpy arrays for metric calculations
all_true_labels = np.array(all_true_labels)
all_predicted_labels = np.array(all_predicted_labels)

# Calculate overall metrics
accuracy = accuracy_score(all_true_labels, all_predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, average='binary')

# Print performance metrics
print(f"Accuracy (Token Classification): {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Levenshtein Distance Count (successful predictions): {lev_distance_count}/{total_count} = {lev_distance_count / total_count:.4f}")

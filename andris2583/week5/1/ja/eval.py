import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from functools import partial
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer
model_dir = './mt5_model/'
model = MT5ForConditionalGeneration.from_pretrained(model_dir).to(device)
tokenizer = MT5Tokenizer.from_pretrained(model_dir)

# Load your validation dataset
valid_data = pd.read_parquet('../../../../validation.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang','answer_inlang'], filters=[('lang', 'in', ['ja']),('answer_inlang', '!=', 'null')])

# Add unique 'id' field to the dataset if it doesn't already exist
valid_data['id'] = valid_data.index.astype(str)
validation_dataset = Dataset.from_pandas(valid_data)

def prepare_data(samples, tokenizer=None, max_input_length=512, max_target_length=64):
    """
    Tokenize input data in the format: "[context] [question]" and target as the answer.
    """
    # Ensure 'answer_inlang' is present, otherwise use an empty string for missing values
    answers = samples.get('answer_inlang', [""] * len(samples['context']))

    # Prepare the input text as "[context] [question]"
    inputs = [f"{context} {question}" for context, question in zip(samples['context'], samples['question'])]
    
    # Tokenize inputs (context + question)
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding="max_length"
    )

    # Tokenize targets (answers)
    labels = tokenizer(
        answers, max_length=max_target_length, truncation=True, padding="max_length"
    ).input_ids

    model_inputs["labels"] = labels
    return model_inputs

# Collate function for DataLoader
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Prediction function
def generate_answers(model, valid_dl):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(valid_dl, desc="Generating answers"):
            batch = {k: v.to(device) for k, v in batch.items()}

            generated_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=64,
                num_beams=4
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(preds)
    return all_predictions

# Load validation dataset and prepare inputs
tokenized_valid = validation_dataset.map(partial(prepare_data, tokenizer=tokenizer), batched=True)
valid_dl = DataLoader(tokenized_valid, collate_fn=collate_fn, shuffle=False, batch_size=4)

# Generate answers
generated_answers = generate_answers(model, valid_dl)

# Post-process the answers for evaluation
predictions = [{'id': str(i), 'prediction_text': pred} for i, pred in enumerate(generated_answers)]

# Assuming the structure of 'valid_data' has answers
gold = [{'id': example['id'], 'answers': example['answer']} for _, example in valid_data.iterrows()]

# Evaluation function (adapted for MT5)
def compute_metrics(predictions, references):
    exact_match = f1 = total = 0
    bleu_score_total = 0
    chencherry = SmoothingFunction()

    for pred, ref in zip(predictions, references):
        total += 1
        pred_text = pred["prediction_text"]
        true_text = ref["answers"]

        exact_match += exact_match_score(pred_text, true_text)
        f1 += f1_score_fn(pred_text, true_text)

        # Tokenize for BLEU score (BLEU expects tokenized input)
        pred_tokens = pred_text.split()
        ref_tokens = [true_text.split()]  # BLEU expects references as a list of token lists

        # Compute BLEU score for this prediction-reference pair
        bleu_score_total += sentence_bleu(ref_tokens, pred_tokens, smoothing_function=chencherry.method1)

    # Average scores
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    avg_bleu = bleu_score_total / total

    return {'exact_match': exact_match, 'f1': f1, 'bleu': avg_bleu}
def exact_match_score(prediction, ground_truth):
    return prediction.strip().lower() == ground_truth.strip().lower()

def f1_score_fn(prediction, ground_truth):
    prediction_tokens = prediction.strip().lower().split()
    ground_truth_tokens = ground_truth.strip().lower().split()
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    if not common_tokens:
        return 0
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# Generate answers
generated_answers = generate_answers(model, valid_dl)

# Post-process the answers for evaluation
predictions = [{'id': str(i), 'prediction_text': pred} for i, pred in enumerate(generated_answers)]

# Assuming the structure of 'valid_data' has answers
gold = [{'id': example['id'], 'answers': example['answer']} for _, example in valid_data.iterrows()]

# Compute metrics (Exact Match and F1 score)
results = compute_metrics(predictions, gold)

# Print the question, context, and predicted answer for each example
print("\nPredictions with Context and Question:\n")
for i, prediction in enumerate(predictions):
    # Retrieve the context, question, and predicted answer
    context = valid_data.iloc[i]['context']
    question = valid_data.iloc[i]['question']
    answer = valid_data.iloc[i]['answer']
    predicted_answer = prediction['prediction_text']
    
    print(f"Example {i + 1}")
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Real answer: {answer}")
    print(f"Predicted Answer: {predicted_answer}\n")
    
# Print results
print(f"Evaluation Results:\nExact Match: {results['exact_match']:.2f}%\nF1 Score: {results['f1']:.2f}%\nBLEU Score: {results['bleu']:.4f}")

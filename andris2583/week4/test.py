import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification

# Load the tokenizer and model from the checkpoint directory
model_save_path = "./model_weights"
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForTokenClassification.from_pretrained(model_save_path)
model.eval()  # Set model to evaluation mode

# Set the maximum sequence length
MAX_LENGTH = 512

valid_data = pd.read_parquet('./validation.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang'], filters=[('lang', 'in', ['ja'])])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for context, question, answerable, real_answer, lang in list(valid_data.itertuples(index=False, name=None))[:200]:
  # Tokenize the context and question
  inputs = tokenizer(context, question, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="pt")

  # Move the input tensors to the GPU if available
  inputs = {key: val.to(device) for key, val in inputs.items()}

  # Pass the inputs through the model
  with torch.no_grad():  # Disable gradient calculation during inference
      outputs = model(**inputs)
      
  # The logits represent predictions (before applying activation function like softmax)
  logits = outputs.logits
  predictions = torch.argmax(logits, dim=-1).cpu().numpy()

  # Convert predicted token labels back to words
  input_ids = inputs['input_ids'].cpu().numpy()[0]  # Get input token ids
  pred_labels = predictions[0]  # Get predicted labels

  # Convert token IDs to tokens (words)
  tokens = tokenizer.convert_ids_to_tokens(input_ids)

  # Extract answer from the predicted labels (label 1 is for answer tokens)
  answer_tokens = [token for token, label in zip(tokens, pred_labels) if label == 1]
  answer = tokenizer.convert_tokens_to_string(answer_tokens)


  # print("Tokenized Input IDs:", inputs['input_ids'])
  # print("Tokens:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
  # print(predictions)

  # Print the predicted answer
  if(len(answer_tokens) != 0):
    print(real_answer,"Predicted answer: ",answer_tokens)
    # print(tokens, pred_labels)

## Evaluate model once we have finetuned it

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import bitsandbytes as bnb
from sklearn.model_selection import train_test_split
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, pipeline
import torch.nn.functional as F
import accelerate
import sklearn
import peft
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from trl import SFTTrainer
from datasets import Dataset
import wandb
from sklearn.metrics import mean_squared_error
from datasets import DatasetInfo, Features, Value
from datasets import load_from_disk
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")


# Load the model
model = LlamaForCausalLM.from_pretrained('/n/home09/lschrage/projects/cs109b/finetuned_model')
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=ACCESS_TOKEN, return_tensors = 'tf')

url = '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/dataframe.csv'
df = pd.read_csv(url)
df.head()
'''
y = df['TweetAvgAnnotation']
X = df

X_train_full, X_test, y_true_train_full, y_true_test = train_test_split(X, y, test_size=0.2, random_state=109, stratify=X['Sentiment'])

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_true_train_full, test_size=0.5, random_state=109, stratify=X_train_full['Sentiment'])

def generate_train_prompt(tweet):
  return f"""
          Analyze the sentiment of the tweet enclosed in square brackets,
          determine if it is positive, neutral, or negative, and return the answer as a float value rounded to two decimal places
          between -3 (corresponding to a  negative sentiment) and 3 (corresponding to a positive sentiment).

          [{tweet["Tweet"]}] = {tweet["TweetAvgAnnotation"]}
          """.strip()
def generate_test_prompt(tweet):
  return f"""
          Analyze the sentiment of the tweet enclosed in square brackets,
          determine if it is positive, neutral, or negative, and return the answer as a float value rounded to two decimal places
          between -3 (corresponding to a  negative sentiment) and 3 (corresponding to a positive sentiment).

          [{tweet["Tweet"]}] =
          """.strip()

X_train_full['Prompt'] = X_train_full.apply(generate_train_prompt, axis=1)
X_train['Prompt'] = X_train.apply(generate_train_prompt, axis=1)
X_test['Prompt'] = X_test.apply(generate_test_prompt, axis=1)
X_val['Prompt'] = X_val.apply(generate_test_prompt, axis=1)

df_val = pd.DataFrame({
    "input_ids": X_val['Prompt'],
    "labels": y_val
})

val_dataset = Dataset.from_pandas(df_val)

val_loader = DataLoader(val_dataset, batch_size=8)
'''
test_dataset = load_from_disk('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama-finetune-test-dataset')
val_dataset = load_from_disk('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama-finetune-val-dataset')

from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


import re

def extract_sentiment(prediction_text):
    # Regex to find the pattern '= <optional space> <number>'
    # It looks specifically for an optional negative sign, followed by one or more digits,
    # optionally followed by a decimal point and more digits (the pattern for a floating point number).
    match = re.search(r'=\s*(-?\d+(?:\.\d+)?)', prediction_text)
    if match:
        return float(match.group(1))
    else:
        return 0.0
    
import torch
from sklearn.metrics import mean_squared_error

def evaluate_model(dataloader):
    model.eval()  # Ensure model is in evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Ensure data is on the correct device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            # Generate outputs
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)

            # Decode generated ids to text and extract sentiment
            prediction_texts = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in outputs]
            sentiments = [extract_sentiment(text) for text in prediction_texts]
            
            # Append results
            predictions.extend(sentiments)
            true_labels.extend(batch['labels'].tolist())

    # Calculate Mean Squared Error
    mse = mean_squared_error(true_labels, predictions)
    return mse

# Calculate MSE for validation and test sets
test_mse = evaluate_model(test_loader)
val_mse = evaluate_model(val_loader)

print(f'Validation MSE: {val_mse}')
print(f'Test MSE: {test_mse}')


####

def plot_predictions_vs_actual_finetune(model, dataset, path):
    # Assuming the dataset is a Huggingface Dataset object and directly accessible
    true_labels = dataset['labels']
    predicted_scores = []
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Loop through the entire dataset for prediction
        for item in dataset:
            input_ids = torch.tensor(item['input_ids']).unsqueeze(0).to('cuda')
            attention_mask = torch.tensor(item['attention_mask']).unsqueeze(0).to('cuda')
            
            # Get the model output
            outputs = model(input_ids, attention_mask=attention_mask)
            score = outputs.logits.squeeze().item()
            predicted_scores.append(score)
    
    # Plot the true labels vs. predicted scores
    plt.figure(figsize=(10, 5))
    plt.scatter(true_labels, predicted_scores, alpha=0.5)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')
    plt.title('Actual vs Predicted Sentiment Scores')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.grid(True)
    plt.savefig(path)
plot_predictions_vs_actual_finetune(trainer.model, test_dataset, 'FINETUNE-llama-actual-vs-predicted.png')

history = pd.DataFrame(trainer.state.log_history)
print(history.columns)
train_loss = history['loss'].dropna()
val_loss = history['eval_loss'].dropna()

def plot_train_val_loss(train_loss, val_loss, path):
    # Assuming train_loss and val_loss are directly passed lists
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Training MSE')
    plt.plot(epochs, val_loss, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
plot_train_val_loss(train_loss, val_loss, 'FINETUNE-llama-train-val-mse.png')
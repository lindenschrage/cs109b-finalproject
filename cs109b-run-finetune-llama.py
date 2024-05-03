
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

# Load the model
model = LlamaForCausalLM.from_pretrained('/n/home09/lschrage/projects/cs109b/finetuned_model')
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=ACCESS_TOKEN, return_tensors = 'tf')

url = '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/dataframe.csv'
df = pd.read_csv(url)
df.head()

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
    "text": X_val['Prompt'],
    "labels": y_val
})

val_dataset = Dataset.from_pandas(df_val)

val_loader = DataLoader(val_dataset, batch_size=8)

import re

def extract_sentiment(prediction_text):
    # Regex to find the pattern '= <optional space> <number>'
    # It looks specifically for an optional negative sign, followed by one or more digits,
    # optionally followed by a decimal point and more digits (the pattern for a floating point number).
    match = re.search(r'=\s*(-?\d+(?:\.\d+)?)', prediction_text)
    if match:
        # If a match is found, convert it to float and return
        return float(match.group(1))
    else:
        # If no valid number is found, return 0.0
        return 0.0
    

import torch
from sklearn.metrics import mean_squared_error

model.eval()  # Set the model to evaluation mode
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        # Generate predictions
        outputs = model.generate(batch['input_ids'], attention_mask=batch['attention_mask'])
        prediction_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        sentiments = [extract_sentiment(text) for text in prediction_texts]
        
        # Collect predictions and actual labels
        predictions.extend(sentiments)
        true_labels.extend(batch['labels'].numpy())  # Assuming labels are in a tensor format

# Ensure predictions and true_labels are the same length and corresponding elements match
assert len(predictions) == len(true_labels)

mse = mean_squared_error(true_labels, predictions)
print(f"Mean Squared Error: {mse}")

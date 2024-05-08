
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
import accelerate
from peft import LoraConfig, get_peft_model 
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

os.environ["WANDB_PROJECT"]="twitter-sentiment-analysis"

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

url = 'dataframe.csv'
df = pd.read_csv(url)
print(df.head())

y = df['TweetAvgAnnotation']
X = df
print('1')

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=109, stratify=X['Sentiment'])

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=109, stratify=X_train_full['Sentiment'])

def generate_train_prompt(tweet):
  return f"""
          Analyze the sentiment of the tweet enclosed in square brackets,
          determine if it is optimistic, neutral, or pessamistic, and return the answer as a float value rounded to two decimal places
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

model = "meta-llama/Llama-2-7b-hf"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

print('2')
llama_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=ACCESS_TOKEN,
    quantization_config=bnb_config,
    output_hidden_states=False,
    output_attentions=False)
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

config = LoraConfig(
        lora_alpha=16, 
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

model = get_peft_model(llama_model, config)

llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=ACCESS_TOKEN)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

df_train = pd.DataFrame({
    "input_ids": X_train['Prompt'],
    "labels": y_train
})

df_val = pd.DataFrame({
    "input_ids": X_val['Prompt'],
    "labels": y_val
})

df_test = pd.DataFrame({
    "input_ids": X_test['Prompt'],
    "labels": y_test
})

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)
test_dataset = Dataset.from_pandas(df_test)

def tokenize_function(df):
    return llama_tokenizer(df["input_ids"], padding="max_length", truncation=True, max_length=1024)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer, return_tensors="pt")

train_loader = DataLoader(
    train_dataset,  
    batch_size=1,   
    shuffle=True,   
    collate_fn=data_collator,  
    drop_last=True  
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=data_collator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=data_collator
)

train_params = TrainingArguments(
    output_dir="/n/home09/lschrage/projects/cs109b/finetuned_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=100,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb",
    evaluation_strategy="steps",
    eval_steps=100,
    metric_for_best_model='accuracy',  

)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=llama_tokenizer,
    args=train_params,
    dataset_text_field = 'input_ids'
)

trainer.train()

history = pd.DataFrame(trainer.state.log_history)
print("Columns in history:", history.columns)
for col in history.columns:
    print(f"First 5 entries in column '{col}':")
    print(history[col].head(), "\n")


##########


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
            prediction_texts = [llama_tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in outputs]
            sentiments = [extract_sentiment(text) for text in prediction_texts]
            
            # Append results
            predictions.extend(sentiments)
            true_labels.extend(batch['labels'].tolist())

    # Calculate Mean Squared Error
    print(true_labels)
    mse = mean_squared_error(true_labels, predictions)
    return mse

# Calculate MSE for validation and test sets
test_mse = evaluate_model(test_loader)
val_mse = evaluate_model(val_loader)

print(f'Validation MSE: {val_mse}')
print(f'Test MSE: {test_mse}')

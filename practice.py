import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaTokenizer, LlamaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from trl import SFTTrainer
from sklearn.metrics import r2_score
from transformers import TrainingArguments, Trainer
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

import peft
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

url = 'dataframe.csv'
df = pd.read_csv(url)
print(df.head())

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model_name = 'meta-llama/Llama-2-7b-hf'
llama_model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=1, quantization_config=bnb_config, token=ACCESS_TOKEN)
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

config = LoraConfig(
        lora_alpha=16, 
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

model = get_peft_model(llama_model, config)

tweet_text = list(df['Tweet'])
tweet_annotations = list(df['TweetAvgAnnotation'])

y = df['TweetAvgAnnotation']
X = df

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=109, stratify=X['Sentiment'])

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.5, random_state=109, stratify=X_train_full['Sentiment'])

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
    return tokenizer(df["input_ids"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

def compute_metrics_for_regression(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

training_args = TrainingArguments(
    output_dir="/n/home09/lschrage/projects/cs109b/FINETUNE-llama",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="mse",
    load_best_model_at_end=True,
    weight_decay=0.001,
    warmup_ratio=0.03, 
    fp16=True,
    bf16=False,
    logging_strategy="epoch",
    logging_steps=50,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_for_regression,
    dataset_text_field='input_ids',
    max_seq_length=512
)

trainer.train()

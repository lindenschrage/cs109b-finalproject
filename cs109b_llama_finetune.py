# -*- coding: utf-8 -*-
"""cs109b-llama-finetune.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Pvg2uGiR91WT88EDkBPLypgTz8JrWCix
"""

#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

'''
!pip install bitsandbytes
!pip install accelerate
!pip install peft
!pip install trl
'''

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

import os
os.environ["WANDB_PROJECT"]="twitter-sentiment-analysis"


## DATA PREPROCESSING
'''
tweet_df = pd.read_csv('/content/drive/My Drive/cs109b-finalproject/tweets_annotation.csv')
tweet_df['TweetAvgAnnotation'] = tweet_df['AverageAnnotation']
tweet_df.drop('AverageAnnotation', axis=1)

user_df = pd.read_csv('/content/drive/My Drive/cs109b-finalproject/user_information.csv')
user_df['UserAvgAnnotation'] = user_df['AverageAnnotation']
user_df.drop('AverageAnnotation', axis=1)

df = pd.merge(tweet_df, user_df, on='Username')
df = df.drop('AverageAnnotation_y', axis=1)
df = df.drop('AverageAnnotation_x', axis=1)

def extract_info(profile_info_str):
    if not isinstance(profile_info_str, str):
        return pd.Series({
          'UserDescription': None,
          'Followers': None,
          'Following': None,
          'TotalTweetCount': None,
          'FavoritesCount': None
        })

    info_dict = {}

    patterns = {
        'UserDescription': r'description: (.*?),',
        'Followers': r'followers: (\d+)',
        'Following': r'following: (\d+)',
        'TotalTweetCount': r'total tweet number: (\d+)',
        'FavoritesCount': r'favorites_count: (\d+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, profile_info_str)
        if match:
            info_dict[key] = match.group(1).strip()
        else:
            info_dict[key] = None

    return pd.Series(info_dict)

new_columns = df['ProfileInfo'].apply(extract_info)
df = pd.concat([df, new_columns], axis=1)
df = df.drop('ProfileInfo', axis=1)

def label_value(x):
    if x < -1:
        return 'Negative'
    elif x > 1:
        return 'Positive'
    else:
        return 'Neutral'

df['Sentiment'] = df['TweetAvgAnnotation'].apply(label_value)
'''

'''
url = 'https://raw.githubusercontent.com/lindenschrage/cs109b-finalproject/main/dataframe.csv'
df = pd.read_csv(url)
df.head()
'''

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

'''
X_val['Random'] = np.round(np.random.uniform(-3, 3, size=len(X_val)), 2)
y_random = list(X_val['Random'])
'''

access_token = "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"
model = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
)


llama_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=access_token,
    quantization_config=bnb_config,
    output_hidden_states=False,
    output_attentions=False)
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

df_train = pd.DataFrame({
    "text": X_train['Prompt'],
    "labels": y_train
})

df_val = pd.DataFrame({
    "text": X_val['Prompt'],
    "labels": y_val
})

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)

peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=1,
    learning_rate=1e-6, 
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=1.0,  
    max_steps=-1,
    warmup_ratio=0.1,  
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb",
    max_seq_length = None,
    evaluation_strategy="steps",
    eval_steps=2000)

fine_tuning = SFTTrainer(
    model=llama_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params,
    compute_metrics=compute_metrics
)

fine_tuning.train()


## YOU HAVE TO CHANGE THIS LINE TO YOUR LOCAL DIRECTORY

#fine_tuning.model.save_pretrained('/content/drive/My Drive/cs109b-finalproject/finetuned_model')
fine_tuning.model.save_pretrained('/n/home09/lschrage/projects/cs109b/finetuned_model')


print("Done")

#model = AutoModelForCausalLM.from_pretrained('/content/drive/My Drive/cs109b-finalproject/finetuned_model')

## We could use something like these functions to make custom loss function and custom trainer which would force the output of llama to only be specific characters

'''
import torch
import torch.nn.functional as F

class CustomLoss(torch.nn.Module):
    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed_token_ids = allowed_token_ids

    def forward(self, predictions, labels):
        loss = F.cross_entropy(predictions, labels, reduction='none')

        penalty_mask = ~labels.unsqueeze(-1).isin(self.allowed_token_ids)
        penalties = penalty_mask.float() * 10.0  # Arbitrary high penalty
        loss += penalties.sum(-1)

        return loss.mean()



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = custom_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

allowed_token_ids = tokenizer.convert_tokens_to_ids(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-'])
custom_loss = CustomLoss(allowed_token_ids)

trainer = CustomTrainer(
    model=llama_model,
    args=train_params,
    train_dataset=train_dataset,
    tokenizer=llama_tokenizer,
    compute_loss=custom_loss
)

'''
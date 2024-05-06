
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
from transformers import LlamaTokenizer, LlamaForSequenceClassification, BitsAndBytesConfig, pipeline
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

y = np.array(y, dtype=np.float16)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=109, stratify=X['Sentiment'])

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=109, stratify=X_train_full['Sentiment'])

model = "meta-llama/Llama-2-7b-hf"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

llama_model = LlamaForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=ACCESS_TOKEN,
    quantization_config=bnb_config,
    num_labels=1,
    problem_type='regression',
    ignore_mismatched_sizes=True,
    torch_dtype=torch.float16,)
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1



config = LoraConfig(
        lora_alpha=16, 
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules="all-linear",
        task_type="SEQ_CLS",
)

model = get_peft_model(llama_model, config)

llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=ACCESS_TOKEN)
llama_tokenizer.pad_token = llama_tokenizer.eos_token 
df_train = pd.DataFrame({
    "input_ids": X_train['Tweet'],
    "labels": y_train
})

df_val = pd.DataFrame({
    "input_ids": X_val['Tweet'],
    "labels": y_val
})

df_test = pd.DataFrame({
    "input_ids": X_test['Tweet'],
    "labels": y_test
})

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)
test_dataset = Dataset.from_pandas(df_test)

def process_inputs(example):
    # Tokenize the inputs
    result = llama_tokenizer(example['input_ids'])
    # Make sure labels are maintained as scalars
    result['labels'] = example['labels']
    return result

train_dataset = train_dataset.map(process_inputs, batched=True)
val_dataset = val_dataset.map(process_inputs, batched=True)
test_dataset = test_dataset.map(process_inputs, batched=True)

print("1", train_dataset['labels'][:5])
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print("2",train_dataset['labels'][:5])


def convert_to_fp16(batch):
    # Convert labels to float16 and reshape to match expected model output dimensions
    labels = batch['labels'].clone().detach().to(dtype=torch.float16).unsqueeze(-1)
  # Shape should be [batch_size, 1]
    batch['labels'] = labels
    return batch


train_dataset = train_dataset.map(convert_to_fp16, batched=True)
val_dataset = val_dataset.map(convert_to_fp16, batched=True)
print("3",train_dataset['labels'][:5])

for data in train_dataset:
    inputs, labels = data
    print("Input shape:", inputs.shape)
    print("Labels shape:", labels.shape)
    if labels.shape != torch.Size([1, 1]):
        print("Incorrect label shape:", labels)
        # Potentially reshape or select the correct labels
        labels = labels.view(1, 1)  # Adjust this based on your specific needs
        print("Fixed Labels shape:", labels.shape)

    # Now pass to model and compute loss
    logits = model(inputs)
    loss = F.mse_loss(logits, labels)
    print("Loss:", loss.item())

#train_dataset.save_to_disk('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama-finetune-train-dataset')
#val_dataset.save_to_disk('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama-finetune-val-dataset')
#test_dataset.save_to_disk('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama-finetune-test-dataset')

train_params = TrainingArguments(
    output_dir="/n/home09/lschrage/projects/cs109b/finetuned_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    save_steps=25,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    metric_for_best_model="mse",
    lr_scheduler_type="linear",
    report_to="wandb",
    evaluation_strategy="steps",
    eval_steps=2000

)


class DebugTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        # Print shapes and values for debugging
        print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        print(f"First few logits: {logits[:5]}, First few labels: {labels[:5]}")
        
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Use this trainer for debugging
fine_tuning = DebugTrainer(
    model=model,
    args=train_params,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=llama_tokenizer,
    dataset_text_field = 'input_ids',
)
'''
fine_tuning = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=llama_tokenizer,
    args=train_params,
    dataset_text_field = 'input_ids',
    max_seq_length=512
)
'''
fine_tuning.train()

fine_tuning.model.save_pretrained('/n/home09/lschrage/projects/cs109b/finetuned_model')


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
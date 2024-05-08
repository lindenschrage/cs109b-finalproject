
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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datasets import DatasetInfo, Features, Value
from datasets import load_from_disk
import accelerate
from peft import LoraConfig, get_peft_model 
import os
from peft import prepare_model_for_kbit_training
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 


os.environ["WANDB_PROJECT"]="twitter-sentiment-analysis"

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

url = 'dataframe.csv'
df = pd.read_csv(url)

y = df['TweetAvgAnnotation']
X = df

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=109, stratify=X['Sentiment'])

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=109, stratify=X_train_full['Sentiment'])


model = "meta-llama/Llama-2-7b-hf"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

llama_model = LlamaForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=ACCESS_TOKEN,
    quantization_config=nf4_config,
    num_labels=1,
    problem_type='regression',
    ignore_mismatched_sizes=True)
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

llama = prepare_model_for_kbit_training(llama_model)

config = LoraConfig(
    r=64, lora_alpha=64, target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ], lora_dropout=0.05, bias="none"
)

llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=ACCESS_TOKEN)
llama_tokenizer.pad_token = llama_tokenizer.eos_token 

model = get_peft_model(llama, config)
model.config.pad_token_id = llama_tokenizer.pad_token_id

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
    result = llama_tokenizer(example['input_ids'])
    result['labels'] = example['labels']
    return result

train_dataset = train_dataset.map(process_inputs, batched=True)
val_dataset = val_dataset.map(process_inputs, batched=True)
test_dataset = test_dataset.map(process_inputs, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


def convert_to_fp16(batch):
    labels = batch['labels'].clone().detach().to(dtype=torch.float16).unsqueeze(-1)
    batch['labels'] = labels
    return batch


train_dataset = train_dataset.map(convert_to_fp16, batched=True)
val_dataset = val_dataset.map(convert_to_fp16, batched=True)
test_dataset = test_dataset.map(convert_to_fp16, batched=True)


from transformers import DataCollatorWithPadding

class CustomCollatorWithPadding:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, batch):
        batch = self.data_collator(batch)
        if 'labels' in batch:
            batch['labels'] = batch['labels'].to(dtype=torch.float16)
        return batch

data_collator = CustomCollatorWithPadding(tokenizer=llama_tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)

def plot_predictions_vs_actual_finetune(model, test_dataset, path):
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)
    true_labels = [item['labels'].item() for item in test_dataset]
    predicted_scores = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_scores = outputs.logits.squeeze().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            predicted_scores.extend(batch_scores)

    plt.figure(figsize=(10, 5))
    plt.scatter(true_labels, predicted_scores, alpha=0.5)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')
    plt.title('Actual vs Predicted Sentiment Scores')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.grid(True)
    plt.savefig(path)
plot_predictions_vs_actual_finetune(model, test_dataset, 'BASELINE-FINETUNE-llama-actual-vs-predicted.png')

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

train_params = TrainingArguments(
    output_dir="/n/home09/lschrage/projects/cs109b/llama_finetuned_model",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=50,
    fp16=True,
    weight_decay=0.01,
    max_steps=280,
    metric_for_best_model="mse",
    load_best_model_at_end=True,
    logging_strategy="steps",
    save_strategy="steps",
    evaluation_strategy="steps",
    logging_steps=40,
    eval_steps=40,
    save_steps=40

)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=llama_tokenizer,
    args=train_params,
    dataset_text_field='input_ids',
    max_seq_length=512,
    data_collator=data_collator,
    compute_metrics=compute_metrics_for_regression
)

train_params.logging_dir = "/n/home09/lschrage/projects/cs109b/llama_finetuned_model_logs"
train_result = trainer.train()

metrics = train_result.metrics
max_train_samples = len(train_dataset)
metrics["train_samples"] = min(train_dataset, len(train_dataset))

# save train results
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# compute evaluation results
metrics = trainer.evaluate()
max_val_samples = len(val_dataset)
metrics["eval_samples"] = min(max_val_samples, len(val_dataset))

# save evaluation results
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

history = pd.DataFrame(trainer.state.log_history)
print("Columns in history:", history.columns)
for col in history.columns:
    print(f"First 5 entries in column '{col}':")
    print(history[col].head(), "\n")



def plot_predictions_vs_actual_finetune(model, test_dataset, path):
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)
    true_labels = [item['labels'].item() for item in test_dataset]
    predicted_scores = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_scores = outputs.logits.squeeze().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            predicted_scores.extend(batch_scores)

    plt.figure(figsize=(10, 5))
    plt.scatter(true_labels, predicted_scores, alpha=0.5)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')
    plt.title('Actual vs Predicted Sentiment Scores')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.grid(True)
    plt.savefig(path)
plot_predictions_vs_actual_finetune(trainer.model, test_dataset, 'FINETUNE-llama-actual-vs-predicted.png')


def plot_predictions_vs_actual_finetune_two(trainer, test_dataset, path):
    result = trainer.predict(test_dataset)
    predictions = result.predictions.squeeze()
    labels = result.label_ids

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.scatter(labels, predictions, alpha=0.5)
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], 'r--')
    plt.title('Actual vs Predicted Sentiment Scores')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.grid(True)
    plt.savefig(path)
plot_predictions_vs_actual_finetune_two(trainer.model, test_dataset, '2-FINETUNE-llama-actual-vs-predicted.png')

'''
def plot_train_val_loss(train_loss, val_loss, path):
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
'''
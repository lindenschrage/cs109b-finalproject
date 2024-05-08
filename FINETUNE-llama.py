# In this script, a LLaMA model is fine-tuned for sentiment regression on tweets. The data is split into training, validation,
# and test sets with stratified sampling. The "meta-llama/Llama-2-7b-hf" model is loaded and configured using a 
# LoRA adapter and BitsAndBytes quantization for efficient training. The data is preprocessed using a tokenizer and a custom 
# data collator to handle padding and batching. Fine-tuning is done through the SFTTrainer API, and the model's performance 
# is assessed using mean squared error metrics and accuracy on positive, negative, and neutral sentiments. Final plots of 
# predicted vs actual scores are generated to visualize the results.

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
from transformers import DataCollatorWithPadding

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
    logging_dir="/n/home09/lschrage/projects/cs109b/llama_finetuned_model_logs",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=50,
    fp16=True,
    weight_decay=0.01,
    max_steps=280,
    metric_for_best_model="mse",
    logging_strategy="steps",
    evaluation_strategy="steps",
    logging_steps=40,
    eval_steps=40,
    do_eval=True,
    prediction_loss_only=True
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

train_result = trainer.train()

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


def get_predictions(model, test_dataset):
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
    return predicted_scores, true_labels
predicted_scores, true_labels = get_predictions(trainer.model, test_dataset)

accuracy = pos_accuracy = neut_accuracy = neg_accuracy = num_pos = num_neg = num_neut = 0
for i in range(len(predicted_scores)):
  if true_labels[i] >= 1.0:
    num_pos += 1
    if predicted_scores[i] >= 1.0:
      accuracy += 1
      pos_accuracy += 1
  elif true_labels[i] <= -1.0:
    num_neg +=1
    if predicted_scores[i] <= -1.0:
      accuracy += 1
      neg_accuracy += 1
  else:
    num_neut+=1
    if ((predicted_scores[i] < 1.0) and (predicted_scores[i] > -1.0)):
      accuracy += 1
      neut_accuracy += 1
accuracy_score = accuracy / len(predicted_scores)
pos_score = pos_accuracy / num_pos
neg_score = neg_accuracy / num_neg
neut_score = neut_accuracy / num_neut


print("Accuracy score", accuracy_score)
print("Pos score", pos_score)
print("Neg score", neg_score)
print("Neut score", neut_score)


metrics = train_result.metrics
print(metrics)
metrics1 = trainer.evaluate()
print(metrics1)

history = pd.DataFrame(trainer.state.log_history)
print("Columns in history:", history.columns)
for col in history.columns:
    print(f"First 5 entries in column '{col}':")
    print(history[col].head(), "\n")

from sklearn.metrics import mean_squared_error

# Ensure the model is in evaluation mode
model.eval()

# Store predictions and actual labels
all_preds = []
all_labels = []

# Perform inference on the validation dataset
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].cpu().numpy()

        # Obtain model outputs
        outputs = model(input_ids, attention_mask=attention_mask)

        # Detach and move predictions to the CPU
        predictions = outputs.logits.squeeze().cpu().numpy()

        if isinstance(predictions, float):
            predictions = [predictions]

        all_preds.extend(predictions)
        all_labels.extend(labels)

# Calculate the mean squared error
val_mse = mean_squared_error(all_labels, all_preds)

print(f'Validation Mean Squared Error: {val_mse}')


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

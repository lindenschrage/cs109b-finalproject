# In this approach, we fine-tuned a BERT model to predict sentiment scores from tweets. We used the BertTokenizer for 
# tokenization and BertForSequenceClassification as the model, customizing it for regression with a single output label.
# We split the data into training, validation, and test sets to evaluate the model's performance. We implemented a custom 
# TextDataset class to handle tokenization and padding for each input tweet, enabling consistent input processing. 
# For training, we configured data loaders and utilized the Trainer API to handle training arguments, data collators, 
# and metrics. Finally, we visualized the results by comparing the predicted sentiment scores to actual scores and plotted
# the training and validation loss curves over epochs.

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
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

df_url = 'data/dataframe.csv'
df = pd.read_csv(df_url)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

tweet_text = list(df['Tweet'])
tweet_annotations = list(df['TweetAvgAnnotation'])

train_texts, temp_texts, train_labels, temp_labels = train_test_split(tweet_text, tweet_annotations, test_size=0.3, random_state=109)
test_texts, val_texts, test_labels, val_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=109)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = [float(label) for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor([label], dtype=torch.float)
        }

max_len = 128
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 25

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def plot_predictions_vs_actual_baseline(model, test_dataset, path):
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
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

plot_predictions_vs_actual_baseline(model, test_dataset, 'BASELINE-FINETUNE-bert-actual-vs-predicted.png')

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
    output_dir="/n/home09/lschrage/projects/cs109b/FINETUNE-bert",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="mse",
    load_best_model_at_end=True,
    weight_decay=0.01,
    logging_strategy="epoch",
    logging_steps=50,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_for_regression,
    data_collator=data_collator,
    dataset_text_field='input_ids',
    max_seq_length=max_len
)

trainer.train()

def plot_predictions_vs_actual_finetune(model, test_dataset, path):
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
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
plot_predictions_vs_actual_finetune(trainer.model, test_dataset, 'FINETUNE-bert-actual-vs-predicted.png')

history = pd.DataFrame(trainer.state.log_history)
train_loss = history['loss'].dropna()
val_loss = history['eval_loss'].dropna()

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
plot_train_val_loss(train_loss, val_loss, 'FINETUNE-bert-train-val-mse.png')

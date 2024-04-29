# -*- coding: utf-8 -*-
import transformers
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import keras
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
from transformers import LlamaForCausalLM, LlamaTokenizer

access_token = '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'

base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token).to('cuda')

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llma-2-7b-hf", token=access_token, return_tensors = 'tf')
tokenizer.pad_token_id = (0)
tokenizer.padding_side = "left"

#url = 'https://raw.githubusercontent.com/lindenschrage/cs109b-finalproject/main/dataframe.csv'
url = '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/dataframe.csv'
df = pd.read_csv(url)
df.head()

top_layers = []

tweet_text = list(df['Tweet'])
with torch.no_grad():
  for tweet in tweet_text:
      tokens = tokenizer(tweet, return_tensors='pt', padding=True).to('cuda')
      output = base_model(**tokens)
      #sentence_embeddings = outputs.hidden_states[-1].mean(dim=1)
      last_hidden_state = output.last_hidden_state.mean(dim=1)
      top_layers.append(last_hidden_state.cpu().detach().numpy())
      #last_token_hidden_state = last_hidden_state[:, -1, :]
      #top_layers.append(last_token_hidden_state.cpu().detach().numpy()) 

top_layer_pickle_path = '/n/home09/lschrage/projects/cs109b/top_layers.pkl'

with open(top_layer_pickle_path, 'wb') as f:
    pickle.dump(top_layers, f)

with open(top_layer_pickle_path, 'rb') as f:
    top_layers_loaded = pickle.load(f)

tweet_annotation = list(df['TweetAvgAnnotation'])

class SentimentRegressionModel(keras.Model):
    def __init__(self):
        super(SentimentRegressionModel, self).__init__()
        self.dense1 = keras.layers.Dense(200, activation='relu', kernel_regularizer=l2(0.4))  
        self.dropout1 = Dropout(0.5)
        self.dense2 = keras.layers.Dense(150, activation='relu', kernel_regularizer=l2(0.4)) 
        self.dropout2 = Dropout(0.5)
        self.dense3 = keras.layers.Dense(150, activation='relu', kernel_regularizer=l2(0.4)) 
        self.dropout3 = Dropout(0.5)
        self.dense4 = keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x) 
        x = self.dense3(x)
        x = self.dropout3(x) 
        outputs = self.dense4(x)
        return outputs
    

top_layers_array = np.vstack(top_layers_loaded)
scaler = StandardScaler()
top_layers_scaled = scaler.fit_transform(top_layers_array)

tweet_annotation_array = np.array(tweet_annotation)

X_train1, X_test, y_train1, y_test = train_test_split(
    top_layers_array,
    tweet_annotation_array,
    test_size=0.1,
    random_state=109
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train1,
    y_train1,
    test_size=0.2,
    random_state=109
)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-6,
    decay_steps=20000,
    decay_rate=0.99)

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)


opt = keras.optimizers.Adam(learning_rate=lr_schedule)
model = SentimentRegressionModel()
model.compile(optimizer=opt, loss='mse', metrics=['mse'])
history = model.fit(
    X_train1, y_train1,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=8,
    callbacks=[early_stopping_monitor]
)
history_df = pd.DataFrame(history.history)
history_df.to_csv('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama_regression/history.csv', index=False)


def plot_loss(history, path):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(path)

plot_loss(history, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama-regression/loss.png')

def plot_mse(history, path):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(path)

plot_mse(history,'/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama-regression/mse.png')

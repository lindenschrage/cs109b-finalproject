# -*- coding: utf-8 -*-
"""cs109b-llama-regression-head.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lrujJWIGh4E79eMg3utnEq9hdQ0X4ssv
"""

import transformers
from transformers import LlamaModel, LlamaTokenizer
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from torch.nn import MSELoss

access_token = "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"

base_model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token).to('cuda')

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, return_tensors = 'tf')
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
      output = base_model(**tokens, return_dict=True)
      last_hidden_state = output.last_hidden_state
      last_token_hidden_state = last_hidden_state[:, -1, :]
      top_layers.append(last_token_hidden_state)

top_layers = [tensor.cpu().detach().numpy() for tensor in top_layers]

print(top_layers[:5])

tweet_annotation = list(df['TweetAvgAnnotation'])

print(tweet_annotation[:5])

class SentimentRegressionModel(keras.Model):
    def __init__(self):
        super(SentimentRegressionModel, self).__init__()
        self.dense1 = keras.layers.Dense(4096, activation='relu')
        self.dense2 = keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        outputs = self.dense2(x)
        return outputs

top_layers_array = np.vstack(top_layers)
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
    test_size=0.1,
    random_state=109
)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)

model = SentimentRegressionModel()
model.compile(optimizer=opt, loss='mse', metrics=['mse'])
history = model.fit(X_train1, y_train1, validation_data=(X_test, y_test), epochs=100, batch_size=8)

def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)
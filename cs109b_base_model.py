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

url = 'https://raw.githubusercontent.com/lindenschrage/cs109b-data/main/dataframe.csv'
df = pd.read_csv(url)

from transformers import BertTokenizer, BertModel
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

def get_embedding(text):
    wrapped_input = bert_tokenizer(text, max_length=15, add_special_tokens=True, truncation=True,
                                   padding='max_length', return_tensors="pt")
    with torch.no_grad():
      output = bert_model(**wrapped_input)
      last_hidden_state, pooler_output = output[0], output[1]
    return pooler_output

df['Tweet-tokens'] = df['Tweet'].apply(get_embedding)
top_layers = list(df['Tweet-tokens'])
tweet_annotation = list(df['TweetAvgAnnotation'])

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
history_df.to_csv('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/basemodel/history.csv', index=False)


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

plot_loss(history, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/base-model/loss.png')

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

plot_mse(history,'/n/home09/lschrage/projects/cs109b/cs109b-finalproject/base-model/mse.png')

bert_model.compile(optimizer=opt, loss='mse', metrics=['mse'])
history = bert_model.fit(X_train1, y_train1, validation_data=(X_test, y_test), epochs=10, batch_size=8)
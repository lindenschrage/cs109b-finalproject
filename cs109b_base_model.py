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
'''
from transformers import AutoTokenizer, AutoModel

bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = AutoModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True).to('cuda')  
inputs = bert_tokenizer(list(df['Tweet']), add_special_tokens=True, truncation=True, padding=True, return_tensors="pt").to('cuda')

with torch.no_grad():
    outputs = bert_model(**inputs)
    last = last_hidden_states[:, 0, :]  # taking  CLS token embeddings from the last layer
    sec = hidden_states[-2][:, 0, :]    # 2nd to last layer
    thr = hidden_states[-3][:, 0, :]    
    frth = hidden_states[-4][:, 0, :] 
    embeddings = last + sec + thr + frth 

df['Tweet-tokens'] = embeddings.cpu().numpy().tolist()

top_layers = list(df['Tweet-tokens'])
'''
tweet_annotation = list(df['TweetAvgAnnotation'])

top_layer_pickle_path = '/n/home09/lschrage/projects/cs109b/top_layers_base.pkl'
'''
with open(top_layer_pickle_path, 'wb') as f:
    pickle.dump(top_layers, f)
'''
with open(top_layer_pickle_path, 'rb') as f:
    top_layers_loaded = pickle.load(f)

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
        self.dense1 = keras.layers.Dense(800, activation='relu', kernel_regularizer=l2(0.01))  
        #self.dropout1 = Dropout(0.5)
        self.dense2 = keras.layers.Dense(800, activation='relu', kernel_regularizer=l2(0.01)) 
        #self.dropout2 = Dropout(0.5)
        self.dense3 = keras.layers.Dense(800, activation='relu', kernel_regularizer=l2(0.01)) 
        #self.dropout3 = Dropout(0.5)
        self.dense4 = keras.layers.Dense(1, activation='linear')
#500, 300, 150
    def call(self, inputs):
        x = self.dense1(inputs)
        #x = self.dropout1(x)
        x = self.dense2(x)
        #x = self.dropout2(x) 
        x = self.dense3(x)
        #x = self.dropout3(x) 
        outputs = self.dense4(x)
        return outputs
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=20000,
    decay_rate=0.99)

early_stopping_monitor = EarlyStopping(
    monitor='val_mse',
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
history_df.to_csv('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/basemodel-history.csv', index=False)


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

plot_loss(history, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/basemodel-loss.png')

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

plot_mse(history,'/n/home09/lschrage/projects/cs109b/cs109b-finalproject/basemodel-mse.png')


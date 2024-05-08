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
import pickle
from helper_functions import get_bert_embeddings
from helper_functions import plot_loss, plot_mse, plot_predictions_vs_actual

url = 'data/dataframe.csv'
df = pd.read_csv(url)

tweet_annotation = list(df['TweetAvgAnnotation'])

get_bert_embeddings(df, '/n/home09/lschrage/projects/cs109b/BERT_embeddings.pkl')

with open('/n/home09/lschrage/projects/cs109b/top_layers_base.pkl', 'rb') as f:
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
    initial_learning_rate=1e-6,
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

plot_loss(history, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/BERT-loss.png')

plot_mse(history,'/n/home09/lschrage/projects/cs109b/cs109b-finalproject/BERT-mse.png')

plot_predictions_vs_actual(model, X_test, y_test, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/BERT-actual-vs-predicted.png')


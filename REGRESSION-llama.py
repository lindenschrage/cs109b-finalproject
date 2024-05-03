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
from embedding_data import get_llama_embeddings
from plot_functions import plot_loss, plot_mse, plot_predictions_vs_actual

url = '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/dataframe.csv'
df = pd.read_csv(url)

get_llama_embeddings(df, '/n/home09/lschrage/projects/cs109b/llama_embeddings.pkl')
with open('/n/home09/lschrage/projects/cs109b/LLAMA_embeddings.pkl', 'rb') as f:
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
history_df.to_csv('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/llama_regression-history.csv', index=False)


plot_loss(history, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/LLAMA-regression-loss.png')

plot_mse(history,'/n/home09/lschrage/projects/cs109b/cs109b-finalproject/LLAMA-regression-mse.png')

plot_predictions_vs_actual(model, X_test, y_test, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/LLAMA-regression-actual-vs-predicted.png')


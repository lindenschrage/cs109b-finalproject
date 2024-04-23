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

'''
access_token = "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"

base_model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token).to('cuda')

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, return_tensors = 'tf')
tokenizer.pad_token_id = (0)
tokenizer.padding_side = "left"
'''
#url = 'https://raw.githubusercontent.com/lindenschrage/cs109b-finalproject/main/dataframe.csv'
url = '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/dataframe.csv'
df = pd.read_csv(url)
df.head()
'''
top_layers = []

tweet_text = list(df['Tweet'])
with torch.no_grad():
  for tweet in tweet_text:
      tokens = tokenizer(tweet, return_tensors='pt', padding=True).to('cuda')
      output = base_model(**tokens, return_dict=True)
      last_hidden_state = output.last_hidden_state
      last_token_hidden_state = last_hidden_state[:, -1, :]
      top_layers.append(last_token_hidden_state.cpu().detach().numpy()) 
'''
top_layer_pickle_path = '/n/home09/lschrage/projects/cs109b/top_layers.pkl'
'''
with open(top_layer_pickle_path, 'wb') as f:
    pickle.dump(top_layers, f)
'''
with open(top_layer_pickle_path, 'rb') as f:
    top_layers_loaded = pickle.load(f)


tweet_annotation = list(df['TweetAvgAnnotation'])


class SentimentRegressionModel(keras.Model):
    def __init__(self):
        super(SentimentRegressionModel, self).__init__()
        self.dense1 = keras.layers.Dense(600, activation='relu', kernel_regularizer=l2(0.01))  
        self.dropout1 = Dropout(0.4)
        self.dense2 = keras.layers.Dense(300, activation='relu', kernel_regularizer=l2(0.01)) 
        self.dropout2 = Dropout(0.4)
        self.dense3 = keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x) 
        outputs = self.dense3(x)
        return outputs
    
class SentimentRegressionModel1(keras.Model):
    def __init__(self):
        super(SentimentRegressionModel1, self).__init__()
        self.dense1 = keras.layers.Dense(400, activation='relu', kernel_regularizer=l2(0.15))  
        self.dropout1 = Dropout(0.5)
        self.dense2 = keras.layers.Dense(200, activation='relu', kernel_regularizer=l2(0.15)) 
        self.dropout2 = Dropout(0.5)
        self.dense3 = keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(0.15)) 
        self.dropout3 = Dropout(0.4)
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
    
class SentimentRegressionModel2(keras.Model):
    def __init__(self):
        super(SentimentRegressionModel2, self).__init__()
        self.dense1 = keras.layers.Dense(1000, activation='relu', kernel_regularizer=l2(0.02))  
        self.dropout1 = Dropout(0.4)
        self.dense2 = keras.layers.Dense(1000, activation='relu', kernel_regularizer=l2(0.02)) 
        self.dropout2 = Dropout(0.4)
        self.dense3 = keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x) 
        outputs = self.dense3(x)
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
    decay_steps=10000,
    decay_rate=0.9)

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
history_df.to_csv('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/history.csv', index=False)

####
early_stopping_monitor2 = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)
opt1 = keras.optimizers.Adam(learning_rate=lr_schedule)
model1 = SentimentRegressionModel1()
model1.compile(optimizer=opt1, loss='mse', metrics=['mse'])
history1 = model1.fit(
    X_train1, y_train1,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=8,
    callbacks=[early_stopping_monitor2]
)
history_df1 = pd.DataFrame(history1.history)
history_df1.to_csv('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/history1.csv', index=False)

early_stopping_monitor2 = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)
opt2 = keras.optimizers.Adam(learning_rate=lr_schedule)
model2 = SentimentRegressionModel2()
model2.compile(optimizer=opt2, loss='mse', metrics=['mse'])
history2 = model2.fit(
    X_train1, y_train1,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=8,
    callbacks=[early_stopping_monitor2]
)
history_df2 = pd.DataFrame(history2.history)
history_df2.to_csv('/n/home09/lschrage/projects/cs109b/cs109b-finalproject/history2.csv', index=False)


######

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

plot_loss(history, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/loss.png')
plot_loss(history1, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/loss1.png')
plot_loss(history2, '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/loss2.png')

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

plot_mse(history,'/n/home09/lschrage/projects/cs109b/cs109b-finalproject/mse.png')
plot_mse(history1,'/n/home09/lschrage/projects/cs109b/cs109b-finalproject/mse1.png')
plot_mse(history2,'/n/home09/lschrage/projects/cs109b/cs109b-finalproject/mse2.png')

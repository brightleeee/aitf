import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

print(tf.__version__)
print(sys.path)

conn = sqlite3.connect("trest_yh.db")
dataset = pd.read_sql("SELECT * FROM test2 where 건물유형='주상복합'", conn)
dataset = dataset[['용량', '세대수', '최대층수', '최저층수', '연면적']]

print(dataset.tail())

dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats2 = train_dataset.describe()
print(train_stats2)
train_stats2.pop("용량")
print(train_stats2)
train_stats2 = train_stats2.transpose()
print(train_stats2)

train_labels = train_dataset.pop('용량')
test_labels = test_dataset.pop('용량')

print("통과?")


def norm(x):
  return (x - train_stats2['mean']) / train_stats2['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
  model2 = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model2.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model2


model2 = build_model()

model2.summary()

example_batch = normed_train_data[:10]
example_result = model2.predict(example_batch)
print(example_result)


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


EPOCHS = 100
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model2.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split=0.2, verbose=0)
# callbacks=early_stop)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

loss, mae, mse = model2.evaluate(normed_test_data, test_labels, verbose=2)

print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

x_test = [[136466, 12, 5, 225]]
# print("통과")

# x_test = [[70000]]
# x_test = pd.DataFrame(x_test, columns=['연면적', '세대수'])
x_test = pd.DataFrame(x_test, columns=['연면적', '최대층수', '최저층수', '세대수'])
normed_x_test = norm(x_test)
y_predict = model2.predict(normed_x_test)
print(y_predict[0])

model2.save('model_v0_cho.h5')

x_test = [[2000, 3, 1, 100]]
x_test = pd.DataFrame(x_test, columns=['연면적', '최대층수', '최저층수', '세대수'])
normed_x_test = norm(x_test)
y_predict = model2.predict(normed_x_test).tolist()
print(y_predict[0])

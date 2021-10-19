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
# dataset = pd.read_sql('SELECT * FROM test2 where 지역="도심" and 건물유형="주상복합"', conn)
dataset = pd.read_sql('SELECT * FROM test2 where 건물유형="상가"', conn)
dataset5 = dataset[['초고속', '최대층수', '최저층수', '연면적']]
dataset5 = dataset5.dropna()

dataset6 = dataset[['전용', '최대층수', '최저층수', '연면적']]
dataset6 = dataset6.dropna()

dataset7 = dataset[['무선', '최대층수', '최저층수', '연면적']]
dataset7 = dataset7.dropna()

dataset8 = dataset[['광전화', '최대층수', '최저층수', '연면적']]
dataset8 = dataset8.dropna()

train_dataset5 = dataset5.sample(frac=0.8, random_state=0)
test_dataset5 = dataset5.drop(train_dataset5.index)

train_dataset6 = dataset6.sample(frac=0.8, random_state=0)
test_dataset6 = dataset6.drop(train_dataset6.index)

train_dataset7 = dataset7.sample(frac=0.8, random_state=0)
test_dataset7 = dataset7.drop(train_dataset7.index)

train_dataset8 = dataset8.sample(frac=0.8, random_state=0)
test_dataset8 = dataset8.drop(train_dataset8.index)

print("테스트")
print(train_dataset5)
print(train_dataset5.keys())
print(len(train_dataset5.keys()))
print(train_dataset8)
print(train_dataset8.keys())
print(len(train_dataset8.keys()))

train_stats5 = train_dataset5.describe()
train_stats5.pop("초고속")
train_stats5 = train_stats5.transpose()
train_labels5 = train_dataset5.pop('초고속')
test_labels5 = test_dataset5.pop('초고속')

train_stats6 = train_dataset6.describe()
train_stats6.pop("전용")
train_stats6 = train_stats6.transpose()
train_labels6 = train_dataset6.pop('전용')
test_labels6 = test_dataset6.pop('전용')

train_stats7 = train_dataset7.describe()
train_stats7.pop("무선")
train_stats7 = train_stats7.transpose()
train_labels7 = train_dataset7.pop('무선')
test_labels7 = test_dataset7.pop('무선')

train_stats8 = train_dataset8.describe()
train_stats8.pop("광전화")
train_stats8 = train_stats8.transpose()
train_labels8 = train_dataset8.pop('광전화')
test_labels8 = test_dataset8.pop('광전화')

print(dataset5.tail())
print(dataset6.tail())
print(dataset7.tail())
print(dataset8.tail())

def norm5(x):
  return (x - train_stats5['mean']) / train_stats5['std']

def norm6(x):
  return (x - train_stats6['mean']) / train_stats6['std']

def norm7(x):
  return (x - train_stats7['mean']) / train_stats7['std']

def norm8(x):
  return (x - train_stats8['mean']) / train_stats8['std']

normed_train_data5 = norm5(train_dataset5)
normed_test_data5 = norm5(test_dataset5)

normed_train_data6 = norm6(train_dataset6)
normed_test_data6 = norm6(test_dataset6)

normed_train_data7 = norm7(train_dataset7)
normed_test_data7 = norm7(test_dataset7)

normed_train_data8 = norm8(train_dataset8)
normed_test_data8 = norm8(test_dataset8)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset5.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='softplus')
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model5 = build_model()
model5.summary()

model6 = build_model()
model6.summary()

model7 = build_model()
model7.summary()

model8 = build_model()
model8.summary()

example_batch5 = normed_train_data5[:10]
example_result5 = model5.predict(example_batch5)
print(example_result5)

example_batch6 = normed_train_data6[:10]
example_result6 = model6.predict(example_batch6)
print(example_result6)

example_batch7 = normed_train_data7[:10]
example_result7 = model7.predict(example_batch7)
print(example_result7)

example_batch8 = normed_train_data8[:10]
example_result8 = model8.predict(example_batch8)
print(example_result8)


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

history5 = model5.fit(
  normed_train_data5, train_labels5,
  epochs=EPOCHS, validation_split=0.2, verbose=0)

history6 = model6.fit(
  normed_train_data6, train_labels6,
  epochs=EPOCHS, validation_split=0.2, verbose=0)

history7 = model7.fit(
  normed_train_data7, train_labels7,
  epochs=EPOCHS, validation_split=0.2, verbose=0)

history8 = model8.fit(
  normed_train_data8, train_labels8,
  epochs=EPOCHS, validation_split=0.2, verbose=0)

hist5 = pd.DataFrame(history5.history)
hist5['epoch'] = history5.epoch
loss, mae, mse = model5.evaluate(normed_test_data5, test_labels5, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

hist6 = pd.DataFrame(history6.history)
hist6['epoch'] = history6.epoch
loss, mae, mse = model6.evaluate(normed_test_data6, test_labels6, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

hist7 = pd.DataFrame(history7.history)
hist7['epoch'] = history7.epoch
loss, mae, mse = model7.evaluate(normed_test_data7, test_labels7, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

hist8 = pd.DataFrame(history8.history)
hist8['epoch'] = history8.epoch
loss, mae, mse = model8.evaluate(normed_test_data8, test_labels8, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

x_test = [[136466, 12, 5]]
x_test = pd.DataFrame(x_test, columns=['연면적', '최대층수', '최저층수'])
normed_x_test5 = norm5(x_test)
normed_x_test6 = norm6(x_test)
normed_x_test7 = norm7(x_test)
normed_x_test8 = norm8(x_test)
y_predict = model5.predict(normed_x_test5)
print(y_predict[0])
y_predict = model6.predict(normed_x_test6)
print(y_predict[0])
y_predict = model7.predict(normed_x_test7)
print(y_predict[0])
y_predict = model8.predict(normed_x_test8)
print(y_predict[0])

model5.save('model_0923_v2_cho.h5')
model6.save('model_0923_v2_jeon.h5')
model7.save('model_0923_v2_moo.h5')
model8.save('model_0923_v2_gwang.h5')


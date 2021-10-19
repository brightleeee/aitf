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
dataset = pd.read_sql('SELECT * FROM test2 where 건물유형="빌딩"', conn)
dataset1 = dataset[['초고속', '최대층수', '최저층수', '연면적']]
dataset1 = dataset1.dropna()

dataset2 = dataset[['전용', '최대층수', '최저층수', '연면적']]
dataset2 = dataset2.dropna()
print(dataset2)

dataset3 = dataset[['무선', '최대층수', '최저층수', '연면적']]
dataset3 = dataset3.dropna()

dataset4 = dataset[['광전화', '최대층수', '최저층수', '연면적']]
dataset4 = dataset4.dropna()

train_dataset = dataset1.sample(frac=0.8, random_state=0)
test_dataset = dataset1.drop(train_dataset.index)

train_dataset2 = dataset2.sample(frac=0.8, random_state=0)
test_dataset2 = dataset2.drop(train_dataset2.index)


train_dataset3 = dataset3.sample(frac=0.8, random_state=0)
test_dataset3 = dataset3.drop(train_dataset3.index)

train_dataset4 = dataset4.sample(frac=0.8, random_state=0)
test_dataset4 = dataset4.drop(train_dataset4.index)

train_stats = train_dataset.describe()
train_stats.pop("초고속")
train_stats = train_stats.transpose()
train_labels = train_dataset.pop('초고속')
test_labels = test_dataset.pop('초고속')

train_stats2 = train_dataset2.describe()
train_stats2.pop("전용")
train_stats2 = train_stats2.transpose()
train_labels2 = train_dataset2.pop('전용')
test_labels2 = test_dataset2.pop('전용')

train_stats3 = train_dataset3.describe()
train_stats3.pop("무선")
train_stats3 = train_stats3.transpose()
train_labels3 = train_dataset3.pop('무선')
test_labels3 = test_dataset3.pop('무선')

train_stats4 = train_dataset4.describe()
train_stats4.pop("광전화")
train_stats4 = train_stats4.transpose()
train_labels4 = train_dataset4.pop('광전화')
test_labels4 = test_dataset4.pop('광전화')

print(dataset1.tail())
print(dataset2.tail())
print(dataset3.tail())
print(dataset4.tail())

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

def norm2(x):
  return (x - train_stats2['mean']) / train_stats2['std']

def norm3(x):
  return (x - train_stats3['mean']) / train_stats3['std']

def norm4(x):
  return (x - train_stats4['mean']) / train_stats4['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_train_data2 = norm2(train_dataset2)
normed_test_data2 = norm2(test_dataset2)

normed_train_data3 = norm3(train_dataset3)
normed_test_data3 = norm3(test_dataset3)

normed_train_data4 = norm4(train_dataset4)
normed_test_data4 = norm4(test_dataset4)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='softplus')
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()

model2 = build_model()
model2.summary()

model3 = build_model()
model3.summary()

model4 = build_model()
model4.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

example_batch2 = normed_train_data2[:10]
example_result2 = model2.predict(example_batch)
print(example_result2)

example_batch3 = normed_train_data3[:10]
example_result3 = model3.predict(example_batch)
print(example_result3)

example_batch4 = normed_train_data4[:10]
example_result4 = model4.predict(example_batch)
print(example_result4)


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split=0.2, verbose=0)

history2 = model2.fit(
  normed_train_data2, train_labels2,
  epochs=EPOCHS, validation_split=0.2, verbose=0)

history3 = model3.fit(
  normed_train_data3, train_labels3,
  epochs=EPOCHS, validation_split=0.2, verbose=0)

history4 = model4.fit(
  normed_train_data4, train_labels4,
  epochs=EPOCHS, validation_split=0.2, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

hist2 = pd.DataFrame(history2.history)
hist2['epoch'] = history2.epoch
loss, mae, mse = model2.evaluate(normed_test_data2, test_labels2, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

hist3 = pd.DataFrame(history3.history)
hist3['epoch'] = history3.epoch
loss, mae, mse = model3.evaluate(normed_test_data3, test_labels3, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

hist4 = pd.DataFrame(history4.history)
hist4['epoch'] = history4.epoch
loss, mae, mse = model4.evaluate(normed_test_data4, test_labels4, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} ".format(mae))

x_test = [[10000, 50, 5]]
x_test = pd.DataFrame(x_test, columns=['연면적', '최대층수', '최저층수'])
normed_x_test = norm(x_test)
y_predict = model.predict(normed_x_test)
print(y_predict[0])

model.save('model_0923_v1_cho.h5')
model2.save('model_0923_v1_jeon.h5')
model3.save('model_0923_v1_moo.h5')
model4.save('model_0923_v1_gwang.h5')

#
# test_predictions = model.predict(normed_test_data).flatten()
#
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])
# plt.show()
#
# test_predictions2 = model2.predict(normed_test_data2).flatten()
#
# plt.scatter(test_labels2, test_predictions2)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])
# plt.show()
#
# test_predictions3 = model3.predict(normed_test_data3).flatten()
#
# plt.scatter(test_labels3, test_predictions3)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])
# plt.show()
#
# test_predictions4 = model4.predict(normed_test_data4).flatten()
#
# plt.scatter(test_labels4, test_predictions4)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])
# plt.show()
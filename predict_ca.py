import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# dataset_path

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
# raw_dataset = pd.read_csv(dataset_path, names=column_names,
#                       na_values = "?", comment='\t',
#                       sep=" ", skipinitialspace=True)
conn = sqlite3.connect("trest.db")
dataset = pd.read_sql("SELECT * FROM test2 where 건물유형='빌딩'", conn)
# dataset = dataset[['연면적', '세대수', '용량']]
dataset = dataset[['연면적', '용량']]
# dataset = raw_dataset.copy()
print(dataset.tail())

# dataset.isna().sum()

dataset = dataset.dropna()

# origin = dataset.pop('Origin')
#
# dataset['USA'] = (origin == 1)*1.0
# dataset['Europe'] = (origin == 2)*1.0
# dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()

train_stats = train_dataset.describe()
train_stats.pop("용량")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('용량')
test_labels = test_dataset.pop('용량')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 3000
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)
  # callbacks=early_stop)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))


# x_test = [[20000, 200]]


x_test = [[70000]]
# x_test = pd.DataFrame(x_test, columns=['연면적', '세대수'])
x_test = pd.DataFrame(x_test, columns=['연면적'])
normed_x_test = norm(x_test)
y_predict = model.predict(normed_x_test)
print(y_predict[0])

model.save('model_v0')


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,100])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,100])
  plt.legend()
  plt.show()

plot_history(history)

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf

train_file_path = "newdata.csv"
#np.set_printoptions(precision=3, suppress=True)
LABELS=[1,2,3,4,5]

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # Artificially small to make examples easier to show.
      label_name='y',
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

def process_continuous_data(mean, data):
  # Normalize data
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])


raw_train_data = get_dataset(train_file_path)
print(raw_train_data)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(178),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(raw_train_data, epochs=20)






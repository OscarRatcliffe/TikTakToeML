import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

# Stolen moddel
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu', input_shape=(None, 9)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

x_train = np.array(
    [[0,0,0,0,0,0,0,0,0],
    [0,0,1,2,0,0,0,0,0]])

y_train = np.array(
    [[1,0,0,0,0,0,0,0,0],
    [1,0,1,2,1,0,0,0,0]])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size=32)

model.fit(train_dataset , epochs=10)

predictions = model.predict()

predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)

# Libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading Dataset

(Xtrain, Ytrain), (Xtest, Ytest) = keras.datasets.cifar10.load_data()

# Normalizing Dataset

Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0

# Model Architecture

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Dropout(0.6))

model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Dropout(0.6))

model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Dropout(0.7))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dropout(0.6))

model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(Xtrain, Ytrain, epochs=100, validation_split=0.2, verbose=1)

# Model's Accuracy and Loss Graphs

plt.plot(history.history['accuracy'], color="red")
plt.plot(history.history['val_accuracy'], color="blue")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()

plt.plot(history.history['loss'], color="red")
plt.plot(history.history['val_loss'], color="blue")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()

# Test Data

print("Test Results:")
model.evaluate(Xtest, Ytest)
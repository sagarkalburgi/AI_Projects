# Traffic_signal_classifier

# Importing the dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image

data = []
labels = []
classes = 43
cur_path = os.getcwd()

#Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '/' + a)
            image = image.resize((32, 32))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# If Categorise the data the use categorical_crossentropy
#y_train = to_categorical(y_train, 43)
#y_test = to_categorical(y_test, 43)

# Converting the images into grayscale and normalizing
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

#X_train_gray = rgb2gray(X_train)
#X_test_gray = rgb2gray(X_test)

#X_train_gray_norm = (X_train_gray - 128)/128
#X_test_gray_norm = (X_test_gray - 128)/128

# Build a Deep CNN Model
from tensorflow.keras import datasets, layers, models
CNN = models.Sequential()

CNN.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNN.add(layers.AveragePooling2D(pool_size=2, strides=2))

CNN.add(layers.Dropout(0.2))

CNN.add(layers.Conv2D(128, (5, 5), activation='relu'))
CNN.add(layers.Conv2D(128, (5, 5), activation='relu'))
CNN.add(layers.AveragePooling2D(pool_size=2, strides=2))

CNN.add(layers.Flatten())

CNN.add(layers.Dense(256, activation='relu'))
CNN.add(layers.Dropout(0.25))
CNN.add(layers.Dense(512, activation='relu'))
CNN.add(layers.Dropout(0.2))
CNN.add(layers.Dense(43, activation='softmax'))
CNN.summary()

# Compile and Train the Deep Network Model
CNN.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_model = CNN.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    batch_size=64,
                    epochs=50,
                    verbose=1)

# Saving the model
CNN.save('Model_self_tuned_SCC.h5')

# Assess Trained CNN model performance
score = CNN.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(score[1]))

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(cnn_model.history['accuracy'], label='training accuracy')
plt.plot(cnn_model.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(cnn_model.history['loss'], label='training loss')
plt.plot(cnn_model.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# Getting confusion matrix
predicted_classes = CNN.predict_classes(X_test)
y_true = y_test

# Use this only if used to_categorical to convert into single digits
#y_true = np.argmax(y_true, axis=1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize=(25, 25))
plt.title('Confusion Matrix')
sns.heatmap(cm, annot=True)
plt.show()

# Facial Expression Recognition

# Importing the libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import utils 
import os

# Images information
train_path = 'D:/GPU testing/Facial expression recognition/train/'
test_path = 'D:/GPU testing/Facial expression recognition/test/'
for expression in os.listdir(train_path):
    print(str(len(os.listdir(train_path + expression))) + " " + expression + " " + 'images')

# Generate Training and Validation Batches
from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size = 16

#width_shift_range=0.2, height_shift_range=0.2, rescale=1./255,rotation_range=20,
#shear_range=0.2, zoom_range=0.2,fill_mode='nearest'
datagen_train = ImageDataGenerator(
                                   horizontal_flip=True
                                   )
train_generator = datagen_train.flow_from_directory(train_path,
                                                    target_size=(48, 48),
                                                    color_mode='grayscale',
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory(test_path,
                                                              target_size=(48, 48),
                                                              color_mode='grayscale',
                                                              batch_size=64,
                                                              class_mode='categorical',
                                                              shuffle=False)

# Creating a CNN Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

cnn = Sequential()

cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())

cnn.add(Dense(512, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))

cnn.add(Dense(256, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))                

cnn.add(Dense(128, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))

cnn.add(Dense(64, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))

cnn.add(Dense(7, activation='softmax'))

#opt = Adam(lr=0.0001)#, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

# Train and Evaluate Model
from IPython.display import SVG, Image
from livelossplot import PlotLossesKerasTF
import tensorflow as tf
steps_per_epochs = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

checkpoint = ModelCheckpoint('model_weights.h5', monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                              min_lr=0.00001, mode='auto')
callbacks = [PlotLossesKerasTF(), checkpoint, reduce_lr]

cnn_history = cnn.fit(x=train_generator,
                      steps_per_epoch=steps_per_epochs,
                      epochs=100,
                      validation_data=validation_generator,
                      validation_steps=validation_steps,
                      callbacks=callbacks)

# Represeting Model as JSON String
cnn_json = cnn.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(cnn_json)

# Image Classification CIFAR-10

# Import Libraries
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


# Data Preprocessing
def get_three_classes(x, y):
    count = x.shape[0]
    indices = np.random.choice(range(count), count, replace=False)
    x = x[indices]
    y = y[indices]
    y = tf.keras.utils.to_categorical(y)
    return x, y

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, y_train = get_three_classes(x_train, y_train)
x_test, y_test = get_three_classes(x_test, y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Visualising Examples
class_names = ['Aeroplane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def show_random_examples(x, y, p):
    indices = np.random.choice(range(x.shape[0]), 10, replace=False)
    x = x[indices]
    y = y[indices]
    p = p[indices]
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'
        plt.xlabel(class_names[np.argmax(p[i])], color=col)
    plt.show()
show_random_examples(x_train, y_train, y_train)
plt.figure(0)

# Shuffling the Images
#from sklearn.utils import shuffle
#X_train, y_train = shuffle(x_train, y_train)

# Creating a CNN Model

from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

cnn = Sequential()

cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
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

cnn.add(Dense(256, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))

cnn.add(Dense(512, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))

cnn.add(Dense(10, activation='softmax'))

opt = Adam(lr=0.0001)
cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

save_path = 'D:/GPU testing/Image Classification CIFAR10/models/'
# Training the Model
checkpoint = ModelCheckpoint(save_path+'model_weights.h5', monitor='val_accuracy',
                             save_weights_only=False, save_best_only=True, mode='max',
                             verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                              min_lr=0.00001, mode='auto')
early_stop = EarlyStopping(monitor='val_accuracy', patience=4)
callbacks = [checkpoint, reduce_lr, early_stop]
model = cnn.fit(x_train/255., y_train,
            validation_data=(x_test/255., y_test),
            epochs=50, batch_size=64,
            callbacks=callbacks)

# Assess Trained CNN model performance
score = cnn.evaluate(x_test/255., y_test)
print('Test Accuracy: {}'.format(score[1]))

# Plotting graphs for accuracy
plt.figure(1)
plt.plot(model.history['accuracy'], label='training accuracy')
plt.plot(model.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(2)
plt.plot(model.history['loss'], label='training loss')
plt.plot(model.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# Getting confusion matrix
predicted_classes = cnn.predict_classes(x_test/255.)
y_true = y_test

# Use this only if used to_categorical to convert into single digits
y_true = np.argmax(y_true, axis=1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize=(25, 25))
plt.title('Confusion Matrix')
sns.heatmap(cm, annot=True)
plt.show()
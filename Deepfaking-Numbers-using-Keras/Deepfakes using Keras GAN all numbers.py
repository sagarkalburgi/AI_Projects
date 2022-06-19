# Deepfakes using Keras GAN

# Importing Libraries
import tensorflow as tf
import numpy as np
import tfutils
import os

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, Reshape, LeakyReLU
from tensorflow.keras.models import Model, Sequential

def get_train_test_data(num):
    # importing the data
    
    x_train = tfutils.datasets.mnist.load_subset([num], X_train, Y_train)
    x_test = tfutils.datasets.mnist.load_subset([num], X_test, Y_test)

    x = np.concatenate([x_train, x_test], axis=0)
    
    return x_train, x_test, x

def create_discriminator():
    # Creating Discriminator
    discriminator = Sequential([Conv2D(64, 3, strides=2, input_shape=(28, 28, 1)),
                                LeakyReLU(),
                                BatchNormalization(),
    
                                Conv2D(128, 5, strides=2),
                                LeakyReLU(),
                                BatchNormalization(),
    
                                Conv2D(256, 5, strides=2),
                                LeakyReLU(),
                                BatchNormalization(),
    
                                Flatten(),
                                Dense(1, activation='sigmoid')])
    opt = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    discriminator.summary()
    return discriminator

def create_generator():
    # Creating Generator
    generator = Sequential([Dense(256, activation='relu', input_shape=(1,)),
                            Reshape((1, 1, 256)),
    
                            Conv2DTranspose(256, 5, activation='relu'),
                            BatchNormalization(),
    
                            Conv2DTranspose(128, 5, activation='relu'),
                            BatchNormalization(),
    
                            Conv2DTranspose(64, 5, strides=2, activation='relu'),
                            BatchNormalization(),
    
                            Conv2DTranspose(32, 5, activation='relu'),
                            BatchNormalization(),
    
                            Conv2DTranspose(1, 4, activation='sigmoid')])
    generator.summary()
    return generator

def create_GAN(discriminator, generator):
    # Generative Adversarial Network
    input_layer = tf.keras.layers.Input(shape=(1, ))
    gen_out = generator(input_layer)
    disc_out = discriminator(gen_out)

    gan = Model(input_layer,disc_out)

    discriminator.trainable = False
    opt = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    gan.summary()
    return gan

def training_GAN(discriminator, generator, gan, x_train, x_test, x):
    # Training the GAN
    epochs = 100
    batch_size = 64
    steps_per_epoch = int(2*x.shape[0]/batch_size)
    
    acc_list = []
    loss_list = []
    ep = []
    for e in range(0, epochs):
        for step in range(0, steps_per_epoch):
            true_examples = x[int(batch_size / 2) * step: int(batch_size / 2) * (step + 1)]
            true_examples = np.reshape(true_examples, (true_examples.shape[0], 28, 28, 1))
    
            noise = np.random.rand(int(batch_size / 2), 1)
            gen_examples = generator.predict(noise)
    
            x_batch = np.concatenate([gen_examples, true_examples], axis=0)
            y_batch = np.array([0] * int(batch_size / 2) + [1] * int(batch_size / 2))
    
            indices = np.random.choice(range(batch_size), batch_size, replace=False)
    
            x_batch = x_batch[indices]
            y_batch = y_batch[indices]
    
            discriminator.trainable = True
            discriminator.train_on_batch(x_batch, y_batch)
            discriminator.trainable = False
    
            loss, _ = gan.train_on_batch(noise, np.ones((int(batch_size / 2), 1)))
            _, acc = discriminator.evaluate(x_batch, y_batch, verbose=False)
        
        acc_list.append(acc)
        loss_list.append(loss)
        ep.append(e)
        predict_and_save_plot(generator, acc, loss, e)
    save_acc_loss_plot(acc_list, loss_list, ep)
    save_model_weights(discriminator, generator, gan, num)
    save_as_json(discriminator, generator, gan, num)
    
def predict_and_save_plot(generator, acc, loss, e):
    # Predicting the results after training
    noise = np.random.randn(1, 1)
    gen_image = generator.predict(noise)[0]
    gen_image = np.reshape(gen_image, (28, 28))
    plt.figure(1)
    plt.imshow(gen_image, cmap='gray')
    plt.xlabel('Discriminator Accuracy:{:.2f}'.format(acc))
    plt.ylabel('GAN Loss:{:.2f}'.format(loss))
    plt.savefig("D:/GPU testing/Deepfake using Keras GAN/Number "+str(num)+"/Epoch " + str(e+1) +".jpg")
    plt.show()
    
def save_acc_loss_plot(acc_list, loss_list, ep):
    # Accuracy and Loss plot per epoch training
    plt.figure(2)
    plt.plot(ep, acc_list, label="Accuracy")
    plt.plot(ep, loss_list, label="Loss")
    plt.legend(loc='best')
    plt.savefig("D:/GPU testing/Deepfake using Keras GAN/Number "+str(num)+"/Accuracy and Loss.jpg")
    plt.show()
    
def save_model_weights(discriminator, generator, gan, num):
    # Saving the Weights
    discriminator.trainable = False
    gan.save("D:/GPU testing/Deepfake using Keras GAN/Number "+str(num)+"/gan_"+str(num)+".h5")
    discriminator.trainable = True
    generator.save('D:/GPU testing/Deepfake using Keras GAN/Number '+str(num)+'/generator_'+str(num)+'.h5')
    discriminator.save('D:/GPU testing/Deepfake using Keras GAN/Number '+str(num)+'/discriminator_'+str(num)+'.h5')


def save_as_json(discriminator, generator, gan, num):
    # Saving as JSON
    discriminator.trainable = False
    gan_json = discriminator.to_json()
    with open('D:/GPU testing/Deepfake using Keras GAN/Number '+str(num)+'/gan_'+str(num)+'.json', 'w') as json_file:
        json_file.write(gan_json)
    discriminator.trainable = True
    generator_json = generator.to_json()
    with open('D:/GPU testing/Deepfake using Keras GAN/Number '+str(num)+'/generator_'+str(num)+'.json', 'w') as json_file:
        json_file.write(generator_json)    
    discriminator_json = discriminator.to_json()
    with open('D:/GPU testing/Deepfake using Keras GAN/Number '+str(num)+'/discriminator_'+str(num)+'.json', 'w') as json_file:
        json_file.write(discriminator_json)

# Loading the MNIST Dataset
(X_train, Y_train), (X_test, Y_test) = tfutils.datasets.mnist.load_data(one_hot=False)

# Setting randomw seed
np.random.seed(5)

# Training the model of the numberset
for num in range(0, 10):
    # Creating the model
    discriminator = create_discriminator()
    generator = create_generator()
    gan = create_GAN(discriminator, generator)
    
    x_train, x_test, x = get_train_test_data(num)
    os.mkdir("D:/GPU testing/Deepfake using Keras GAN/Number_"+str(num))
    print("Training Number : " + str(num))
    training_GAN(discriminator, generator, gan, x_train, x_test, x)

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:27:22 2021

@author: JoseDavid
"""

from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

num_classes = 10

(train_X, train_y), (test_X, test_y) = mnist.load_data()

#The training input vector is of the dimension [60000 X 28 X 28].
#The training output vector is of the dimension [60000 X 1].
#Each individual input vector is of the dimension [28 X 28].
#Each individual output vector is of the dimension [1].

#CNNs are designed to take input an RGB image, but MNIST dataset images are in grayscale (1 channel)

#Adjust images to be valid for the input of the CNN (The CNN model will require one more dimension)
#train_X.shape[0] = (28,28)

train_X = train_X.reshape(train_X.shape[0], 28,28,1)
test_X = test_X.reshape(test_X.shape[0], 28,28,1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')


#Normalize data 
train_X /= 255
test_X /= 255

#convert class vectors to binary class matrices (dummy variables)(one-hot encoder)
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)


# applying transformation to image 
train_gen = ImageDataGenerator(rotation_range=8, 
                               width_shift_range=0.08, 
                               shear_range=0.3, 
                               height_shift_range=0.08, 
                               zoom_range=0.08 )
test_gen = ImageDataGenerator()

train_set= train_gen.flow(train_X, train_y, batch_size=128)
test_set= test_gen.flow(test_X, test_y, batch_size=128)


#Initialize CNN
classifier = Sequential()

#Convolution layer
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1), activation="relu"))
#Pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening layer
classifier.add(Flatten())
#Full connection
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=num_classes, activation="softmax"))

classifier.compile(optimizer="adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

classifier.summary()

#Train the CNN
classifier.fit(
    train_set,
    steps_per_epoch=60000/128,
    epochs=10,
    validation_data = test_set,
    validation_steps=10000/128
  )

print("The model has been successfully trained")

classifier.save('mnist.h5')
print("Saving the model as mnist.h5")

#Evaluate the model
evaluate = classifier.evaluate(test_X, test_y, verbose=1)
print('Test loss:', evaluate[0])
print('Test accuracy:', evaluate[1])


#show predictions:
import numpy as np
predictions = classifier.predict(test_X) #Float predictions
predictions = predictions.round() #Integer predictions -> [0,0,0,0,0,1,0,0,0,0] = 5

predictions = np.argmax(predictions, axis=1) #select the index number which has a higher value in a row.

new_test_y= np.argmax(test_y, axis=1)

correct = np.sum(predictions == new_test_y)

print ("Found %d correct labels" % correct)

for i, correct in enumerate(test_y[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predictions[i], new_test_y[i]))
    plt.tight_layout()
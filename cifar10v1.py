from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout
from keras import backend as K

import wandb
from wandb.keras import WandbCallback


run = wandb.init()
config = run.config

## Import the cifar10 dataset
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

## Initializing the number of classes to 10
num_classes = 10

## Converting the target variable to 10x1 categorical array 
import keras
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## Image dimension
config.img_width, config.img_height = 32, 32

## Initializing config : epochs, batch size
config.epochs = 20
config.batch_size = 256

## Initializing input shape
input_shape = (config.img_width, config.img_height, 3)

#### Layer 1
model = Sequential()
model.add(Conv2D(64, (5, 5), input_shape= input_shape, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#### Layer 2
model.add(Conv2D(192,(5,5), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#### layer 3
model.add(Conv2D(384,(3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#### Layer 4
model.add(Conv2D(256,(5,5), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#### Layer 5
model.add(Conv2D(256,(5,5), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#### Layer 6
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))

#### Layer 7
model.add(Dense(4096))
model.add(Activation('relu'))

#### final output
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
             metrics=['accuracy'])


model.fit(x_train, y_train, 
          epochs=config.epochs, 
          batch_size= config.batch_size,
          validation_data=(x_test, y_test), 
          callbacks=[WandbCallback(data_type="image", save_model=False)])
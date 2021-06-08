import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

run = neptune.init(project='common/keras-example-project', api_token='ANONYMOUS', tags='integration')

params = {
    'num_classes': 10,
    'input_shape': (28, 28, 1),
    'epochs': 1,
    'lr': 0.1,
    'batch_size': 128,
    'validation_split' :0.1
}

# log params
run['hyparameters'] = params

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images [0,1] range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to one-hot encoded vectors 
y_train = keras.utils.to_categorical(y_train, params['num_classes'])
y_test = keras.utils.to_categorical(y_test, params['num_classes'])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(params['batch_size'])


# model
model = keras.Sequential(
    [
        keras.Input(shape = params['input_shape']),
        layers.Conv2D(32, kernel_size = (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size = (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(params['num_classes'], activation = 'softmax')

    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
    x_train, y_train, 
    batch_size=params['batch_size'], 
    epochs=params['epochs'], 
    validation_split=params['validation_split'],
    callbacks = [NeptuneCallback(run, base_namespace = 'metrics')])
    

run.stop()



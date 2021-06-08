import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

params = {
    'num_classes': 10,
    'input_shape': (28, 28, 1),
    'epochs': 1,
    'lr': 0.1
}

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images [0,1] range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, params['num_classes'])
y_test = keras.utils.to_categorical(y_test, params['num_classes'])

print()
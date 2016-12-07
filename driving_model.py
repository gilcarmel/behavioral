import cv2
import keras
import numpy as np
import os
from PIL import Image
import scipy.misc
import csv
from keras import backend as K

from keras.layers import Convolution2D, Cropping2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def load_labels():
    labels = {}
    with open('IMG/driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = os.path.basename(row[0])
            labels[key] = float(row[3])
    return labels

def load_images():
    images = {}
    path = 'IMG'
    for (_, _, filenames) in os.walk(path):
        for filename in filenames:
            if 'center' in filename:
                images[filename] = scipy.misc.imread(path + '/' + filename, flatten=False, mode='RGB')
        break
    return images


def load_data():
    steering_angles = load_labels()
    images = load_images()
    X = []
    Y = []
    for key in images:
        X.append(images[key])
        Y.append(steering_angles[key])
    return np.array(X),np.array(Y)


def preprocess_images(images, fx=1.0, fy=1.0):
    images = np.array([cv2.resize(image, (0,0), fx=fx, fy=fy) for image in images])
    return (images - 128.0) / 255.0

X_train, y_train = load_data()

X_train = preprocess_images(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33)


def train_simple_cnn():
    model = Sequential(name='simple_cnn')
    shape = X_train[0].shape
    topCropPixels = int(float(shape[0] * 0.3))
    model.add(Cropping2D(cropping=((topCropPixels, 0), (0, 0)), input_shape=shape))

    model.add(Convolution2D(24, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100, name="hidden1"))
    model.add(Activation('relu'))
    model.add(Dense(50, name="hidden1"))
    model.add(Activation('relu'))
    model.add(Dense(10, name="hidden1"))
    model.add(Activation('relu'))
    model.add(Dense(1, name='output'))

    model.summary()

    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=['mean_absolute_error'])

    write_cropped_layer_output(model)

    history = model.fit(X_train, y_train,
                        batch_size=256, nb_epoch=10,
                        verbose=1, validation_split=0.2,
                        validation_data=(X_test, y_test),
                        shuffle=True)
    print(history)
    return model


def write_cropped_layer_output(model):
    # Write out the cropping layer's output to make sure we're doing it right...
    get_0_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[0].output])
    layer_output = get_0_layer_output([X_train, 0])[0]
    cv2.imwrite("cropped.jpg", (layer_output[0] + 0.5) * 255.0)


model = train_simple_cnn()

model.save(model.name + ".h5")
with open(model.name+".json", "w") as text_file:
    print(model.to_json(), file=text_file)

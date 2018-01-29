import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121

from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Model
import os
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import pandas as pd
import warnings

print(keras.__version__)


#####################
# General Functions #
#####################


def get_test_data(img_paths):
    # reading images recursively
    # n
    data = []
    for (i, img_path) in enumerate(os.listdir(img_paths)):
        image = cv2.imread(test_set + "/" + img_path)
        data.append(image)
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(img_paths)))
    return data


def get_key_by_value(mydict, value):
    return list(mydict.keys())[list(mydict.values()).index(value)]

###################
# Model Functions #
###################


def get_vgg16():
    base_model = VGG16(include_top=False, weights=None, input_shape=(512, 512, 3))
    x = base_model.output
    # let's add a fully-connected layer
    # x = Dense(150, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dense(80, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Flatten()(x)
    predictions = Dense(10, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions), "vgg16"


def get_inceptionv3():
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    x = base_model.output
    # let's add a fully-connected layer
    # x = Dense(224, activation='relu')(x)
    x = Flatten()(x)
    # x = Dense(128)(x)
    # x = Dense(128)(x)
    # x = Dense(10, kernel_regularizer=l2(0.00001),activation='linear')(x)
    predictions = Dense(10, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions), "inceptionV3"


def get_densnet121():
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    x = base_model.output
    # let's add a fully-connected layer
    # x = Dense(224, activation='relu')(x)
    x = Flatten()(x)
    # x = Dense(128)(x)
    # x = Dense(128)(x)
    # x = Dense(10, kernel_regularizer=l2(0.00001),activation='linear')(x)
    predictions = Dense(10, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions), "DenseNet121"


def get_custom_model():
    model = Sequential()
    model.add(Conv2D(32, (11, 11), input_shape=(512, 512, 3), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(31, (3, 3), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    # model.add(Dense(512))
    # model.add(Dense(10, kernel_regularizer=l2(0.1)))
    # model.add(Activation('linear'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model, "custom"


########
# Paths#
########

# define paths
train_set = "../processed_data/train_set_cropped"
val_set = "../processed_data/val_set_cropped"
test_set = "../processed_data/test_folder"

# mobile models
models = os.listdir(train_set)
print(models)

#########
# Model #
#########

# options: {get_densNet121, get_vgg16, get_inceptionv3, get_custom_model}
model, model_name = get_densnet121()

################################
# Loss, optimizer and metrics! #
################################

# Optimizer
#adam = optimizers.Adam(lr=0.0001)
sgd = optimizers.sgd(lr=0.0001)

# Loss & metrics
#model.compile(loss='categorical_hinge', optimizer='adadelta', metrics=['accuracy'])
model.compile(loss = 'categorical_crossentropy',
                optimizer = sgd,
                metrics = ['accuracy'])


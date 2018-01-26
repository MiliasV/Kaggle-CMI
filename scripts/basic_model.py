# First Experiment
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import os
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import pandas as pd

print(keras.__version__)


def get_test_data(img_paths):
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


# define paths
train_set = "../processed_data/train_set_cropped"
val_set = "../processed_data/val_set_cropped"
test_set = "../processed_data/test_folder"

# print mobile models
models = os.listdir(train_set)
print(models)


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

# Define loss, optimizer and metrics!
#adam = optimizers.Adam(lr=0.0001)
sgd = optimizers.sgd(lr=0.001)
model.compile(loss = 'categorical_crossentropy',
                optimizer = sgd,
                metrics = ['accuracy'])

#model.compile(loss='categorical_hinge', optimizer='adadelta', metrics=['accuracy'])


datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True)
        # fill_mode='nearest')

val_datagen = ImageDataGenerator(
        rescale=1./255,

        horizontal_flip=True,
        vertical_flip = True
        # fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
        rescale=1./255
        )

batch_size = 32 #32

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

train_generator = datagen.flow_from_directory(
        train_set,  # this is the target directory
        target_size=(512, 512),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = val_datagen.flow_from_directory(
        val_set,
        target_size=(512, 512),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_set,  # this is the target directory
        target_size=(512, 512),  # all images will be resized to 150x150
        batch_size=1,
        save_format="tif",
        class_mode='categorical')

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
#model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)


model.fit_generator(train_generator,
        steps_per_epoch=30, #30
        epochs=20, #20
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=20)

# model.save_weights('first_try.h5')  # always save your weights after training or during training


# predictions
test_filenames = test_generator.filenames
print(test_filenames)

classes_dict = train_generator.class_indices
predictions = model.predict_generator(test_generator, len(test_filenames))

# Submissions
test_files = []
final_predictions = []

for i, pred in enumerate(predictions):
     #print(i, pred, test_filenames[i], np.argmax(pred))
     test_files.append(test_filenames[i].split("/")[-1])
     final_predictions.append(get_key_by_value(classes_dict, np.argmax(pred)))

submission_dir = "submission"
# # Write Submission File
df = pd.DataFrame(columns=['fname', 'camera'])
df['fname'] = test_files
df['camera'] = final_predictions
sub_file = os.path.join(submission_dir, "sgd.csv")
df.to_csv(sub_file, index=False)


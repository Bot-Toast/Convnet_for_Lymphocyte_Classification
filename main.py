# Utility imports
import glob
from typing import List

import data_visualiser as dfv
import pandas as pds



import numpy

# matplot for drawing graphs/images
import numpy as np
from keras.layers import Rescaling
from matplotlib import pyplot as pyplt

import tensorflow as tf

from tensorflow import keras

# Activation functions
from tensorflow.keras.layers import Activation

# Overfitting reduction
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# Model Optimisation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam

# Model Metrics
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix

# Image pre-processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

# Network layer apis
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten
from tensorflow.python.keras.layers import RandomFlip, RandomRotation, RandomZoom

"""
Setup GPU compute!
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("num gpus available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""

# Constructed Numpy arrays for dataset labels and samples.
train_set_labels = ["non-stressed", "stressed"]
train_set_samples = []
df = pds.DataFrame()
df.insert(0, 'file', str)
df.insert(1, 'prediction', str)
df.insert(2, 'percentage', str)

# import datasets to array

# covid infected t-cells & healthy t-cells.
train_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Train'
# Subset of the above categories.
valid_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Validation'
# A dataset with mixed unlabeled images.
control_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Test'

#  preprocessing data

"""
Preprocesses datasets, using a vgg16 (need to research this, seems commonly used in convnet).
It then pulls each file from the directory specified making sure to set:
image size, labels, and batch size.

preprocessing_function=tf.keras.applications.vgg16.preprocess_input
preprocessing_function=tf.keras.applications.resnet50.preprocess_input
preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
"""

training_set = ImageDataGenerator().\
    flow_from_directory(directory=train_file_path, target_size=(360, 363),
                        classes=['non-stressed', 'stressed'], batch_size=16)

validation_set = ImageDataGenerator(). \
    flow_from_directory(directory=valid_file_path, target_size=(360, 363),
                        classes=['non-stressed', 'stressed'], batch_size=16)

test_set = ImageDataGenerator(). \
    flow_from_directory(directory=control_file_path, target_size=(360, 363),
                        classes=['non-stressed', 'stressed'], batch_size=24,
                        shuffle=False)

imgs, labels = next(training_set)
imgss, labelss = next(test_set)
imgsss, labelsss = next(validation_set)

dfv.image_plot(imgs)
dfv.image_plot(imgss)
dfv.image_plot(imgsss)
print(labels)

"""
model = Sequential([
    Rescaling(1./255, input_shape=(360, 360, 3)),
    RandomFlip("vertical"),
    RandomRotation(0.2),
    RandomFlip("horizontal"),
    RandomZoom(0.2),
    Convolution2D(filters=6, kernel_size=(5, 5), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Convolution2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(2, activation='softmax'),
])
"""

model = Sequential([
    Rescaling(1. / 255, input_shape=(360, 363, 3)),
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    Dense(8, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(2, activation='softmax')
])


model.summary()

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(x=training_set,
                    validation_data=validation_set,
                    epochs=15,
                    verbose=2)

model.evaluate(training_set)
model.evaluate(validation_set)

"""
images = tf.keras.preprocessing.image.load_img(
    test_set, target_size=(360, 360)
)
img_array = tf.keras.preprocessing.image.img_to_array(images)
img_array = tf.expand_dims(img_array, 0) """

# Preprocess data


# Data Augmentation function with image plot


# Neural Network topology / Model Architecture


# Model configuration


# Visualises training and validation metrics

acc = (history.history['accuracy'])
val_acc = (history.history['val_accuracy'])

loss = (history.history['loss'])
val_loss = (history.history['val_loss'])

epochs_range = range(15)

pyplt.figure(figsize=(8, 8))
pyplt.subplot(1, 2, 1)
pyplt.plot(epochs_range, acc, label='Training Accuracy')
pyplt.plot(epochs_range, val_acc, label='Validation Accuracy')
pyplt.legend(loc='lower right')
pyplt.title('Training and Validation Accuracy')

pyplt.subplot(1, 2, 2)
pyplt.plot(epochs_range, loss, label='Training Loss')
pyplt.plot(epochs_range, val_loss, label='Validation Loss')
pyplt.legend(loc='upper right')
pyplt.title('Training and Validation Loss')
pyplt.show()

# Predicition

predictions = model.predict(x=test_set)
np1: list[str] = []
np2: list[str] = []
for i in range(len(predictions)):
    score = predictions[i]
    score2: str = "{}".format(train_set_labels[np.argmax(score)])
    print(f"THIS IS SCORE2: {score2}")
    # cont = test_set.filepaths[i]
    score3: str = ("{:.2f}".format(100 * np.max(score)))
    print(f"THIS IS SCORE3: {score3}")
    np1.append(score2)
    np2.append(score3)
    print('dataframe columns: ', df.columns)
    print("this image is called: {} is likely classified as: {} with a: {:.2f} percent confidence".
          format(test_set.filepaths[i], train_set_labels[np.argmax(score)], 100 * np.max(score)))


print(np.round(predictions))
df['file'] = test_set.filepaths
df['prediction'] = np1
df['percentage'] = np2
classes = np.argmax(predictions, axis=1)
print(classes)
df.to_csv(f"Results" + ".csv")





# Confusion Matrix


# Model Training & Validation


# History and Model Evaluation


# Panda Model Plot

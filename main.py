# Utility imports
import glob
import numpy

# matplot for drawing graphs/images
import numpy as np
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
train_set_labels = []
train_set_samples = []

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
"""


training_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input). \
    flow_from_directory(directory=train_file_path, target_size=(360, 360),
                        classes=['non-stressed', 'stressed'], batch_size=16)

validation_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input). \
    flow_from_directory(directory=valid_file_path, target_size=(360, 360),
                        classes=['non-stressed', 'stressed'], batch_size=16)

test_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input). \
    flow_from_directory(directory=control_file_path, target_size=(360, 360),
                        classes=[''], batch_size=10, shuffle=False)

imgs, labels = next(training_set)


def image_plot(image_array):
    fig, axes = pyplt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(image_array, axes):
        ax.imshow(img)
        ax.axis('off')
    pyplt.tight_layout()
    pyplt.show()

image_plot(imgs)
print(labels)

"""
model = Sequential([
    Convolution2D(filters=62, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(360, 360, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Dropout(0.2),
    Flatten(),
    Dense(2, activation='softmax'),
])
"""

model = Sequential([
    Dense(32, activation='relu', input_shape=(360, 360, 3)),
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=training_set,
          validation_data=validation_set,
          epochs=10,
          verbose=2)

model.evaluate(training_set)
model.evaluate(validation_set)

"""
images = tf.keras.preprocessing.image.load_img(
    test_set, target_size=(360, 360)
)
img_array = tf.keras.preprocessing.image.img_to_array(images)
img_array = tf.expand_dims(img_array, 0) """

predictions = model.predict(test_set)
for i in range(len(predictions)):
    score = tf.nn.softmax(predictions[0])
    #score2: float = score
    print("this image is likely {} with a {:.2f} percent confidence".
          format(labels[np.argmax(score)], 100*np.max(score)))




"""
# Preprocess data


# Data Augmentation function with image plot




# Neural Network topology / Model Architecture

conv_neural_net = Sequential()

conv_neural_net.addDense(64, activation='relu', input_shape=(32,))
conv_neural_net.add(BatchNormalization())

conv_neural_net.addDense(32, activation='relu')
conv_neural_net.add(BatchNormalization())

conv_neural_net.addDense(16, activation='relu')
conv_neural_net.add(BatchNormalization())

conv_neural_net.addDense(2, activation='softmax')

# Model configuration


# Predicition


# Confusion Matrix


# Model Training & Validation


# History and Model Evaluation


# Panda Model Plot """

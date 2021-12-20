# Utility imports
import glob

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

test_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input). \
    flow_from_directory(directory=control_file_path, target_size=(360, 360),
                        classes=['non-stressed', 'stressed'], batch_size=10,
                        shuffle=False)

imgs, labels = next(training_set)
imgss, labelss = next(test_set)
imgsss, labelsss = next(validation_set)

def image_plot(image_array):
    fig, axes = pyplt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(image_array, axes):
        ax.imshow(img)
        ax.axis('off')
    pyplt.tight_layout()
    pyplt.show()


image_plot(imgs)
image_plot(imgss)
image_plot(imgsss)
print(labels)



"""
model = Sequential([
    RandomFlip("horizontal", input_shape=(360, 360, 3)),
    RandomRotation(0.1),
    RandomZoom(0.1),
    Convolution2D(filters=62, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Dropout(0.2),
    Flatten(),
    Dense(2, activation='softmax'),
])
"""

model = Sequential([
    Rescaling(1./255, input_shape=(360, 360, 3)),
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    Dense(32, activation='relu'),
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

history = model.fit(x=training_set,
          validation_data=validation_set,
          epochs=10,
          verbose=2)

model.evaluate(training_set)
model.evaluate(validation_set)

acc = (history.history['accuracy'])
val_acc = (history.history['val_accuracy'])

loss = (history.history['loss'])
val_loss = (history.history['val_loss'])

epochs_range = range(10)

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

"""
images = tf.keras.preprocessing.image.load_img(
    test_set, target_size=(360, 360)
)
img_array = tf.keras.preprocessing.image.img_to_array(images)
img_array = tf.expand_dims(img_array, 0) """

predictions = model.predict(x=test_set)
print(np.round(predictions))


for i in range(len(predictions)):
    score = predictions[i]
    # score2: float = score
    print("this image is likely {} with a {:.2f} percent confidence".
          format(train_set_labels[np.argmax(score)], 100 * np.max(score)))

"""
# Preprocess data


# Data Augmentation function with image plot




# Neural Network topology / Model Architecture


# Model configuration


# Predicition


# Confusion Matrix


# Model Training & Validation


# History and Model Evaluation


# Panda Model Plot """

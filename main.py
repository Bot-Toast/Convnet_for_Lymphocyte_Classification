# Utility imports
import glob
import PIL
import numpy
import itertools
import os
import shutil
import random
from PIL import Image
import warnings
warnings.simplefilter(action='ignore', category='FutureWarning')

# matplot for drawing graphs/images
from matplotlib import pyplot as pyplt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

# Image pre-processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Network layer apis
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten

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
from sklearn.metrics import  confusion_matrix



import pathlib as plib

"""
Setup GPU compute!
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("num gpus available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""





# import datasets to array

# covid infected t-cells & healthy t-cells.
train_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\train\\'
# Subset of the above categories.
valid_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Validation'
# A dataset with mixed unlabeled images.
control_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Test\\IM positive control group'

"""
Preprocesses datasets, using a vgg16 (need to research this, seems commonly used in convnet). 
It then pulls each file from the directory specified making sure to set:
image size, labels, and batch size.
"""
training_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.prepocess_input).\
    flow_from_directory(directory=train_file_path, target_size=(360, 360),
                        classes=['stressed', 'non-stressed'], batch_size=10)

validation_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.prepocess_input).\
    flow_from_directory(directory=valid_file_path, target_size=(360, 360),
                        classes=['stressed', 'non-stressed'], batch_size=10)

test_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.prepocess_input).\
    flow_from_directory(directory=control_file_path, target_size=(360, 360),
                        classes=['stressed', 'non-stressed'], batch_size=10)
"""
# Load in data


# Preprocess data


# Data Augmentation function with image plot

def image_plot(image_array):
    fig, axes = pyplt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(image_array, axes):
        ax.imshow(img)
        ax.axis('off')
    pyplt.tight_layout()
    pyplt.show()


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
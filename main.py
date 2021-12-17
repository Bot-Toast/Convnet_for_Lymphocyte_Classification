import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

# Network layer apis
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D

# Overfitting reduction
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# Model Optimisation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

# Model Metrics
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from tensorflow.keras.metrics import categorical_crossentropy

# matplot for drawing graphs/images
from matplotlib import pyplot as pyplt


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


# Panda Model Plot

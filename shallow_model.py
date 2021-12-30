# Utility imports
import data_visualiser as dfv
from utility_functions import prediction_result_appender as pra
import tensorflow as tf

# Overfitting reduction in a model.
from tensorflow.keras.layers import Dropout, BatchNormalization

# Model Optimisation
from tensorflow.keras.optimizers import Adam, SGD, Nadam

# Loss functions.
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy

# Image pre-processing and Data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model type
from tensorflow.keras.models import Sequential

# Network layer types.
from tensorflow.keras.layers import Dense, Convolution2D, MaxPool2D, AvgPool2D, Flatten

"""
REMOVE RUN_INT FROM FUNCTIONS WHEN NOT LOOP TESTING
"""
def run_dense_net(run_int):
    # covid infected t-cells & healthy t-cells.
    train_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Train'
    # Subset of the above categories.
    valid_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Validation'
    # A dataset with mixed evaluation images.
    control_file_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Test'

    # set epoch
    epoch_max = 10

    """
    The ImageDataGenerator parameters augment data, to add variance to the model.
    This reduces overfitting. The added Flow_from function iterates files in a directory.

    rescale=1 / 255.,
                                        vertical_flip=True,
                                        horizontal_flip=True,
                                        rotation_range=90,
                                        zoom_range=0.15,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.15,
                                        brightness_range=[0.1, 1.2]
    """

    training_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                                      rescale=1 / 255.). \
        flow_from_directory(directory=train_file_path, target_size=(360, 360),
                            classes=['non-stressed', 'stressed'], batch_size=12)

    validation_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                                        rescale=1 / 255.). \
        flow_from_directory(directory=valid_file_path, target_size=(360, 360),
                            classes=['non-stressed', 'stressed'], batch_size=12)

    test_set = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input). \
        flow_from_directory(directory=control_file_path, target_size=(360, 360),
                            classes=['non-stressed', 'stressed'], batch_size=24,
                            shuffle=False)

    """
    These are here just to visualise the type of data going in.
    May not be required at a later stage or should be reduced at the very least.
    """

    """
    imgs, labels = next(training_set)
    imgss, labelss = next(test_set)
    imgsss, labelsss = next(validation_set)

    dfv.image_plot(imgs)
    dfv.image_plot(imgss)
    dfv.image_plot(imgsss)
    print(labels)
    """

    """
    Neural Network topology / Model Architecture
    Filters and layer amounts were arbitrarily chosen initially,
    but over time have been modified to help the model fit.
    These numbers still require a good tuning.
    This model was based on LeNet and uses layer stacking of 3x1/1x3 vs 5x5 to save computation time. 
    """

    conv_net = Sequential()
    conv_net.add(Dense(16, activation='relu', input_shape=(360, 360, 3))),
    conv_net.add(Dense(64, activation='relu')),
    conv_net.add(Dense(32, activation='relu')),
    conv_net.add(Dense(16, activation='relu')),
    conv_net.add(Dropout(0.25)),
    conv_net.add(Flatten()),
    conv_net.add(Dense(2, activation="softmax"))

    # Ronsil.
    conv_net.summary()

    # Model Configuration / Hyper Parameters
    conv_net.compile(optimizer=Adam(learning_rate=0.00015),
                     loss=categorical_crossentropy,
                     metrics=['accuracy'])

    model_history = conv_net.fit(x=training_set,
                                 validation_data=validation_set,
                                 epochs=epoch_max,
                                 verbose=2)
    conv_net.save('Models/my_model')

    # I think there is too many comments at this point, but I can't stop.
    conv_net.evaluate(training_set)
    conv_net.evaluate(validation_set)

    # It displays nice graphs. https://imgflip.com/i/5z8zsv
    dfv.plot_results(epoch_max, model_history, run_int)

    # Inference call.
    predictions = conv_net.predict(x=test_set)

    # Results fed through this function that writes to disk.
    pra(predictions, test_set, run_int)

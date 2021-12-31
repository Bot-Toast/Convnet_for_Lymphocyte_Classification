
"""
model = keras.models.load_model('E:\\Dissertation\\Dissertation_code\\s203594_DeepLearning_Image_classifier\\Models\\v1 '
                                'Conv_Net\\my_model v1')
control_file_path = 'E:\\Dissertation\\Dissertation_code\\s203594_DeepLearning_Image_classifier\\Datasets\\Test'
test_set = ImageDataGenerator(). \
    flow_from_directory(directory=control_file_path, target_size=(360, 360),
                        classes=['non-stressed', 'stressed'], batch_size=24,
                        shuffle=False)

for i in range(30):
    predict = model.predict(test_set)
    utility_functions.prediction_result_appender(predict, test_set, i)
"""


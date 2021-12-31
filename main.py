import convolutional_model
import shallow_model
import os
import time

"""
Disables GPU for Testing
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

augmentexample = ImageDataGenerator(brightness_range=[0.05, 0.3]
                                    )

example_image = load_img('LY_335097.jpg')
i_2_arr = img_to_array(example_image)
i_2_arr = i_2_arr.reshape((1,) + i_2_arr.shape)

i = 0
for batch in augmentexample.flow(i_2_arr, batch_size=1,
                                 save_to_dir=f'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys',
                                 save_prefix=f'image{i}',
                                 save_format='png'):
    i += 1
    if i > 1:
        break

for i in range(1):
    convolutional_model.run_conv_net()

# shallow_model.run_dense_net()


# Confusion Matrix TBA .

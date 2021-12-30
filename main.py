import convolutional_model
import shallow_model
import os
import time

"""
Disables GPU for Testing
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



for i in range(1):
    convolutional_model.run_conv_net()

# shallow_model.run_dense_net()


# Confusion Matrix TBA .

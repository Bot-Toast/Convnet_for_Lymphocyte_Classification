# Convnet_for_Lymphocyte_Classification

Hey, hope all is well.
This is code for a prototype deep-learning model.
HIGHLY RECOMMEND SETTING UP GPU UTILISATION IF YOU HAVE A CUDA ENABLE CARD Nvidia 1660ti+
Otherwise, you might pass into the next life before doing anything significant!

The CNN model is in convolutional_model.py.
Point file paths towards your own datasets.
Play around with hyperparameters and augmentation values to see what happens.
Call convolutional_model.run_conv_net() in main to execute the model.
To save a model conv_net.save('filepath')
To load a model conv_net.load('filepath')
To predict with that model conv_net.predict(x=test_set).

It is that simple.

Regards,
BT.

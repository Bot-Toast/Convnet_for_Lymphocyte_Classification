# https://www.youtube.com/watch?v=qFJeN9V1ZsI /keras tutorials

"""

start by pre-processing data
Setting up the weights and attributes
Set up probabilities
Set up the training functions
Set up validation functions
Set up testing functions

Create a data pipeline to correct mistakes within the Neural network-
Look at Activation functions:
    Sigmoid
    Relu
Look at Optimisers:
    SGD
Loss Functions:
    Sparse Categorical Cross Entropy
    Mean Square Error
Notes:
    Single pass of data is called an Epoch.

Steps to take for the dissertation:
    build a prototype
    fill out a tech milestone

    for the model consider:
        what the data is?
            what bias it contains?
                any artifacts?
                poor quality images?
                Jpeg vs Bmp vs PNG
                can we make these 'issues' better or reduce the effect on the process?
        Pre-processing methodologies?
            how is it being pre-processed?
            why is it being pre-processed?
            what pre-processing techniques are being utilised?
            when should the data be pre-processed?
        Learning requirements?
            What are we trying to predict or ascertain from the images?
            how can the model be programmed to fulfil this?
            what other requirements are required?
                pipeline to correct mistakes.
        Learning style?
            Supervised
            Unsupervised
            Semi-Supervised

            Sequential vs ?
            Dense vs ?
                Parameters to use in the layers.
                    Greyscale? 2
                    RGB? 3
                    how many nodes?
                        more?
                        less?
                        what are the effects of these options?
                    how many layers?
                        more?
                        less?
                        what are the effects of these options?
            What Activation function will be used?
                Relu vs Sigmoid vs ? (see week 10 slides for more info)
            What optimiser will be used?
                Adam vs SGD vs ?
            What Loss function will be used?
                SCCentropy vs MSE vs ?
            Learning Rate
                how was the learning rate determined?
                what effects on the project did specific learning rates have?
                what is the importance of setting the correct learning rate?
            How will the data sets be organised and structured?
                how does keras/tensorflow expect the data to come?
            How will the validation set be constructed?
                Using a separate data set?
                Or using a partition of data within the training set?
                Explain why these methods are beneficial or not to this model.
            How this project countered overfitting?
                how did the project know it was overfitting?
                what were the effects of overfitting on this project?
                what techniques/countermeasures can be implemented to avoid?
                    learning rate testing
                    add MOAR data
                    Data augmentation



"""
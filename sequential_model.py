
"""
conv_net = Sequential([
    Rescaling(1. / 255, input_shape=(360, 363, 3)),
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    Dense(8, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(2, activation='softmax')
])
"""
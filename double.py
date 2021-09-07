"""
Creating a model which will take a number and return 2 * the number
Kind of useless program but it shows how to create a model and train it
"""

# importing tensorflow, numpy and keras (the library we are using)
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Creating a simple sequential model with one layer
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Using the adam optimizer with custom learning rate
# 0.001 seems like a good learning rate for this problem
optimizer_function = tf.keras.optimizers.Adam(
    learning_rate=0.001,
)

# Using mean_squared_error as the loss function
# https://en.wikipedia.org/wiki/Mean_squared_error
loss_function = tf.keras.losses.MeanSquaredError()

# adding the loss and optimizer functions to the model
model.compile(optimizer=optimizer_function, loss=loss_function)

# Creating training data

# Input is number from 0 to 10_000 (so 1, 2, 3, 4... and so on)
inputs = [float(i) for i in range(0, 10000)]

# Output is 2 * input (so 0, 2, 4, 6, 8... and so on)
outputs = [float(i) * 2 for i in range(0, 10000)]

#  Converting the input and output data to numpy arrays
xs = np.array(inputs, dtype=float)
ys = np.array(outputs, dtype=float)


#  Train the model with the inputs and outputs
#  30 epochs seems like a good number for this problem
model.fit(xs, ys, epochs=30)

# Using custom hardcoded training data
# Using random number and test function would be a better way to do this
print(f"Expected output is 32.00 we got: {model.predict([16.00])}")
print(f"Expected output is 200 we got: {model.predict([100.00])}")
print(f"Expected output is 500 we got: {model.predict([250.00])}")
print(f"Expected output is 1002 we got: {model.predict([501.00])}")


# Here's the output i got

#  Expected output is 32.00 we got: [[33.312225]]
# Expected output is 200 we got: [[201.29579]]
# Expected output is 500 we got: [[501.26645]]
# Expected output is 1002 we got: [[1003.21735]]

# Not bad, could be better though.
# The loss decreases very very very quickly.

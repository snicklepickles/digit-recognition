import tensorflow as tf
import numpy as np

"""
Loads the MNIST dataset
x_train is a NumPy array of grayscale image data
y_train is a NumPy array of digit labels (integers in range 0-9)
"""
(x_train, y_train) = tf.keras.datasets.mnist.load_data()[0]

# Reshapes x_train to a 2D array of 60000 rows (images) and 784 columns (28 x 28 pixels per image)
features = np.reshape(x_train, (60000, 784))

# Reshapes y_train to a 1D array of 60000 rows (labels)
targets = np.reshape(y_train, (60000,))

print("Classes: " + str(np.unique(targets)))
print("Features' shape " + str(features.shape))
print("Targets' shape " + str(targets.shape))
print("min: " + str(np.min(features)), "max: " + str(np.max(features)))

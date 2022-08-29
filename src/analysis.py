import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

"""
Loads the MNIST dataset
X is a NumPy array of grayscale image data
y is a NumPy array of digit labels (integers in range 0-9)
"""

(X, y) = tf.keras.datasets.mnist.load_data()[0]

# Reshapes x_train to a 2D array of 6000 rows (images) and 784 columns (28 x 28 pixels per image)
features = np.reshape(X, (60000, 784))[:6000, :]

# Reshapes y_train to a 1D array of 6000 rows (labels)
targets = np.reshape(y, (60000,))[:6000]

# Splits the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=40)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("Proportion of samples per class in train set:")
s = pd.Series(y_train)
print(round(s.value_counts(normalize=True), 2))

# print("Classes: " + str(np.unique(targets)))
# print("Features' shape " + str(features.shape))
# print("Targets' shape " + str(targets.shape))
# print("min: " + str(np.min(features)), "max: " + str(np.max(features)))

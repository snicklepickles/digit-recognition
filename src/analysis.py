import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# x is an array of grayscale image data
# y is an array of digit labels (integers in range 0-9)
(x, y) = tf.keras.datasets.mnist.load_data()[0]

# Reshapes x to a 2D array (number of images x pixels per image)
X = np.reshape(x, (60000, 28 * 28))

# Splits the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X[:6000], y[:6000], test_size=0.3, random_state=40)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("Proportion of samples per class in train set:")
ds = pd.Series(y_train)
print(round(ds.value_counts(normalize=True), 2))

# print("Classes: " + str(np.unique(targets)))
# print("Features' shape " + str(features.shape))
# print("Targets' shape " + str(targets.shape))
# print("min: " + str(np.min(features)), "max: " + str(np.max(features)))

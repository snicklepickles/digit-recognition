import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# x is an array of grayscale image data
# y is an array of digit labels (integers in range 0-9)
(x, y) = tf.keras.datasets.mnist.load_data()[0]

# Reshapes x to a 2D array (number of images x pixels per image)
X = np.reshape(x, (60000, 28 * 28))

# Splits the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X[:6000], y[:6000], test_size=0.3, random_state=40)


# Trains a classifier and returns its precision score
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = precision_score(target_test, y_pred, average='macro')
    print(f'Model: {model}\nAccuracy: {score.round(2)}\n')
    return score


# Trains four different classifiers and prints the one with the highest precision score
highest_score = 0
best_model = None
for classifier in (KNeighborsClassifier(), DecisionTreeClassifier(random_state=40), LogisticRegression(random_state=40), RandomForestClassifier(random_state=40)):
    score = fit_predict_eval(classifier, x_train, x_test, y_train, y_test)
    if score > highest_score:
        highest_score = score
        best_model = type(classifier).__name__
print(f'The answer to the question: {best_model} - {highest_score.round(3)}')

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import Normalizer

# x is an array of grayscale image data
# y is an array of digit labels (integers in range 0-9)
(x, y) = tf.keras.datasets.mnist.load_data()[0]

# Reshapes x to a 2D array (number of images x pixels per image)
X = np.reshape(x, (60000, 28 * 28))

# Splits the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X[:6000], y[:6000], test_size=0.3, random_state=40)

# Normalises the training data
x_train = Normalizer().fit_transform(x_train)
x_test = Normalizer().fit_transform(x_test)


# Trains a classifier and returns its precision score
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = precision_score(target_test, y_pred, average='macro')
    print(f'Model: {model}\nAccuracy: {score.round(2)}\n')
    return score


# Trains four different classifiers and prints the one with the highest precision score
scores = []
for classifier in (KNeighborsClassifier(), DecisionTreeClassifier(random_state=40), LogisticRegression(random_state=40), RandomForestClassifier(random_state=40)):
    score = fit_predict_eval(classifier, x_train, x_test, y_train, y_test)
    scores.append((type(classifier).__name__, score))
scores.sort(key=lambda x: x[1], reverse=True)
print(f'Two best-scoring models: {scores[0][0]}: {scores[0][1].round(3)}, {scores[1][0]}: {scores[1][1].round(3)}')

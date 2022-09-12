import pickle
from cv2 import imread, resize


def predict(image_path):
    # Loads the best models from disk
    knn_model = pickle.load(open('knn_model.sav', 'rb'))
    rf_model = pickle.load(open('rf_model.sav', 'rb'))

    # Reads the image from disk
    image = imread(image_path, 0)

    # Resizes the image to 20 x 20 pixels
    image = resize(image, (20, 20))

    # Resizes the image to 28 x 28 pixels
    image = resize(image, (28, 28))

    # Reshapes the image to a 1D array
    image = image.reshape(1, 28 * 28)

    # Normalises the image
    image = image / 255

    # Predicts the digit using the two best models
    knn_prediction = knn_model.predict(image)[0]
    rf_prediction = rf_model.predict(image)[0]

    # Returns the predictions
    return knn_prediction, rf_prediction

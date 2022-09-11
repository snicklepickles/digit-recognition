import pickle
import cv2

# Loads trained models
knn_model = pickle.load(open('knn_model.sav', 'rb'))
rf_model = pickle.load(open('rf_model.sav', 'rb'))

# Reads image as grayscale (black = 0, white = 1)
img = (cv2.imread('test_digit.jpg', 0) / 255).reshape(1, 784)

# Predicts digit from image
knn_result = knn_model.predict(img)
rf_result = rf_model.predict(img)
print(f'K-Nearest Neighbour predicts: {knn_result[0]}')
print(f'Random Forest predicts: {rf_result[0]}')

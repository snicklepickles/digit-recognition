from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle


# Tunes the hyperparameters of the two best-scoring models (KNeighborsClassifier and RandomForestClassifier)
def train(x_train, y_train):
    knn_cv = GridSearchCV(KNeighborsClassifier(),
                          {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']},
                          scoring='accuracy', n_jobs=-1)
    rf_cv = GridSearchCV(RandomForestClassifier(random_state=40),
                         {'n_estimators': [300, 500], 'max_features': ['sqrt', 'log2'],
                          'class_weight': ['balanced', 'balanced_subsample']}, scoring='accuracy', n_jobs=-1)
    knn_cv.fit(x_train, y_train)
    rf_cv.fit(x_train, y_train)
    print(f'K-nearest neighbours algorithm\nbest estimator: {knn_cv.best_estimator_}\naccuracy: {knn_cv.best_score_.round(3)}')
    print(f'Random forest algorithm\nbest estimator: {rf_cv.best_estimator_}\naccuracy: {rf_cv.best_score_.round(3)}')

    # Saves the best models to disk
    pickle.dump(knn_cv.best_estimator_, open('../models/knn_model.sav', 'wb'))
    pickle.dump(rf_cv.best_estimator_, open('../models/rf_model.sav', 'wb'))

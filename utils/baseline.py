import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_regression_prediction(X_train, Y_train, group_labels_train, X_test, Y_test, variant='full time series'):
    """
    Performs logistic regression to predict group labels based on the provided features.
    
    Parameters:
    - X_train: np.ndarray, shape (n_samples, n_features), training input features
    - Y_train: np.ndarray, shape (n_samples, n_features), training output features
    - group_labels_train: np.ndarray, shape (n_samples,), training group labels
    - X_test: np.ndarray, shape (n_samples, n_features), testing input features
    - Y_test: np.ndarray, shape (n_samples, n_features), testing output features
    - group_labels_test: np.ndarray, shape (n_samples,), testing group labels
    - variant: str, either 'full time series' or 'distribution moments' to determine
      the feature representation for logistic regression.
      
    Returns:
    - predictions: np.ndarray, predicted group labels for the test set"""
    model = LogisticRegression()
    if variant == 'full time series':
        # concatenate X_train and Y_train as follows
        # row 1 [x1, x2, ..., xT, y1, y2, ..., yT]
        time_series_train = np.hstack((X_train, Y_train))
        model.fit(time_series_train, group_labels_train)

        time_series_test = np.hstack((X_test, Y_test))
        return model.predict(time_series_test)

    elif variant == 'distribution moments':
        # Compute distribution moments (mean and std) for each patient
        mean_X_train = np.mean(X_train, axis=1)
        std_X_train = np.std(X_train, axis=1)
        mean_Y_train = np.mean(Y_train, axis=1)
        std_Y_train = np.std(Y_train, axis=1)
        # Concatenate the moments to form the feature vector for each patient
        moments_train = np.column_stack((mean_X_train, std_X_train, mean_Y_train, std_Y_train))
        model.fit(moments_train, group_labels_train)

        mean_X_test = np.mean(X_test, axis=1)
        std_X_test = np.std(X_test, axis=1)
        mean_Y_test = np.mean(Y_test, axis=1)
        std_Y_test = np.std(Y_test, axis=1)
        moments_test = np.column_stack((mean_X_test, std_X_test, mean_Y_test, std_Y_test))
        return model.predict(moments_test)
    else:
        raise ValueError("Invalid variant. Choose 'full time series' or 'distribution moments'.")
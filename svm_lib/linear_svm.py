import numpy as np

def train_svm(X, y, C=1.0, lr=0.01, epochs=1000):
    """
    Train a linear SVM using gradient descent.

    X: training data, shape (n_samples, n_features)
    y: labels, must be -1 or +1 (not 0/1)
    C: regularization strength
    lr: learning rate
    epochs: number of passes over the data
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for epoch in range(epochs):
        dw = w.copy()
        db = 0.0

        for i in range(n_samples):
            margin = y[i] * (np.dot(X[i], w) + b)
            if margin < 1:
                dw -= C * y[i] * X[i]
                db -= C * y[i]

        w -= lr * dw
        b -= lr * db

    return w, b


def predict(X, w, b):
    """Return predicted class (-1 or +1) for each row in X."""
    return np.sign(X @ w + b)
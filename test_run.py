import numpy as np
from sklearn.datasets import make_blobs
from svm_lib.linear_svm import train_svm, predict

X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

w, b = train_svm(X, y, C=1.0, lr=0.001, epochs=1000)

preds = predict(X, w, b)
accuracy = np.mean(preds == y)
print(f"Training accuracy: {accuracy:.2%}")
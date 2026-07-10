from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from svm_lib.linear_svm import train_svm, predict

# Generate data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

# Train my SVM
w, b = train_svm(X, y, C=1.0, lr=0.001, epochs=1000)
margins = y * (X @ w + b)
print("Number of points violating margin (<1):", np.sum(margins < 1))
print("Min margin value:", margins.min())

# Train sklearn's version on the same data
sklearn_model = SVC(kernel='linear', C=1.0)
sklearn_model.fit(X, y)

# Compare accuracy
my_preds = predict(X, w, b)
sklearn_preds = sklearn_model.predict(X)

print(f"My SVM accuracy: {np.mean(my_preds == y):.2%}")
print(f"sklearn SVM accuracy: {np.mean(sklearn_preds == y):.2%}")

# Compare the actual boundary sklearn found
w_sklearn = sklearn_model.coef_[0]
b_sklearn = sklearn_model.intercept_[0]
print(f"My w: {w}, My b: {b}")
print(f"sklearn w: {w_sklearn}, sklearn b: {b_sklearn}")

# Train sklearn's version on the same data
sklearn_model = SVC(kernel='linear', C=1.0)
sklearn_model.fit(X, y)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', label='Data')

x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)



# My boundary: w[0]*x + w[1]*y + b = 0  ->  solve for y
y_vals_mine = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals_mine, 'k-', linewidth=2, label='My SVM')

# sklearn's boundary
y_vals_sklearn = -(w_sklearn[0] * x_vals + b_sklearn) / w_sklearn[1]
plt.plot(x_vals, y_vals_sklearn, 'g--', linewidth=2, label='sklearn SVM')

plt.title("My SVM vs sklearn SVM — Decision Boundary Comparison")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("svm_comparison.png")
plt.show()


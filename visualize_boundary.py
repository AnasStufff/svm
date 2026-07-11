import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from svm_lib.linear_svm import train_svm, predict

# Generate data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

#Train 
w, b = train_svm(X, y, C=1.0, lr=0.001, epochs=100)

# Plot the points
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

#Plot the decision boundary: w[0]*x + w[1]*y + b = 0  ->  solve for y
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, 'k-', label='Decision boundary')

#Plot the margins(offset by 1/||w|| on each side)
margin = 1 / np.linalg.norm(w)
y_vals_up = y_vals + margin
y_vals_down = y_vals - margin
plt.plot(x_vals, y_vals_up, 'k--', alpha=0.5)
plt.plot(x_vals, y_vals_down, 'k--', alpha=0.5)

plt.title("My SVM Decision Boundary")
plt.legend()
plt.savefig("my_svm_boundary.png")
plt.show()



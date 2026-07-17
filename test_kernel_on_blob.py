import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from svm_lib.kernel_svm import KernelSVM

X, y =make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)
model=KernelSVM(kernel="poly", C=1.0, gamma=0.5)
model.fit(X,y)
pred=model.predict(X)
print(f"Accutacy of the model {np.mean(y==pred):.2%}")
print(f"Support vectors {len(model.alphas)} out of {len(X)}")
xx,yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100))
grid_points=np.c_[xx.ravel(), yy.ravel()]
Z=model.predict(grid_points).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title("Kernel SVM (RBF) on linearly-separable blobs")
plt.savefig("kernel_on_blobs.png")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from svm_lib.kernel_svm import KernelSVM

#generate circular data 
X, y =make_circles(n_samples=1000, factor=0.4, noise=0.05, random_state= 42)
y= np.where(y==0,-1,1)
#train the model with rbf kernel
model= KernelSVM( kernel="rbf", C=1.0, gamma=2.0)
model.fit(X,y)
#checks accuracy
preds=model.predict(X)
accuracy=np.mean(preds==y)
print(f"SVM accuracy is {accuracy:.2%}")
#visualization
xx, yy=np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 100),np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 100))
grid_points=np.c_[xx.ravel(), yy.ravel()]
Z=model.predict(grid_points)
Z=Z.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolors="k")
plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1],s=100, facecolors='none', edgecolors='green', linewidths=2, label='Support Vectors')
plt.title("Kernel SVM (RBF) — Circular Data")
plt.legend()
plt.savefig("kernel_svm_boundary.png")
plt.show()
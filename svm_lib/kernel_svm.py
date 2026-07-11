import numpy as np
import cvxopt
from svm_lib.kernels import linear_kernel, polynomial_kernel, rbf_kernel

class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=0.5, degree=3):
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0.0

    def _kernel_function(self, x1, x2):
        if self.kernel_name == 'linear':
            return linear_kernel(x1, x2)
        elif self.kernel_name == 'poly':
            return polynomial_kernel(x1, x2, degree=self.degree)
        elif self.kernel_name == 'rbf':
            return rbf_kernel(x1, x2, gamma=self.gamma)
    def fit(self, X, y):
        n_samples, n_features = X.shape
        #Compute the kernel matrix(similarity between every pair of points)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        #Build the matrices cvxopt's QP solver expects
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)
        #solving
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])
        #Identify support vectors(non-zero alphas)
        sv_threshold = 1e-5
        sv_indices = alphas > sv_threshold

        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]

        print(f"Found {len(self.alphas)} support vectors out of {n_samples} training points")

        #Compute bias b using the support vectors
        self.b = 0.0
        for i in range(len(self.alphas)):
            self.b += self.support_vector_labels[i]
            self.b -= np.sum(self.alphas * self.support_vector_labels *
                              K[sv_indices][i][sv_indices])
        self.b /= len(self.alphas)
    def predict(self, X):
        predictions = []
        for x in X:
            result = 0
            for alpha, sv_y, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                result += alpha * sv_y * self._kernel_function(x, sv)
            result += self.b
            predictions.append(np.sign(result))
        return np.array(predictions)
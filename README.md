# svm
SVM implemintation from scratch and real data testing 

# kernel comparison
On linearly-separable data, kernel choice matters even when it 'shouldn't'. RBF actively curves around clusters, using far more support vectors than necessary, while a polynomial kernel of low effective complexity naturally converges toward something close to the true linear-optimal boundary, closely matching sklearn's exact solution.

# confusion matrix aka precision 
Out of 43 actual malignant cases in the test set, the model correctly identified 41 and missed 2. A 95% recall on the class where false negatives carry the highest risk. This highlights a key limitation: a model optimized purely for overall accuracy may not minimize the error type that matters most in a clinical setting.

# svm
A from-scratch implementation of Support Vector Machines (linear and kernelized), validated against scikit-learn and applied to real tabular and image classification tasks.

# How to run it
```bash
git clone https://github.com/AnasStufff/svm.git
cd svm
python -m venv venv
venv\Scripts\Activate.ps1 # Windows
pip install -r requirements.txt
python compare_sklearn.py  # linear SVM validation
python test_kernel.py  # kernel SVM on circular data
python breast_cancer_analysis.py  # real tabular data
python mnist_classification.py # real image data
```
# Part 1: 
## linear SVM
Implemented the model using batch gradient descent on the hinge-loss objective. Compared with scikit-learn's SVC(kernel="linear"). Both achieved 100% accuracy on the linearly separable synthetic dataset. The learned hyperplanes differed slightly because my implementation optimizes the primal hinge-loss objective using gradient descent, whereas scikit-learn optimizes the SVM objective with a dedicated quadratic optimization algorithm. Look at svm_comparison.png


## kernel SVM
Implemented via the dual formulation, solved using `cvxopt`'s QP solver. Supports linear, polynomial, and RBF kernels. Enabling non-linear decision boundaries that the linear implementation cannot produce. Look at kernel_svm_boundary.png


## kernel comparison
Even on linearly separable data, different kernels can produce different decision boundaries and numbers of support vectors. RBF creates more flexible curved boundaries and uses more support vectors than the simpler kernels, while a polynomial kernel of low effective complexity naturally converges toward something close to the true linear-optimal boundary,closely matching scikit-learn's optimized SVM solution. Look at kernel_comparison.png

# Part 2:
## breast cancer classification 
Applied linear svm to the Wisconsin Breast Cancer dataset, with feature scaling and 80/20 training/testing split. 

## confusion matrix 
Out of 40 actual malignant cases in the test set, the model correctly identified 38 and missed 2. The model achieved 95% recall for malignant tumors, the class where false negatives are the most clinically significant. A model optimized only for overall accuracy may still miss more malignant cases. Look at confusion_matrix.png

## comparison on breast cancer data
On this particular train or test split, my implementation achieved marginally higher test accuracy than sklearn's SVC (by tenth of percent). This difference is not evidence that my implementation is superior though. Both models achieve comparable, strong performance, and the small gap likely reflects normal variance from a single data split rather than a meaningful difference in solution quality.

## many random states comparisons 
Across 6 different train and test splits, my from-scratch linear SVM achieved a mean test accuracy of 97.22%, compared to 96.93% for scikit-learn's SVC. The two implementations produced identical accuracy on 4 of 6 splits. The small remaining differences are consistent with normal variation across train/test splits and minor optimization differences between gradient descent and scikit-learn's optimization routine

# Part 3:
## mnist classification comparison 
Extended the binary svm to handwritten digit recognition. (training 10 binary SVMs, one per digit, and selecting the highest-confidence prediction at inference time)
I used a One-vs-Rest strategy, which performed slightly worse than sklearn's default One-vs-One approach. My implementation achieved 96.67% test accuracy, compared to 97.50% for sklearn's SVC. This gap likely comes from two factors: One-vs-One tends to outperform One-vs-Rest since each binary model only has to separate two specific digits rather than one digit from all nine others, and LIBSVM's optimized solver converges more precisely than my gradient-descent approach.

# findings 
Gradient descent optimizes the SVM objective iteratively and typically converges to a large-margin solution, but not necessarily the same maximum-margin solution found by specialized SVM optimization algorithms. This gap is small on easy binary problems and compounds as problem difficulty increases.  
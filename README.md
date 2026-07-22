# svm
SVM implemintation from scratch and real data testing 

# kernel comparison
On linearly-separable data, kernel choice matters even when it 'shouldn't'. RBF actively curves around clusters, using far more support vectors than necessary, while a polynomial kernel of low effective complexity naturally converges toward something close to the true linear-optimal boundary, closely matching sklearn's exact solution.

# confusion matrix aka precision 
Out of 43 actual malignant cases in the test set, the model correctly identified 41 and missed 2. A 95% recall on the class where false negatives carry the highest risk. This highlights a key limitation: a model optimized purely for overall accuracy may not minimize the error type that matters most in a clinical setting.

# comparison on breast cancer data
On this particular train or test split, my implementation achieved marginally higher test accuracy than sklearn's SVC (by tenth of percent). This difference is not evidence that my implementation is superior though. Both models achieve comparable, strong performance, and the small gap likely reflects normal variance from a single data split rather than a meaningful difference in solution quality.

# many random states comparisons 
Across 6 different train and test splits, my from-scratch linear SVM achieved a mean test accuracy of 97.22%, compared to 96.93% for scikit-learn's SVC. The two implementations agreed exactly on 4 of 6 splits, with the small remaining differences attributable to normal variance between gradient-descent and exact QP optimization, rather than a meaningful difference in model quality.

# mnist classification comparison 
I used a One-vs-Rest strategy, which performed slightly worse than sklearn's default One-vs-One approach. My implementation achieved 96.67% test accuracy, compared to 97.50% for sklearn's SVC. This gap likely comes from two factors: One-vs-One tends to outperform One-vs-Rest since each binary model only has to separate two specific digits rather than one digit from all nine others, and sklearn's exact QP solver converges more precisely than my gradient-descent approach — a pattern seen consistently throughout this project.
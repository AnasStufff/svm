import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from svm_lib.linear_svm import train_svm,predict


data=load_digits()
X=data.data
y=data.target
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Clases: {np.unique(y)}")
plt.figure(figsize=(3, 3))
plt.imshow(data.images[0], cmap='gray')
plt.title(f"Label: {y[0]}")
plt.savefig("sample_digit.png")
plt.show()

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2, random_state=42)
models={}
for digit in range(10):
    y_train_binary=np.where(y_train==digit,1,-1)
    w,b=train_svm(X_train,y_train_binary, C=1.0, lr=0.001, epochs=200)
    models[digit]=(w,b)
    
def predict_multiclass(X, models):
    n_samples=X.shape[0]
    scores=np.zeros((n_samples,10))
    for digit,(w,b) in models.items():
        scores[:,digit]=X @ w + b
    predictions=np.argmax(scores,axis=1)
    return predictions

train_p=predict_multiclass(X_train,models)
test_p=predict_multiclass(X_test,models)
train_acc=np.mean(train_p==y_train)
test_acc=np.mean(test_p==y_test)
        
print(f"Training accuracy: {train_acc:.2%}")
print(f"Test accuracy: {test_acc:.2%}")
        
from sklearn.svm import SVC

sklearn_model=SVC(kernel="linear", C=1.0)
sklearn_model.fit(X_train, y_train)
sklearn_train_acc=sklearn_model.score(X_train, y_train)
sklearn_test_acc=sklearn_model.score(X_test,y_test)
print(f"Comparison: My One-vs-Rest SVM vs sklearn")
print(f"{'Model':<15}{'Train Acc':<12}{'Test Acc'}")
print(f"{'Mine':<15}{train_acc:<12.2%}{test_acc:.2%}")
print(f"{'sklearn':<15}{sklearn_train_acc:<12.2%}{sklearn_test_acc:.2%}")
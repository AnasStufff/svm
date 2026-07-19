import numpy as np
from sklearn.datasets import load_breast_cancer 
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from svm_lib.linear_svm import train_svm, predict
from sklearn.metrics import confusion_matrix, classification_report 
import matplotlib.pyplot as plt
import seaborn as sns

data= load_breast_cancer()
X=data.data 
y=data.target 
feature_names=data.feature_names
print(f"Number of samples: {X.shape[0]}")
print(f"Number of feature: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"Clases: {data.target_names}")
print(f"Class balance: {np.bincount(y)}")
df=pd.DataFrame(X, columns=feature_names)
df["diagnosis"]=y
print(df.head())
print(df.describe())

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print("Before scaling mean radius range:", round(X[:, 0].min(), 2), "to", round(X[:, 0].max(),2))
print("After scaling mean radius range:", round(X_scaled[:, 0].min(),2), "to", round(X_scaled[:, 0].max(),2))

X_train, X_test, y_train, y_test= train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

y_train_svm=np.where(y_train==0,-1,1)
y_test_svm=np.where(y_test==0,-1,1)
w,b =train_svm(X_train, y_train_svm, C=1.0, lr=0.001, epochs=1000)
train_preds=predict(X_train, w,b)
test_preds=predict(X_test,w,b)
train_accuracy=np.mean(train_preds==y_train_svm)
test_accuracy=np.mean(test_preds==y_test_svm)
print(f"Training accuracy: {train_accuracy:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")

cm=confusion_matrix(y_test_svm, test_preds)
print("Confusion matrix: ")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test_svm, test_preds, target_names=["malignant", "benign"]))

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["malignant", "benign"], yticklabels= ["malignant", "benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Breast cancer classification (confusion matrix)")
plt.savefig("confusion_matrix.png")
plt.show()


#for comparison
from sklearn.svm import SVC
sklearn_model=SVC(kernel="linear", C=1.0)
sklearn_model.fit(X_train, y_train_svm)
sklearn_train_preds=sklearn_model.predict(X_train)
sklearn_test_preds=sklearn_model.predict(X_test)
sklearn_train_acc=np.mean(sklearn_train_preds==y_train_svm)
sklearn_test_acc=np.mean(sklearn_test_preds==y_test_svm)
print("Comparison")
print(f"{'Model':<15}{'Train Acc':<12}{'Test Acc'}")
print(f"{'Mine':<15}{train_accuracy:<12.2%}{test_accuracy:.2%}")
print(f"{'sklearn':<15}{sklearn_train_acc:<12.2%}{sklearn_test_acc:.2%}")


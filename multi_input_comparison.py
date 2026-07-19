import numpy as np
from sklearn.datasets import load_breast_cancer 
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from svm_lib.linear_svm import train_svm, predict
from sklearn.metrics import confusion_matrix, classification_report 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

# copied from breast_cancer_analysis.py
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

inputs=[0,7,42,100,1000,2026]
my_acc=[]
sklearn_acc=[]
for input in inputs:
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=input)
    y_tr_svm = np.where(y_tr == 0, -1, 1)
    y_te_svm = np.where(y_te == 0, -1, 1)
    w, b =train_svm(X_tr, y_tr_svm, C=1.0, lr=0.001, epochs=1000)
    my_ac = np.mean(predict(X_te, w, b) == y_te_svm)
    my_acc.append(my_ac)
    sk_model=SVC(kernel="linear", C=1.0)
    sk_model.fit(X_tr, y_tr_svm)
    sk_ac=np.mean(sk_model.predict(X_te)==y_te_svm)
    sklearn_acc.append(sk_ac)
    
print(f" Multi-input comparison.")
print(f"My SVM mean: {np.mean(my_acc):.2%}, per-seed: {[f'{a:.2%}' for a in my_acc]}")
print(f"sklearn mean: {np.mean(sklearn_acc):.2%}, per-seed: {[f'{a:.2%}' for a in sklearn_acc]}")
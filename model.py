import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Determine dataset path
dataset_filename = "Heart_disease_dataset.csv"
if os.path.exists(dataset_filename):
    dataset_path = dataset_filename
else:
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_filename)

data = pd.read_csv(dataset_path)

print(data.head())
print("Original shape:", data.shape)
print("Columns:", data.columns.tolist())

# Remove duplicate rows
print("Duplicate rows:", data.duplicated().sum())
data = data.drop_duplicates()
print("Shape after removing duplicates:", data.shape)

# Features and target
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=15,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Baseline SVM (RBF Kernel)
modelsvm = SVC(kernel='rbf')
modelsvm.fit(xtrain, ytrain)
ypredsvm = modelsvm.predict(xtest)

# Evaluation
print("\n===== BASELINE SVM (RBF) =====")
print("Accuracy :", accuracy_score(ytest, ypredsvm) * 100)
print("Precision:", precision_score(ytest, ypredsvm) * 100)
print("Recall   :", recall_score(ytest, ypredsvm) * 100)

cm = confusion_matrix(ytest, ypredsvm)
print("Confusion Matrix:\n", cm)

# Compare kernels
print("\n===== KERNEL COMPARISON =====")
kernels = ["linear", "poly", "rbf"]
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(xtrain, ytrain)
    prediction = model.predict(xtest)
    print(f"\nKernel: {kernel}")
    print("Accuracy :", accuracy_score(ytest, prediction) * 100)
    print("Precision:", precision_score(ytest, prediction) * 100)
    print("Recall   :", recall_score(ytest, prediction) * 100)
    print("F1 Score :", f1_score(ytest, prediction) * 100)

# Hyperparameter tuning with GridSearchCV
print("\n===== HYPERPARAMETER TUNING (GridSearchCV) =====")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(xtrain, ytrain)
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_ * 100)

best_model = grid.best_estimator_
ypred_tuned = best_model.predict(xtest)

print("\n===== TUNED SVM RESULTS =====")
print("Accuracy :", accuracy_score(ytest, ypred_tuned) * 100)
print("Precision:", precision_score(ytest, ypred_tuned) * 100)
print("Recall   :", recall_score(ytest, ypred_tuned) * 100)
print("F1 Score :", f1_score(ytest, ypred_tuned) * 100)

# Save artifacts
joblib.dump(best_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and scaler saved successfully to 'svm_model.pkl' and 'scaler.pkl'!")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RAW_PATH = "dataset/raw.csv"

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

def _cap_outliers_iqr(X, cols: list):
    X = X.copy()
    for col in cols:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        X[col] = X[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return X


def _scale(X, cols: list):
    X = X.copy()
    X[cols] = StandardScaler().fit_transform(X[cols])
    return X


def _ohe(X, cols: list):
    return pd.get_dummies(X, columns=cols, drop_first=False)


def _split_rejoin(df, transform):
    X = df.drop(TARGET, axis=1).copy()
    y = df[TARGET].reset_index(drop=True)
    X = transform(X)
    return pd.concat([X.reset_index(drop=True), y], axis=1)


# KNN
# Distance-based: scale continuous, one-hot encode categorical, cap outliers.
def preprocess_knn(df):
    def transform(X):
        X = _cap_outliers_iqr(X, CONTINUOUS)
        X = _ohe(X, CATEGORICAL)
        X = _scale(X, CONTINUOUS)
        return X
    return _split_rejoin(df, transform)


# Logistic Regression 
# Scale continuous, one-hot encode categorical.
def preprocess_logistic_regression(df):
    def transform(X):
        X = _ohe(X, CATEGORICAL)
        X = _scale(X, CONTINUOUS)
        return X
    return _split_rejoin(df, transform)


# Random Forest 
# skip numeric ordinal encoding
def preprocess_random_forest(df):
    return df.copy()


# Decision Tree 
# skip preprocessing
def preprocess_decision_tree(df):
    return df.copy()


# SVM 
# Kernel-based: very sensitive to feature scale and outliers.
# Cap outliers, one-hot encode categorical, scale continuous features.
def preprocess_svm(df):
    def transform(X):
        X = _cap_outliers_iqr(X, CONTINUOUS)
        X = _ohe(X, CATEGORICAL)
        X = _scale(X, CONTINUOUS)
        return X
    return _split_rejoin(df, transform)


PREPROCESSORS = {
    "knn": preprocess_knn,
    "logistic_regression": preprocess_logistic_regression,
    "random_forest": preprocess_random_forest,
    "decision_tree": preprocess_decision_tree,
    "svm": preprocess_svm,
}

if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    print(f"Raw data: {df.shape[0]} rows x {df.shape[1]} cols\n")

    for name, fn in PREPROCESSORS.items():
        out = fn(df)
        path = f"dataset/processed_{name}.csv"
        out.to_csv(path, index=False)

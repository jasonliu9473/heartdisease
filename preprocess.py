import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

class IQRCapper(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Clip each column to [Q1 - factor*IQR, Q3 + factor*IQR].
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        Q1, Q3 = np.nanpercentile(X, [25, 75], axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


def _numeric(cap: bool):
    steps = [("cap", IQRCapper())] if cap else []
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _column_transformer(cap: bool):
    return ColumnTransformer([
        ("num", _numeric(cap), CONTINUOUS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])

#   KNN / SVM          : IQR cap + one-hot + scale
#   LogisticRegression : one-hot + scale
#   RandomForest / Tree: keep ordinal integer encodings, no scaling
def build_preprocessor(model_name: str):
    if model_name in ("knn", "svm"):
        return _column_transformer(cap=True)
    if model_name == "logistic_regression":
        return _column_transformer(cap=False)
    if model_name in ("random_forest", "decision_tree"):
        return "passthrough"
    raise ValueError(f"unknown model_name: {model_name!r}")


def build_pipeline(model_name: str, clf):
    return Pipeline([("prep", build_preprocessor(model_name)), ("clf", clf)])

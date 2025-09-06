from __future__ import annotations
from sklearn.linear_model import LogisticRegression

def train_logreg(X, y):
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

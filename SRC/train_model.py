# ================================
# train_model.py
# ================================


import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_decision_tree(X_train, y_train, max_depth=5):

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train, n_estimators=100):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model


def tune_random_forest(X_train, y_train):

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5]
    }

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_


def save_model(model, path):

    joblib.dump(model, path)

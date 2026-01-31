# ================================
# feature_engineering.py
# ================================


import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_features(df, target="Attrition"):
    """
    Prepare ML features and split data
    """

    X = df.drop(columns=[target])
    y = df[target]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

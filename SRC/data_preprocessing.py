# ================================
# data_preprocessing.py
# ================================

import pandas as pd


def load_and_clean_data(path):
    """
    Load and clean the dataset
    """

    df = pd.read_csv(path)

    # Drop unnecessary columns
    drop_cols = [
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "EmployeeNumber"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode target variable
    df["Attrition"] = df["Attrition"].map({
        "No": 0,
        "Yes": 1
    })

    return df

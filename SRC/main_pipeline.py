# ================================
# main_pipeline.py (Optional Runner)
# ================================


import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))


from data_preprocessing import load_and_clean_data
from feature_engineering import prepare_features
from train_model import tune_random_forest, save_model
from evaluate_model import evaluate_model


def main():

    # File path (change if needed)
    DATA_PATH = "data/raw/employee_attrition.csv"
    MODEL_PATH = "models/final_model.pkl"

    print("Loading data...")
    df = load_and_clean_data(DATA_PATH)

    print("Preparing features...")
    X_train, X_test, y_train, y_test = prepare_features(df)

    print("Training model...")
    model, params = tune_random_forest(X_train, y_train)

    print("Best Parameters:")
    print(params)

    print("Evaluating model...")
    metrics, cm, report = evaluate_model(model, X_test, y_test)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    print("Saving model...")
    save_model(model, MODEL_PATH)

    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    main()

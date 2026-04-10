import pandas as pd
import os
from src.data_cleaning import clean_pipeline
from src.feature_engineering import feature_engineering
from src.model_training2 import prepare_data, train_models

# Load data
df = pd.read_csv("dataset/dataset.csv")   # Adjust path

# 1. Cleaning
df = clean_pipeline(df)

os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned.csv", index=False)

# 2. Feature Engineering
df = feature_engineering(df)

# 3. Split
X_train, X_test, y_train, y_test = prepare_data(df)

# 4. Train
results, trained_models = train_models(X_train, X_test, y_train, y_test)

# Save results
os.makedirs("results", exist_ok=True)
with open("results/result.txt", "w") as f:
    f.write("Model Performance Results\n")
    f.write("=========================\n\n")
    for model_name, metrics in results.items():
        f.write(f"{model_name}\n")
        f.write("-------------------------\n")
        f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall   : {metrics['recall']:.4f}\n")
        f.write(f"F1 Score : {metrics['f1_score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + "\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'] + "\n")
        f.write("====================================\n\n")

print("Pipeline completed. Results saved to results/result.txt")
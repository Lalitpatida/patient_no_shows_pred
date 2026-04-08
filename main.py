import pandas as pd
import os
from src.data_cleaning import clean_pipeline
from src.feature_engineering import feature_engineering
from src.model_training import prepare_data, train_models
from src.feature_importance import compute_feature_importance, save_feature_importance_plot


df = pd.read_csv("C:/Users/developer/Desktop/projects/apt_no_shows/dataset/dataset.csv")

# Step 1: Cleaning
df = clean_pipeline(df)

os.makedirs("data", exist_ok=True)
df.to_csv("data/final.csv", index=False)
os.makedirs("results", exist_ok=True)

# Step 2: Feature Engineering
df = feature_engineering(df)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = prepare_data(df)

# Step 4: Handle imbalance
#X_res, y_res = handle_imbalance(X_train, y_train)

# Step 5: Train models
#results = train_models(X_res, X_test, y_res, y_test)
results, trained_models  = train_models(X_train, X_test, y_train, y_test)
feature_names = df.drop('No_show', axis=1).columns

for model_name, model in trained_models.items():
    importance_dict = compute_feature_importance(model, feature_names)
    save_feature_importance_plot(importance_dict, model_name)

print("\nFinal Results:")
for model, metrics in results.items():
    print(f"\n{model}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")


with open("results/result.txt", "w") as f:
    f.write("Model Performance Results\n")
    f.write("=========================\n\n")

    for model, metrics in results.items():
        f.write(f"{model}\n")
        f.write("-------------------------\n")
        f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall   : {metrics['recall']:.4f}\n")
        f.write(f"F1 Score : {metrics['f1_score']:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"{metrics['confusion_matrix']}\n\n")

        f.write("Classification Report:\n")
        f.write(f"{metrics['classification_report']}\n")
        f.write("====================================\n\n")

print("Results saved to results/result.txt")
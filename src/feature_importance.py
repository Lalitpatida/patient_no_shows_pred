import numpy as np
import matplotlib.pyplot as plt
import os

def compute_feature_importance(model, feature_names, top_n=10):
    """
    Compute top_n feature importance for a model.
    Returns a dict of all features importance.
    """
    importance_dict = {}

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        sorted_idx = np.argsort(fi)[::-1]
        for i in sorted_idx[:top_n]:
            importance_dict[feature_names[i]] = fi[i]

    elif hasattr(model, "coef_"):
        coefs = model.coef_[0]
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        for i in sorted_idx[:top_n]:
            importance_dict[feature_names[i]] = coefs[i]

    else:
        print("Model does not support feature importance")
        return None

    return importance_dict


def save_feature_importance_plot(importance_dict, model_name, save_dir="results/feature_importance", top_n=10):
    """
    Save top_n feature importance as a horizontal bar plot
    """
    if importance_dict is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    sorted_items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, values = zip(*sorted_items)

    plt.figure(figsize=(10, 6))
    plt.barh(features[::-1], values[::-1], color="skyblue")
    plt.xlabel("Importance")
    plt.title(f"{model_name} Feature Importance")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_feature_importance.png")
    plt.savefig(save_path)
    plt.close()  # close figure to avoid display
    print(f"Saved feature importance plot for {model_name} at {save_path}")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def save_combined_roc_auc_plot(models_dict, X_test, y_test, save_dir="results/roc_auc"):
    """
    Plot ROC curves of ALL models in ONE figure with AUC scores in legend.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # For models without predict_proba (rare), fallback to decision function or skip
            print(f"Skipping {name} - no predict_proba")
            continue
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Plot the random classifier line
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curves - All Models Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "all_models_roc_auc.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined ROC-AUC plot at: {save_path}")
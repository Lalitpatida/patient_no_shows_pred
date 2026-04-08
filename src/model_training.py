import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -------------------- Train-Test Split --------------------
def prepare_data(df):
    X = df.drop(columns=['No_show'])
    y = df['No_show']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------- Handle Imbalance --------------------
def handle_imbalance(X_train, y_train):
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Before:", np.bincount(y_train))
    print("After :", np.bincount(y_res))
    return X_res, y_res

# -------------------- Hyperparameter Tuning --------------------
def tune_models(X_train, y_train):
    print("\nStarting Hyperparameter Tuning...\n")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # -------------------- RandomForest --------------------
    rf_params = {
        "n_estimators": [200, 500, 700, 1000],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ['sqrt', 'log2', None]
    }

    rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
    rf_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=10,
        scoring='f1',      # balance precision & recall
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    print("Best RandomForest Params:", rf_search.best_params_)

    # -------------------- XGBoost --------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_params = {
        "n_estimators": [300, 500, 700, 1000],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [1, 1.5, 2]
    }

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )

    xgb_search = RandomizedSearchCV(
        xgb,
        xgb_params,
        n_iter=10,
        scoring='f1',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    print("Best XGBoost Params:", xgb_search.best_params_)

    return rf_search.best_estimator_, xgb_search.best_estimator_

# -------------------- Find Optimal Threshold --------------------
def find_best_threshold(model, X_val, y_val):
    """Find threshold that maximizes F1 score"""
    probs = model.predict_proba(X_val)[:, 1]
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

# -------------------- Train Models --------------------
def train_models(X_train, X_test, y_train, y_test):
    results = {}

    # Step 1: Tune models
    tuned_rf, tuned_xgb = tune_models(X_train, y_train)

    # Step 2: Define all models
    models = {
        "RandomForest_Tuned": tuned_rf,
        "XGBoost_Tuned": tuned_xgb,
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, class_weight='balanced')
    }

    # Step 3: Train & Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Apply probability threshold tuning for models that support it
        if hasattr(model, "predict_proba"):
            best_thresh = find_best_threshold(model, X_train, y_train)
            print(f"{name} Best Threshold: {best_thresh:.2f}")
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs > best_thresh).astype(int)
        else:
            preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"\n{name} Results:")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        results[name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(y_test, preds),
            "classification_report": classification_report(y_test, preds),
            "best_threshold": best_thresh if hasattr(model, "predict_proba") else None
        }

    return results, models




# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
#     confusion_matrix
# )
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# from xgboost import XGBClassifier

# # -------------------- Train-Test Split --------------------
# def prepare_data(df):
#     X = df.drop(columns=['No_show'])
#     y = df['No_show']
#     return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # -------------------- Hybrid Sampling --------------------
# def balance_data(X_train, y_train, smote_ratio=0.5, undersample_ratio=0.8):
#     """Apply SMOTE + random undersampling to balance classes"""
#     print("Original class distribution:", np.bincount(y_train))

#     smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
#     X_res, y_res = smote.fit_resample(X_train, y_train)
#     print("After SMOTE:", np.bincount(y_res))

#     undersample = RandomUnderSampler(sampling_strategy=undersample_ratio, random_state=42)
#     X_res, y_res = undersample.fit_resample(X_res, y_res)
#     print("After Undersampling:", np.bincount(y_res))

#     return X_res, y_res

# # -------------------- Hyperparameter Tuning --------------------
# def tune_models(X_train, y_train):
#     print("\nStarting Hyperparameter Tuning...\n")
#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#     # RandomForest
#     rf_params = {
#         "n_estimators": [200, 500, 700, 1000],
#         "max_depth": [5, 10, 15, 20],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4],
#         "max_features": ['sqrt', 'log2', None]
#     }
#     rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
#     rf_search = RandomizedSearchCV(rf, rf_params, n_iter=10, scoring='f1', cv=cv, verbose=1, n_jobs=-1)
#     rf_search.fit(X_train, y_train)
#     print("Best RandomForest Params:", rf_search.best_params_)

#     # XGBoost
#     scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
#     xgb_params = {
#         "n_estimators": [300, 500, 700, 1000],
#         "max_depth": [4, 6, 8, 10],
#         "learning_rate": [0.01, 0.05, 0.1],
#         "subsample": [0.7, 0.8, 1.0],
#         "colsample_bytree": [0.7, 0.8, 1.0],
#         "gamma": [0, 0.1, 0.2],
#         "reg_alpha": [0, 0.01, 0.1],
#         "reg_lambda": [1, 1.5, 2]
#     }
#     xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42)
#     xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=10, scoring='f1', cv=cv, verbose=1, n_jobs=-1)
#     xgb_search.fit(X_train, y_train)
#     print("Best XGBoost Params:", xgb_search.best_params_)

#     return rf_search.best_estimator_, xgb_search.best_estimator_

# # -------------------- Threshold Tuning --------------------
# def find_best_threshold(model, X_val, y_val):
#     probs = model.predict_proba(X_val)[:, 1]
#     best_f1 = 0
#     best_thresh = 0.5
#     for thresh in np.arange(0.1, 0.9, 0.01):
#         preds = (probs > thresh).astype(int)
#         f1 = f1_score(y_val, preds)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thresh = thresh
#     return best_thresh

# # -------------------- Train Models with Stacking --------------------
# def train_models(X_train, X_test, y_train, y_test):
#     results = {}

#     # Step 1: Balance data for all models
#     X_bal, y_bal = balance_data(X_train, y_train)

#     # Step 2: Tune base models
#     tuned_rf, tuned_xgb = tune_models(X_bal, y_bal)

#     # Step 3: Base models for stacking
#     base_models = [
#         ('rf', tuned_rf),
#         ('xgb', tuned_xgb),
#         ('lr', LogisticRegression(max_iter=1000, class_weight='balanced'))
#     ]
#     stack_model = StackingClassifier(
#         estimators=base_models,
#         final_estimator=LogisticRegression(max_iter=1000),
#         cv=3,
#         n_jobs=-1,
#         passthrough=True
#     )

#     # Step 4: Train all models
#     models = {
#         "Stacked_Ensemble": stack_model,
#         "RandomForest_Tuned": tuned_rf,
#         "XGBoost_Tuned": tuned_xgb,
#         "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
#         "DecisionTree": DecisionTreeClassifier(max_depth=10, class_weight='balanced')
#     }

#     for name, model in models.items():
#         model.fit(X_bal, y_bal)

#         # Threshold optimization
#         if hasattr(model, "predict_proba"):
#             best_thresh = find_best_threshold(model, X_bal, y_bal)
#             print(f"{name} Best Threshold: {best_thresh:.2f}")
#             probs = model.predict_proba(X_test)[:, 1]
#             preds = (probs > best_thresh).astype(int)
#         else:
#             preds = model.predict(X_test)

#         acc = accuracy_score(y_test, preds)
#         precision = precision_score(y_test, preds)
#         recall = recall_score(y_test, preds)
#         f1 = f1_score(y_test, preds)

#         print(f"\n{name} Results:")
#         print(f"Accuracy : {acc:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall   : {recall:.4f}")
#         print(f"F1 Score : {f1:.4f}")
#         print("\nConfusion Matrix:")
#         print(confusion_matrix(y_test, preds))
#         print("\nClassification Report:")
#         print(classification_report(y_test, preds))

#         results[name] = {
#             "accuracy": acc,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1,
#             "confusion_matrix": confusion_matrix(y_test, preds),
#             "classification_report": classification_report(y_test, preds),
#             "best_threshold": best_thresh if hasattr(model, "predict_proba") else None
#         }

#     return results, models
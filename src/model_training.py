import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import fbeta_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from src.evaluation_plot import save_combined_roc_auc_plot



# -------------------- Train-Test Split --------------------
# def prepare_data(df, test_size=0.2, random_state=42):
#     X = df.drop(columns=['No_show'])
#     y = df['No_show']
#     return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)



def prepare_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['No_show'])
    y = df['No_show']
    groups = df['PatientId']   

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


# -------------------- Improved Balancing --------------------
def balance_data(X_train, y_train, method='smotetomek'):
    print("Original class distribution:", np.bincount(y_train))
    #sampler = SMOTE(random_state=42)
    if method == 'smotetomek':
        sampler = SMOTETomek(sampling_strategy= 'auto', random_state=42)
    else:
        sampler = SMOTE(sampling_strategy= 'auto', random_state=42)

    #X_res, y_res = sampler.fit_resample(X_train, y_train)
    #print("After balancing:", np.bincount(y_res))
    #return X_res, y_res
    return X_train, y_train

# -------------------- Threshold Tuning with F-beta (β=1.5 → favors recall) --------------------
def find_best_threshold(model, X_val, y_val, beta=1.5):
    probs = model.predict_proba(X_val)[:, 1]
    best_fbeta = 0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (probs > thresh).astype(int)
        fbeta = fbeta_score(y_val, preds, beta=beta)
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_thresh = thresh

    print(f"Best threshold (F-beta={beta}): {best_thresh:.3f} with F-beta={best_fbeta:.4f}")
    return best_thresh

# -------------------- Hyperparameter Tuning --------------------
def tune_models(X_train, y_train):
    print("\n=== Starting Hyperparameter Tuning ===\n")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # RandomForest
    # rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    # rf_params = {
    #     "n_estimators": [300, 600],
    #     "max_depth": [12, 18, 25],
    #     "min_samples_split": [4, 8],
    #     "min_samples_leaf": [2, 4],
    #     "max_features": ['sqrt', 0.8]
    # }
    # rf_search = RandomizedSearchCV(rf, rf_params, n_iter=12, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
    # rf_search.fit(X_train, y_train)

    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_params = {
        "n_estimators": [400, 700],
        "max_depth": [5, 7, 9],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.2],
        "reg_lambda": [1, 2]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=12, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
    xgb_search.fit(X_train, y_train)

    # # LightGBM
    # lgb = LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    # lgb_params = {
    #     "n_estimators": [400, 700],
    #     "max_depth": [8, 12, -1],
    #     "learning_rate": [0.02, 0.05, 0.1],
    #     "num_leaves": [31, 63, 127],
    #     "subsample": [0.8, 1.0],
    #     "colsample_bytree": [0.8, 1.0]
    # }
    # lgb_search = RandomizedSearchCV(lgb, lgb_params, n_iter=12, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
    # lgb_search.fit(X_train, y_train)

    #print("Best RF params:", rf_search.best_params_)
    print("Best XGB params:", xgb_search.best_params_)
    #print("Best LGB params:", lgb_search.best_params_)

    #return rf_search.best_estimator_, xgb_search.best_estimator_, lgb_search.best_estimator_
    return xgb_search.best_estimator_

# -------------------- Train & Evaluate --------------------
def train_models(X_train, X_test, y_train, y_test):
    results = {}

    X_train = X_train.drop(columns=['PatientId'])
    X_test = X_test.drop(columns=['PatientId'])

    

    # Balance data
    X_bal, y_bal = balance_data(X_train, y_train, method='smotetomek')

    # Tune models
    #tuned_rf, tuned_xgb, tuned_lgb = tune_models(X_bal, y_bal)
    tuned_xgb = tune_models(X_bal, y_bal)

    # Stacking
    base_models = [
        #('rf', tuned_rf),
        ('xgb', tuned_xgb),
        #('lgb', tuned_lgb),
        #('lr', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1))
    ]

    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=3,
        n_jobs=-1,
        passthrough=True
    )

    models = {
        #"Stacked_Ensemble": stack_model,
        "XGBoost_Tuned": tuned_xgb,
        #"LightGBM_Tuned": tuned_lgb,
        #"RandomForest_Tuned": tuned_rf,
        #"LogisticRegression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        #"DecisionTree": DecisionTreeClassifier(max_depth=12, class_weight='balanced')
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        # model.fit(X_bal, y_bal)

        # if hasattr(model, "predict_proba"):
        #     best_thresh = find_best_threshold(model, X_bal, y_bal, beta=1.5)
        #     probs = model.predict_proba(X_test)[:, 1]
        #     preds = (probs > best_thresh).astype(int)
        # else:
        #     preds = model.predict(X_test)
        #     best_thresh = None


        if hasattr(model, "predict_proba"):

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
            )

            model.fit(X_tr, y_tr)

            best_thresh = find_best_threshold(model, X_val, y_val, beta=1.5)

            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs > best_thresh).astype(int)

        else:
            model.fit(X_bal, y_bal)
            preds = model.predict(X_test)
            best_thresh = None

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"\n{name} Results:")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
            "confusion_matrix": confusion_matrix(y_test, preds),
            "classification_report": classification_report(y_test, preds),
            "best_threshold": best_thresh
        }

    # ==================== ADD THIS BLOCK ====================
    print("\n=== Generating Combined ROC-AUC Plot for All Models ===")
    save_combined_roc_auc_plot(models, X_test, y_test)
    # =======================================================


    return results, models

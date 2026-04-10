import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupShuffleSplit, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, fbeta_score
)
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.evaluation_plot import save_combined_roc_auc_plot


# -------------------- Train-Test Split --------------------
def prepare_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['No_show'])
    y = df['No_show']
    groups = df['PatientId']

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


# -------------------- Balancing --------------------
def balance_data(X_train, y_train, method='smotetomek'):
    print("Original class distribution:", np.bincount(y_train))

    if method == 'smotetomek':
        sampler = SMOTETomek(random_state=42)
    else:
        sampler = SMOTE(random_state=42)

    X_res, y_res = sampler.fit_resample(X_train, y_train)

    print("After balancing:", np.bincount(y_res))
    return X_res, y_res


# -------------------- Threshold Tuning --------------------
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

    print(f"Best threshold (F-beta={beta}): {best_thresh:.3f} | Score={best_fbeta:.4f}")
    return best_thresh


# -------------------- Hyperopt Tuning --------------------
def tune_xgboost(X_train, y_train):

    print("\n=== Hyperopt Tuning for XGBoost ===\n")

    best_score = 0
    iteration = 0

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(params):
        nonlocal best_score, iteration
        iteration += 1

        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            colsample_bytree=params['colsample_bytree'],
            subsample=params['subsample'],
            min_child_weight=int(params['min_child_weight']),
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=skf,
            scoring='roc_auc',
            n_jobs=-1
        ).mean()

        if score > best_score:
            best_score = score
            print(f" Iter {iteration} | Best ROC-AUC: {best_score:.4f}")

        return {'loss': -score, 'status': STATUS_OK}

    space = {
        'n_estimators': hp.quniform('n_estimators', 200, 800, 1),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.15),
        'gamma': hp.uniform('gamma', 0, 0.5),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 8, 1)
    }

    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    print("\nBest Params:", best)

    final_model = XGBClassifier(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        learning_rate=best['learning_rate'],
        gamma=best['gamma'],
        colsample_bytree=best['colsample_bytree'],
        subsample=best['subsample'],
        min_child_weight=int(best['min_child_weight']),
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_train, y_train)
    return final_model


# -------------------- Train & Evaluate --------------------
def train_models(X_train, X_test, y_train, y_test):

    # Remove leakage column
    X_train = X_train.drop(columns=['PatientId'])
    X_test = X_test.drop(columns=['PatientId'])

    # Balance dataset
    X_bal, y_bal = balance_data(X_train, y_train)

    # Train final tuned model
    model = tune_xgboost(X_bal, y_bal)

    print("\nTraining Final Model Split...")

    # Validation split for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_bal, y_bal,
        test_size=0.2,
        stratify=y_bal,
        random_state=42
    )

    model.fit(X_tr, y_tr)

    best_thresh = find_best_threshold(model, X_val, y_val)

    # Predictions
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > best_thresh).astype(int)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n=== Final XGBoost Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    #  FIXED STRUCTURE (IMPORTANT)
    results = {
        "XGBoost": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(y_test, preds),
            "classification_report": classification_report(y_test, preds),
            "best_threshold": best_thresh
        }
    }

    # ROC Curve
    print("\n=== Generating ROC-AUC Plot ===")
    save_combined_roc_auc_plot({"XGBoost": model}, X_test, y_test)

    return results, {"XGBoost": model}
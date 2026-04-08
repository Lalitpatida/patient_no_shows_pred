from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))
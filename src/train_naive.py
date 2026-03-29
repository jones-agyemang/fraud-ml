import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

def main():
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    print(f"Training samples: {len(train_df):,}")
    print(f"Testing samples: {len(test_df)}:,")
    print(f"Training fraud ratio: {train_df['is_fraud'].mean():.2%}")

    # Encode categorical feature
    print("\nEncoding categorical features")
    encoder = LabelEncoder()
    train_df['merchant_encoded'] = encoder.fit_transform(train_df["merchant_category"])
    test_df['merchant_encoded'] = encoder.transform(test_df["merchant_category"])

    print(f"Merchant category encoding: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    # Prepare features and labels
    feature_cols = ["amount", "hour", "day_of_week", "merchant_encoded"]
    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"]

    # Creating and fitting RandomForestClassifier
    print("Creating model(RandomForestClassifier)")
    model = RandomForestClassifier(
        criterion='gini',
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training... Complete!")
    print(model)

    # Evaluation
    print("\n" + "="*50)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"{y_pred.sum()}")
    print(y_proba)
    print("="*50)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall score: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score {f1_score(y_test, y_pred):.4f}")

    print("\nConfusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"(True Negative) Rightly identified as Legitimate: {tn}")
    print(f"(False Negative) Fraudulent, but predicted as Legitimate: {fp}")
    print(f"(False Positive) Legitimate, but predicted as Fraudulent: {fn}")
    print(f"(True Positive) Rightly identified as Fraudulent: {tp}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    # Feature importance
    for name, importance in sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{name}: {importance:.4f}")

    # save model and encoder
    print("\nSaving model to models/model.pkl...")
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, encoder), f)

    print("\nModel trained and saved successfully!")

if __name__=="__main__":
    main()
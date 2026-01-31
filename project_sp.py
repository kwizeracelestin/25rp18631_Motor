# =========================================================
# INTELLIGENT MOTOR FAULT DETECTION & HEALTH MONITORING
# Robust single-file version (Spyder safe)
# =========================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# FILE NAMES
# =========================================================
DATASET_FILE = "dataset.xlsx"
MODEL_FILE = "project_model.pkl"
SCALER_FILE = "project_scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"

# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset():
    df = pd.read_excel(DATASET_FILE)
    print("✅ Dataset loaded")
    return df

# =========================================================
# TRAIN MODEL
# =========================================================
def train_and_save_model(df):

    # --- Detect label column automatically ---
    possible_labels = ["Fault_Label", "fault", "label", "Condition", "condition"]
    label_col = None

    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError("❌ No fault label column found in dataset")

    print(f"🔍 Using label column: {label_col}")

    X = df.drop(label_col, axis=1)
    y = df[label_col]

    # --- Encode labels if text ---
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Scaling ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Model ---
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- Save everything ---
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print("💾 Model, scaler & encoder saved")

# =========================================================
# TEST PREDICTION
# =========================================================
def test_prediction(df):

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)

    feature_names = df.drop(df.columns[-1], axis=1).columns.tolist()

    # Example sample (same order as dataset)
    sample_values = [df[feature].mean() for feature in feature_names]

    sample = pd.DataFrame([sample_values], columns=feature_names)
    sample_scaled = scaler.transform(sample)

    pred = model.predict(sample_scaled)
    label = encoder.inverse_transform(pred)[0]

    print("\n🔍 SAMPLE PREDICTION RESULT")
    print("Motor Condition:", label)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("🚀 Starting Motor Fault Detection System\n")

    df = load_dataset()

    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
        train_and_save_model(df)
    else:
        print("ℹ Existing model found, skipping training")

    test_prediction(df)

    print("\n✅ System running successfully (Spyder)")
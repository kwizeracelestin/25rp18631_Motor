# =========================================================
# INTELLIGENT MOTOR FAULT DETECTION & HEALTH MONITORING
# Streamlit-safe & Robust version
# =========================================================

import os
import pandas as pd
# ---- Safe joblib import (important for Streamlit) ----
try:
    import joblib
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "❌ joblib is not installed. "
        "Add 'joblib' to requirements.txt and redeploy the app."
    )

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
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"❌ Dataset not found: {DATASET_FILE}")

    df = pd.read_excel(DATASET_FILE, engine="openpyxl")
    print("✅ Dataset loaded successfully")
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
        raise ValueError("❌ No fault/condition label column found in dataset")

    print(f"🔍 Using label column: {label_col}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # --- Encode labels ---
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # --- Save model artifacts ---
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print("💾 Model, scaler & encoder saved successfully")

# =========================================================
# TEST PREDICTION
# =========================================================
def test_prediction(df):

    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, ENCODER_FILE]):
        raise FileNotFoundError("❌ Model files not found. Train the model first.")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)

    # Use same features as training
    feature_cols = df.drop(df.columns[-1], axis=1).columns.tolist()

    # Example sample: mean of each feature
    sample = pd.DataFrame(
        [df[feature_cols].mean().values],
        columns=feature_cols
    )

    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)
    label = encoder.inverse_transform(pred)[0]

    print("\n🔍 SAMPLE PREDICTION RESULT")
    print("⚙️ Motor Condition:", label)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("🚀 Starting Motor Fault Detection System\n")

    df = load_dataset()

    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
        train_and_save_model(df)
    else:
        print("ℹ️ Existing model found — skipping training")

    test_prediction(df)

    print("\n✅ System executed successfully")

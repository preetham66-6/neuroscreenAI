import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.stdout.reconfigure(line_buffering=True)

# SECTION 1: Load Dataset
DATA_PATH = os.path.join("data", "parkinsons.csv")
print("[INFO] Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("[OK] Dataset loaded! Shape:", df.shape[0], "rows x", df.shape[1], "columns")

# SECTION 2: Missing Values
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("[OK] No missing values found.")
else:
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print("[WARNING] Filled missing values with column mean.")

# SECTION 3: Drop 'name' column
if "name" in df.columns:
    df.drop(columns=["name"], inplace=True)
    print("[INFO] Dropped 'name' column.")

# SECTION 4: DROP THE 6 NONLINEAR FEATURES
NONLINEAR_FEATURES = ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
cols_to_drop = [c for c in NONLINEAR_FEATURES if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[INFO] Dropped {len(cols_to_drop)} nonlinear features: {cols_to_drop}")
else:
    print("[WARNING] Nonlinear columns not found!")

print(f"[INFO] Remaining: {df.shape[1]} columns (should be 17 = 16 features + status)")

# SECTION 5: X and y
X = df.drop(columns=["status"])
y = df["status"]
print("[INFO] Features:", X.shape[1], "columns:", list(X.columns))

# SECTION 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("[INFO] Train:", X_train.shape[0], "| Test:", X_test.shape[0])

# SECTION 7: Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("[OK] Scaling done.")

# SECTION 8: Save
os.makedirs("models", exist_ok=True)
joblib.dump(X_train_scaled,  os.path.join("models", "X_train.pkl"))
joblib.dump(X_test_scaled,   os.path.join("models", "X_test.pkl"))
joblib.dump(y_train,         os.path.join("models", "y_train.pkl"))
joblib.dump(y_test,          os.path.join("models", "y_test.pkl"))
joblib.dump(scaler,          os.path.join("models", "scaler.pkl"))
joblib.dump(list(X.columns), os.path.join("models", "feature_names.pkl"))
print("[OK] All files saved to models/")
print("X_train shape:", X_train_scaled.shape, " <-- should be (156, 16)")
print("DONE. Run 3_train_models.py next.")